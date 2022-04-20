import argparse
import glob
import logging
import os
import sys
from collections import namedtuple
from collections.abc import MutableMapping

from librosa import cqt
import h5py
import numpy as np
import pandas as pd
from pycbc.waveform import get_td_waveform
from scipy import interpolate
from joblib import Parallel, delayed
from pycbc.distributions.angular import SinAngle
from tqdm import tqdm

from scipy.signal.windows import tukey

from processing import ReShape

def start_logger_if_necessary():
	logger = logging.getLogger("mylogger")
	if len(logger.handlers) == 0:
		logger.setLevel(logging.INFO)

		fh = logging.FileHandler('generate.log')
		fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
		logger.addHandler(fh)

		sh = logging.StreamHandler()
		sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
		sh.setLevel(logging.WARNING)
		logger.addHandler(sh)

	return logger


logger = start_logger_if_necessary()


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.') -> MutableMapping:
	items = []
	for k, v in d.items():
		new_key = parent_key + sep + k if parent_key else k
		if isinstance(v, MutableMapping):
			items.extend(flatten_dict(v, new_key, sep=sep).items())
		else:
			items.append((new_key, v))
	return dict(items)

Spectrogram = namedtuple("Spectrogram", ["data", "metadata"])

class RandomModel:
	def __init__(self, mmin: float = 1.0, mmax: float = 2.0, lmin: float = 0.0, lmax: float = 5000.0):
		self.mmin = mmin
		self.mmax = mmax
		self.lmin = lmin
		self.lmax = lmax

	def random_config(self) -> dict:
		rng = np.random.default_rng()

		mass1 = rng.uniform(self.mmin, self.mmax)
		mass2 = rng.uniform(self.mmin, mass1)

		lambda1 = rng.uniform(self.lmin, self.lmax)
		lambda2 = rng.uniform(self.lmin, lambda1)

		config = {
			'mass1': mass1,
			'mass2': mass2,
			'lambda1': lambda1,
			'lambda2': lambda2,
		}
		return config


class Models:
	def __init__(self, model_name: str, data_file: str = "models.csv",
				 mmin: float = 1.0, mmax: float = 2.0):

		data = pd.read_csv(data_file)
		assert all(col in ["m", "lambda", "model"] for col in data.columns), \
			"Models file doesn't contain the required columns (m, lambda, eos)."

		self.model_name = model_name
		self.data = data[data["model"] == model_name]
		if self.data.empty:
			raise Exception(f"No available data for model with name \'{model_name}\'.")

		self.mmin = mmin
		self.mmax = mmax

		self.rng = np.random.default_rng()

		self.inclination_dist = SinAngle(inclination=None)
        
	@property
	def data(self):
		return self._data

	@data.setter
	def data(self, val):
		if val.empty:
			raise Exception(f"No available data for model with name \'{self.model_name}\'.")
		self._data = val
		self._interpolator = interpolate.interp1d(x=self._data["m"], y=self._data["lambda"], kind="linear")

	@property
	def mass2lambda(self):
		return self._interpolator

	def _error(self, mass: float, lamb: float):
		return np.random.normal(loc=0, scale=lamb*0.1)

	def random_config(self, include_error: bool = False) -> dict:

		mass1 = float(self.rng.uniform(self.mmin, self.mmax))
		mass2 = float(self.rng.uniform(self.mmin, self.mmax))

		lambda1 = float(self.mass2lambda(mass1))
		lambda2 = float(self.mass2lambda(mass2))

		inclination = float(self.inclination_dist.rvs()["inclination"])

		if include_error:
			lambda1 = max(0, lambda1 + self._error(mass1, lambda1))
			lambda2 = max(0, lambda2 + self._error(mass2, lambda2))

		config = {
			'mass1': mass1,
			'mass2': mass2,
			'lambda1': lambda1,
			'lambda2': lambda2,
			"inclination": inclination
		}
		return config


class Generator:

	def __init__(self, work_dir: str = "gw_ts", model_name: str = None, model_file: str = None,
				 sample_rate: float = 4096, distance: float = 39, fmin: float = 20,
				 approximant: str = "IMRPhenomPv2_NRTidalv2", model_config: dict = {}):

		self.work_dir = os.path.normpath(work_dir)

		# Processing config
		self.process_params = {"out_size": sample_rate*2,
							   "crop_radius": sample_rate-100,
							   "centered": False
							   }

		# Generator config
		self.fmin = fmin
		self.sample_rate = sample_rate
		self.distance = distance
		self.approximant = approximant
		self.model_name = model_name

		# Model
		if model_name:
			try:
				if model_file:
					self.model = Models(model_name=model_name, data_file=model_file, **model_config)
				else:
					self.model = Models(model_name=model_name, **model_config)

			except Exception as e:
				raise ValueError(f"Couldn't load model. Defaulting to random generator.\n\tINFO:{e}")
				
		else:
			self.model_name = "random"
			self.model = RandomModel(**model_config)

	@staticmethod
	def gen_data_strain(config: dict):
		hp, hc = get_td_waveform(approximant=config["approximant"],
								 mass1=config['mass1'],
								 mass2=config['mass2'],
								 lambda1=config["lambda1"],
								 lambda2=config["lambda2"],
								 f_lower=config["fmin"],
								 f_final=config['sample_rate'],
								 distance=config['distance'],
								 delta_t=config["delta_t"])

		return hp, hc

	@property
	def generator_config(self):
		config = {"sample_rate": self.sample_rate,
				"fmin": self.fmin,
				"delta_t": 1.0/self.sample_rate,
				"distance": self.distance,
				"approximant": self.approximant,
		}
		return config

	def random_config(self) -> dict:
		model_config = self.model.random_config()
		gen_config = self.generator_config
		return {**model_config, **gen_config}

	def gen_spec(self, array):

		window = tukey(M=int(len(array)))
		window[int(0.5*len(array)):] = 1
		array *= window

		spec = abs(cqt(
				array, 
				sr=self.sample_rate, 
				hop_length=64, 
				pad_mode="constant", 
				fmin=32, 
				bins_per_octave=28, 
				n_bins=128, 
				tuning=0
				)).transpose()
		# ~761 hertz

		return spec

	def _run_gen(self,
			  polarization: str = None,
			  process: bool = False,
			  log: bool = True
			  ) -> dict:

		if log:
			logger = start_logger_if_necessary()

		if polarization is None:
			pol = ["hp", "hc"]
		elif polarization == "hp":
			pol = ["hp"]
		elif polarization == "hc":
			pol = ["hc"]
		else:
			raise ValueError("\'{}\' is not a valid value for field \'polarization\'. Use \'None\', \'hp\' or \'hc\'.")

		success = False
		while not success:
			config = self.random_config()

			try:
				ts_hp, ts_hc = self.gen_data_strain(config)
				success = True
			except Exception as e:
				if log:
					logger.exception(e)
		else:
			files = []
			if "hp" in pol:
				if process:
					ts_hp = ReShape.reshape_array(ts_hp, **self.process_params)

				ts_hp = ts_hp/np.max(ts_hp)

				spec_hp = self.gen_spec(ts_hp)
				spec_hp = spec_hp / np.max(spec_hp)
				spec_hp = spec_hp.astype(np.float16)
				meta_data = {**config, "polarization": "hp", "model": self.model_name}
				run_hp = Spectrogram(data=spec_hp, metadata=meta_data)

			else:
				run_hp = None

			if "hc" in pol:
				if process:
					ts_hc = ReShape.reshape_array(ts_hc, **self.process_params)

				ts_hc = ts_hc/np.max(ts_hc)

				spec_hc = self.gen_spec(ts_hc)
				spec_hc = spec_hc / np.amx(spec_hc)
				spec_hc = spec_hc.astype(np.float16)
				meta_data = {**config, "polarization": "hc", "model": self.model_name}
				run_hc = Spectrogram(data=spec_hc, metadata=meta_data)

			else:
				run_hc = None

		return run_hp, run_hc

	def run_gen(self,
				n: int,
				out_file: str = "dataset.h5",
				polarization: str = None,
				process: bool = False,
				) -> None:

		idx = [f"{i:08d}" for i in range(1, n+1)]
	
		runs = Parallel(n_jobs=-1)(delayed(self._run_gen)(polarization=polarization, process=process)
							for _ in tqdm(idx,
											  desc=f"Generating waveforms ".ljust(40, "_")
											  )
							)

		print(f"\nSaving results...\n\t - {out_file=}")
		grp = "waveforms"
		with h5py.File(out_file, "w") as f:
			f.create_group(grp)
			for i, r in zip(idx, runs):
				# TODO: If not whitened need to save as double
				for p in r:
					if p is not None:
						
						dataset = f[grp].create_dataset(name=f"{i}_{p.metadata['polarization']}", data=p.data)

						if isinstance(p.metadata, dict):
							for k, v in flatten_dict(p.metadata).items():
								if v is not None:
									dataset.attrs[k] = v
		print("Done!")

	@staticmethod
	def get_dataset_keys(f):
		keys = []
		f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
		return keys



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description="Generate GW time-series data.")
	parser.add_argument("n", metavar="N", type=int, default=int(1e3),
						help="Number of time-series files to generate.")
	parser.add_argument("-w", metavar="PATH", type=str, default="gw_ts",
						help="Working directory.")
	parser.add_argument("-p", metavar="hc, hp", type=str, nargs="?", default=None,
						help="Save plus polarization (hc) or cross polarization (hp).")
	parser.add_argument("--model-name", metavar="NAME", type=str, default=None,
						help="Name of EOS used for mass/lambda parameters. If None random values are used.")
	parser.add_argument("--model-file", metavar="PATH", type=str, default=None,
						help="Path of file with available EOS data.")
	parser.add_argument("--process", action='store_true',
						help="Save processed files.")

	args = parser.parse_args()

	config = {}
	config["mmin"] = 1
	config["mmax"] = 1.5

	# Generate
	gen = Generator(work_dir=args.w,
					model_name=args.model_name,
					model_file=args.model_file,
					model_config=config
					)

	out_file = os.path.join(args.w, "dataset.h5")

	part_run = gen.run_gen(n=args.n, out_file=out_file, polarization=args.p, process=args.process)

