import argparse
import glob
import logging
import os
import sys

from mpi4py import MPI
from librosa import cqt
import h5py
import numpy as np
import pandas as pd
# from mpi4py import MPI
from pycbc.waveform import get_td_waveform
from scipy import interpolate

from processing import ReShape


# Get MPI info
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
        rng = np.random.default_rng()

        mass1 = float(rng.uniform(self.mmin, self.mmax))
        mass2 = float(rng.uniform(self.mmin, mass1))

        lambda1 = float(self.mass2lambda(mass1))
        lambda2 = float(self.mass2lambda(mass2))

        if include_error:
            lambda1 = max(0, lambda1 + self._error(mass1, lambda1))
            lambda2 = max(0, lambda2 + self._error(mass2, lambda2))

        config = {
            'mass1': mass1,
            'mass2': mass2,
            'lambda1': lambda1,
            'lambda2': lambda2,
        }
        return config


class Generator:

    def __init__(self, rank: int, work_dir: str = "gw_ts", model_name: str = None, model_file: str = None,
                 sample_rate: float = 4*4096, distance: float = 39, inclination: float = 10, fmin: float = 20,
                 approximant: str = "IMRPhenomPv2_NRTidalv2"):

        # General config
        self.rank = int(rank)
        self.work_dir = os.path.normpath(work_dir)

        # Processing config
        self.process_params = {"out_size": 2048*16,
                               "crop_radius": 2048*8-100,
                               "centered": False
                               }

        # Generator config
        self.fmin = fmin
        self.sample_rate = sample_rate
        self.distance = distance
        self.inclination = inclination
        self.approximant = approximant
        self.model_name = model_name

        # Model
        if model_name:
            try:
                if model_file:
                    self.model = Models(model_name=model_name, data_file=model_file)
                else:
                    self.model = Models(model_name=model_name)
            except Exception as e:
                print(f"Couldn't load model. Defaulting to random generator.\n\tINFO:{e}")
                self.model = RandomModel()
                self.model_name = "random"
        else:
            self.model_name = "random"
            self.model = RandomModel()

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
                                 inclination=config['inclination'],
                                 delta_t=config["delta_t"])

        return hp, hc

    @property
    def generator_config(self):
        config = {"sample_rate": self.sample_rate,
                "fmin": self.fmin,
                "delta_t": 1.0/self.sample_rate,
                "distance": self.distance,
                "inclination": self.inclination,
                "approximant": self.approximant,
        }
        return config

    def random_config(self) -> dict:
        model_config = self.model.random_config()
        gen_config = self.generator_config
        return {**model_config, **gen_config}

    def _save_run(self, array, run_name, meta_data=None):

        # New
        spec = abs(cqt(
            array, 
            sr=self.sample_rate, 
            hop_length=218, 
            pad_mode="constant", 
            fmin=32, 
            bins_per_octave=32, 
            n_bins=128, 
            tuning=0
            ))

        with h5py.File(self.out_file, 'a') as f:
            run = f["waveforms"].create_dataset(run_name, data=spec)
            if isinstance(meta_data, dict):
                for k, v in meta_data.items():
                    run.attrs[k] = v

    def _run_gen(self,
              polarization: str = None,
              process: bool = False,
              run_id: int = None,
              id_places: int = 8,
              log: bool = True
              ) -> dict:

        assert run_id is not None, "\'run_id\' is None. Int required for saving."

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

        run_id = str(run_id).zfill(id_places)

        success = False
        while not success:
            config = self.random_config()

            if log:
                logger.info("Generating GW time-series for run %s.", run_id)

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
                    logger.info("Processing file %s (hp).", run_id)
                    ts_hp = ReShape.reshape_array(ts_hp, **self.process_params)

                logger.info("Saving data of run %s (hp).", run_id)
                meta_data = {**config, "polarization": "hp", "model": self.model_name}
                name = f"ts_hp_{run_id}"
                self._save_run(array=ts_hp, run_name=name,  meta_data=meta_data)
                files.append(name)
            if "hc" in pol:
                if process:
                    logger.info("Processing run %s (hc).", run_id)
                    ts_hc = ReShape.reshape_array(ts_hc, **self.process_params)

                logger.info("Saving data of run %s (hc).", run_id)
                meta_data = {**config, "polarization": "hc", "model": self.model_name}
                name = f"ts_hc_{run_id}"
                self._save_run(array=ts_hc, run_name=name,  meta_data=meta_data)
                files.append(name)
        return files

    def run_gen(self,
                n: int,
                n_start: int,
                polarization: str = None,
                process: bool = False,
                ) -> None:

        # Create out file.
        self.out_file = os.path.join(self.work_dir, f"dataset.h5.{rank:03}")
        logger.info("(Thread %i) Creating out file \'%s\'.", self.rank, self.out_file)
        with h5py.File(self.out_file, 'w') as f:
            f.create_group("waveforms")

        logger.info("Starting generation of n=%i GW time-series.", n)
        print(f"(Thread {self.rank}) Generating time-series...")

        files = []
        for i in range(n_start+1, n_start+n+1):
            fnames = self._run_gen(run_id=i, polarization=polarization, process=process)
            files.append(fnames)

        n_files = len(files)
        msg = f"(Thread {self.rank}) Finished generating. Files generated: {n_files}."
        logger.info(msg)

        out_file = self.out_file
        del self.out_file

        return out_file

    @staticmethod
    def get_dataset_keys(f):
        keys = []
        f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
        return keys


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

# Can use MPI, but it's so fast it doesn't make a difference
n_pt = [args.n // size + ((args.n % size) > r) for r in range(size)]
n_start = [sum(n_pt[:r]) for r in range(size)]

# Generate
gen = Generator(rank=rank, work_dir=args.w,
                model_name=args.model_name,
                model_file=args.model_file
                )
part_run = gen.run_gen(n=n_pt[rank], n_start=n_start[rank], polarization=args.p, process=args.process)

# Merge
parts = comm.gather(part_run, root=0)
if rank == 0:
    print("Merging parts...")
    if len(parts) != size:
        print("ERROR: Missing parts.")
        sys.exit(1)
    with h5py.File(os.path.join(args.w, f"dataset.h5"), "w") as dataset:
        grp = dataset.create_group("waveforms")
        for f in parts:
            with h5py.File(f, "r") as part:
                print(f"Merging {f}...")
                for k, v in part["waveforms"].items():
                    part.copy(v, grp, name=k)
            os.remove(f)
