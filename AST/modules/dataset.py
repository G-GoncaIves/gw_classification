# Built-in:
import os
import atexit
from datetime import datetime
import random
import collections

# Installed:
import h5py
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from librosa import cqt
import torch.nn.functional
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import Resize

# Custom:
from .recibo import progress_csv

class My_DataSet(Dataset):
	def __init__(
		self, 
		input_hdf5_path: str,
		label_dict: dict = None,
		atributes: list = ["mass1", "mass2", "lambda1", "lambda2", "model"]
		):

		assert os.path.isfile(input_hdf5_path), f"Input Data File not found. Checked: {input_hdf5_path}"

		# Spectograms and labels
		self.spectograms = []
		self.labels = []

		# Extracts Data from the hdf5
		self.data = h5py.File(input_hdf5_path, "r")
		self.keys = self.get_keys(self.data["waveforms"])
		atexit.register(self.data.close)

		print(len(self.keys))
		self.atributes = self.get_atributes(waveform_data=self.data["waveforms"], keys=self.keys, atributes_list=atributes)

		# Converts labels into the one hot format:
		if self.desired_eos is None:
			unique_labels = list(set(self.atributes[d]["model"] for d in self.keys))

		else:
			unique_labels = sorted(set(self.desired_eos))
			print(f"unique labels = {unique_labels}")

		if label_dict is None:
			raw_one_hot = {name: torch.FloatTensor([0]*i + [1] + [0]*(len(unique_labels)-i-1)) for i, name in enumerate(unique_labels)}
			self.one_hot = collections.OrderedDict(sorted(raw_one_hot.items()))
			print(f"\n\n\n One hot dict: \n {self.one_hot} \n\n\n")
		else:
			self.one_hot = label_dict
			print(f"\n\n\n One hot dict: \n {self.one_hot} \n\n\n")

		# Variable that dictates wether SpecAug is aplied or not:
		self.spec_aug = None

	def get_atributes(self, waveform_data, keys, atributes_list):

		required_atributes = {}

		for key in keys:

			required_atributes[key] = {}
			global_atributes = waveform_data[key].attrs

			for at in global_atributes:
				for r_at in atributes_list:
					
					if r_at in at:
						required_atributes[key][r_at] = global_atributes[at]

		return required_atributes


	def get_keys(self, h5_file):
		keys = []
		h5_file.visit(lambda key: keys.append(key) if isinstance(h5_file[key], h5py.Dataset) else None)
		return keys

	def record_stats(self, csv_path):

		for n, key in enumerate(self.keys):
		
			params = self.atributes[key]
			m1 = params["mass1"]
			m2 = params["mass2"]
			lambda1 = params["lambda1"]
			lambda2 = params["lambda2"]
			model = params["model"]
		
			access_type = "w+" if n==0 else "a+"
			progress_csv(
					n,
					m1,
					m2,
					lambda1,
					lambda2,
					model,
					access_type = access_type,
					csv_path = csv_path
					) 

	def __getitem__(self, idx):
		"""
		"""
		spec = self.spectograms[idx]
		label = self.labels[idx]

		if self.mix_up == True:

			random_idx = random.randint(0, len(self.spectograms)-1)
			random_spec = self.spectograms[random_idx]
			random_label = self.labels[random_idx]

			_lambda = np.random.beta(10,10)

			spec = _lambda * spec + (1 - _lambda) * random_spec
			label = _lambda * label + (1 - _lambda) * random_label

		if self.spec_aug == True:

			time_bins, freq_bins = spec.shape
			freq_mask = torchaudio.transforms.FrequencyMasking(freq_bins/2)
			time_mask = torchaudio.transforms.TimeMasking(time_bins/2)

			spec = spec.to(torch.float32)
			spec = freq_mask(spec)
			spec = time_mask(spec)
			spec = spec.to(torch.float16)


		# The output spec shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
		return spec, label

	def __len__(self):
		return len(self.spectograms)

class WaveForms(My_DataSet):

	def __init__(
		self, 
		input_hdf5_path: str,
		audio_conf: dict, 
		norm_mean: float = None, 
		norm_std: float = None,
		verbose=False,
		label_dict:dict = None
		):

		super().__init__(self, input_hdf5_path=input_hdf5_path, label_dict=label_dict)
		assert all([required_key in audio_conf.keys() for required_key in ["target_length","num_mel_bins","freqm","timem"]]), "Configuration Dict missing required parameters."

		self.audio_conf = audio_conf

		if norm_mean == None or norm_std == None:
			self.calc_stats()
		else:
			self.norm_mean = norm_mean
			self.norm_std = norm_std

		self.gen_spectograms()

	def calc_stats(self, indices = None):

		set_mean = []
		set_std = []

		if indices:
			keys = (self.keys[i] for i in indices)
		
		else:
			keys = self.keys

		for k in tqdm(keys):

			v = np.array(self.data["waveforms"][k])
			v_norm = v / v.max()

			set_mean.append(v_norm.mean())
			set_std.append(v_norm.std())

		self.norm_mean = np.array(set_mean).mean()
		self.norm_std = np.array(set_std).std()

	def gen_spectograms(self):

		for k in tqdm(self.keys, leave=False):

			waveform_tensor= self.atributes[k]
			label = self.one_hot[waveform_tensor["model"]]

			# Normalize
			waveform = np.asarray(waveform_tensor, dtype="f")
			waveform = waveform / waveform.max()

			# Convert to tensor
			waveform = torch.from_numpy(waveform)
			waveform = waveform.reshape((1, -1))
			
			spectrogram = abs(cqt(
									waveform.numpy(),
									sr = 4096*4,
									fmin = 32,
									n_bins = 256,
									bins_per_octave = 32,
									hop_length = 128,
									pad_mode = 'edge'
									))

			self.spectograms.append(spectrogram)
			self.labels.append(label)

class Spectrograms(My_DataSet):

	def __init__(
		self, 
		input_hdf5_path: str,
		verbose=False,
		spec_rezise_ex=True,
		label_dict:dict = None,
		mix_up:bool = False,
		mass_range:str = None,
		lambda_range:str = None,
		desired_eos:str = None
		):
   
		constraint_string = ""

		self.mass_range = mass_range
		if mass_range is not None:

			mass_min_str, mass_max_str = mass_range.split(",")
			self.mass_min, self.mass_max = float(mass_min_str), float(mass_max_str)
			constraint_string += f"\n\t Mass Range = [{self.mass_min:.1f}, {self.mass_max:.1f}]"

		self.lambda_range = lambda_range
		if lambda_range is not None:

			lambda_min_str, lambda_max_str = lambda_range.split(",")
			self.lambda_min, self.lambda_max = float(lambda_min_str), float(lambda_max_str)
			constraint_string += f"\n\t Lambda Range = [{self.lambda_min:.1f}, {self.lambda_max:.1f}]"

		self.desired_eos = desired_eos
		if desired_eos is not None:

			self.desired_eos = list(desired_eos.split(","))
			constraint_string += f"\n\t EOS used = {desired_eos}"

		if len(constraint_string) > 0 and verbose:
			print(f"Train Set Constraints: \n", constraint_string)			
		
		super().__init__(input_hdf5_path=input_hdf5_path, label_dict=label_dict)

		self.spec_resize_ex = spec_rezise_ex
		self.mix_up = mix_up

		self.hdf5_path = input_hdf5_path
		assert os.path.isfile(self.hdf5_path), f"Spectrogram HDF5 not found. Checked {self.hdf5_path}"

		self.mass_ratio = {}
		self.class_sample_count = {}
		for model in self.one_hot.keys():

			self.mass_ratio[model] = []
			self.class_sample_count[model] = 0


		self.load_spectrograms()
		print("loaded spectograms")

	def load_spectrograms(self):

		for k in tqdm(self.keys, leave=False):

			# This is an instance of 'h5py._hl.dataset.Dataset', hence the name
			spec_dataset = self.atributes[k]

			valid = True

			if self.mass_range is not None:

				for mass in [spec_dataset["mass1"], spec_dataset["mass2"]]:

					if mass < self.mass_min or mass > self.mass_max: 
						valid = False

			if self.lambda_range is not None:

				for _lambda in [spec_dataset["lambda1"], spec_dataset["lambda2"]]:  
					
					if _lambda < self.lambda_min or _lambda > self.lambda_max: 
						valid = False

			if self.desired_eos is not None:

				if spec_dataset["model"] not in self.desired_eos:
					valid = False

			if valid:

				# Converted to an instance of 'numpy.ndarray'
				spec_array = np.asarray(self.data["waveforms"][k])
				# Converted to an instance of 'torch.Tensor'
				spec_tensor = torch.from_numpy(spec_array)
				
				spec_model = spec_dataset["model"]
				label = self.one_hot[spec_model]

				ratio = float(spec_dataset["mass1"]) / float(spec_dataset["mass2"]) 
				self.mass_ratio[spec_model].append(ratio)

				self.class_sample_count[spec_model] += 1

				spec_dims = spec_tensor.shape
				assert len(spec_dims) == 2, "Spectrograms in unexpected format."
				i, j = spec_dims
				# i - time bins number ; j - frequency bins number
				if j != 128:

					transform = Resize((i,128))
					tensor = torch.reshape(spec_tensor, (1, i, j))
					transformed_tensor = transform(tensor)
					spec_tensor = transformed_tensor.squeeze()

					if self.spec_resize_ex:

						spec_dir = os.path.dirname(self.hdf5_path)
						before_spec_file_name = os.path.join(spec_dir, "before_resize_spec.png")
						after_spec_file_name = os.path.join(spec_dir, "after_resize_spec.png")
						
						plt.imsave(before_spec_file_name, spec_array, cmap="turbo", origin="lower")
						plt.imsave(after_spec_file_name, spec_tensor.numpy(), cmap="turbo", origin="lower")
				
						self.spec_resize_ex = False

				self.spectograms.append(spec_tensor.half())
				self.labels.append(label)
