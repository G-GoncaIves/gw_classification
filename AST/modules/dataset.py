import atexit

import h5py
import numpy as np
import torch
import torch.nn.functional
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm
from datetime import datetime
from librosa import cqt
import os
from torchvision.transforms import Resize
import matplotlib.pyplot as plt


class My_DataSet(Dataset):
    def __init__(
        self, 
        input_hdf5_path: str
        ):

        assert os.path.isfile(input_hdf5_path), f"Input Data File not found. Checked: {input_hdf5_path}"

        # Spectograms and labels
        self.spectograms = []
        self.labels = []

        # Extracts Data from the hdf5
        self.data = h5py.File(input_hdf5_path, "r")
        self.keys = self.get_keys(self.data["waveforms"])
        atexit.register(self.data.close)

        # Converts labels into the one hot format:
        unique_labels = list(set(self.data["waveforms"][d].attrs["model"] for d in self.keys))
        self.one_hot = {name: torch.FloatTensor([0]*i + [1] + [0]*(len(unique_labels)-i-1)) for i, name in enumerate(unique_labels)}


    def get_keys(self, h5_file):
        keys = []
        h5_file.visit(lambda key: keys.append(key) if isinstance(h5_file[key], h5py.Dataset) else None)
        return keys

    def __getitem__(self, idx):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        spec = self.spectograms[idx]
        label = self.labels[idx]

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
        verbose=False
        ):

        super().__init__(self, input_hdf5_path=input_hdf5_path)
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

            waveform_tensor= self.data["waveforms"][k]
            label = self.one_hot[waveform_tensor.attrs["model"]]

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
        spec_rezise_ex=True
        ):
        
        super().__init__(input_hdf5_path)

        self.spec_resize_ex = spec_rezise_ex

        self.hdf5_path = input_hdf5_path
        assert os.path.isfile(self.hdf5_path), f"Spectrogram HDF5 not found. Checked {self.hdf5_path}"
       
        self.load_spectrograms()

    def load_spectrograms(self):

        for k in tqdm(self.keys, leave=False):

            # This is an instance of 'h5py._hl.dataset.Dataset', hence the name
            spec_dataset = self.data["waveforms"][k]
            # Converted to an instance of 'numpy.ndarray'
            spec_array = np.asarray(spec_dataset)
            # Converted to an instance of 'torch.Tensor'
            spec_tensor = torch.from_numpy(spec_array)
            
            spec_model = spec_dataset.attrs["model"]
            label = self.one_hot[spec_model]

            spec_dims = spec_tensor.shape
            assert len(spec_dims) == 2, "Spectrograms in unexpected format."
            i, j = spec_dims
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

