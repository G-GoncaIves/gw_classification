import h5py
import atexit
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Generate GW time-series data.")
parser.add_argument("-w", metavar="PATH", type=str, help="Root directory where sub_directories with the EOS-specific hdf5 are stored")

args = parser.parse_args()

eos_dir = args.w

eos_available = os.listdir(eos_dir)

for eos in eos_available:

    eos_h5_path = os.path.join(eos_dir, eos, "dataset.h5")
    
    data = h5py.File(eos_h5_path, "r")
    atexit.register(data.close)

    keys = []
    for key in data["waveforms"].keys():

    	keys.append(key)

    waveform = data["waveforms"][keys[0]]
    model_attr = waveform.attrs["model"]
    waveform_array = np.asarray(waveform)

    spec_file_name = os.path.join(eos_dir, eos, f"{eos}_spec.png")
    plt.imsave(spec_file_name, waveform_array, cmap="turbo", origin="lower")
