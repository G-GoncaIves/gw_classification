import glob
import os
import sys
import argparse

import h5py
import tqdm


parser = argparse.ArgumentParser(description="Generate GW time-series data.")
parser.add_argument("-w", metavar="PATH", type=str, help="Root directory where sub_directories with the EOS-specific hdf5 are stored")

args = parser.parse_args()
base_dir = args.w 

print(f"Starging merge ({base_dir})")
files = glob.glob(os.path.join(base_dir, "*/dataset.h5"))

i = 1   # TODO: Check this

with h5py.File(os.path.join(base_dir, "dataset.h5"), "w") as dataset:
    grp = dataset.create_group("waveforms")
    for f in tqdm.tqdm(files):
        print(f"Merging {f}...")
        with h5py.File(f, "r") as part:
            for k, v in part["waveforms"].items():

                part.copy(v, grp, name=f"ts_{i:08d}")
                i += 1
print("Done.")
