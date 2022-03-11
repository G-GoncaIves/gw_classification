import glob
import os
import sys

import h5py
import tqdm

base_dir = "/home/goncalo/GW_pycbc/AST/modules/test_dataset/valid"

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
# print("Cleaning...")
# for f in files:
#     os.remove(f)

print("Done.")