# Built-in:
import os

# Installed:
import h5py
import argparse
import matplotlib.pyplot as plt

# Custom:
from AST.modules.dataset import Spectrograms

# Required paths:
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--h5_path", help="Path to the hdf5.")
parser.add_argument("--csv_path", help="Path where to create the csv.")
parser.add_argument("--plot_path", help="Path where to save the plot.")

args = parser.parse_args()

h5_path = args.h5_path
csv_path = args.csv_path
plot_path = args.plot_path

dataset = Spectrograms(h5_path)
dataset_stats = dataset.record_stats(csv_path=csv_path)

def read_csv(csv_path, idx_list):

	with open(csv_path, "r")as csv_f:

		lines = csv_f.readlines()

	data_dict = dict()
	for line in lines:

		elements = line.strip().split(";")
		model = elements[-1]

		for k in range(len(idx_list)):

			el = float(elements[idx_list[k]])
			
			try:
				data_dict[f"{model}"][k].append(el)
				
			except KeyError:

				data_dict[f"{model}"] = []
				
				for u in idx_list:
					data_dict[f"{model}"].append([])

				data_dict[f"{model}"][k].append(el)
				
	return data_dict

data_dict = read_csv(csv_path=csv_path, idx_list=[1, 3])

models = list(data_dict.keys())
print(len(models))
fig, axs = plt.subplots(3,2)
fig.patch.set_facecolor('black')
fig.set_dpi(200)

i = 0
for axi in axs:
	for sub_axi in axi:

		if i <5:
			model = models[i]
			mass_array = data_dict[model][0]
			lambda_array = data_dict[model][1]

			sub_axi.scatter(mass_array, lambda_array, color='cornflowerblue', s=0.8)
			sub_axi.set_title(f"{model}", color='cornflowerblue')
			
		else:

			for model in models:
				mass_array = data_dict[model][0]
				lambda_array = data_dict[model][1]

				sub_axi.scatter(mass_array, lambda_array, label=f"{model}", s=0.8)
				box = sub_axi.get_position()
				sub_axi.set_position([
					box.x0, 
					box.y0 + box.height * 0.1, 
					box.width, 
					box.height * 0.9])
				sub_axi.legend(
					loc='upper center', 
					bbox_to_anchor=(0.5, -0.05),
          			fancybox=True, 
          			shadow=True, ncol=5,
          			fontsize="xx-small",
          			facecolor="black",
          			labelcolor="white")
			
		sub_axi.set_facecolor('black')
		sub_axi.spines['bottom'].set_color('salmon')
		sub_axi.spines['left'].set_color('salmon')
		sub_axi.xaxis.label.set_color('salmon')
		sub_axi.yaxis.label.set_color('salmon')
		sub_axi.tick_params(axis='x', colors='salmon')
		sub_axi.tick_params(axis='y', colors='salmon')

		i += 1

plt.subplots_adjust(
	left=0.1,
	bottom=0.1,
	right=0.9,
	top=0.9,
	wspace=0.5,
	hspace=0.5)
plt.savefig(plot_path)




