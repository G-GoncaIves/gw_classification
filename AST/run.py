import argparse
import os
import sys
import torch
import ast
import torch.optim as optim
import pickle
import tqdm as tqdm
from datetime import datetime

torch.cuda.empty_cache()
def train_model(
	dataloader, 
	train_function, 
	classifier, 
	dataset, 
	args, 
	verbose=False
	):

	if verbose:
		global_start_time = datetime.now()
		print(f"\t > Began Training Set-up, {global_start_time}.")

	required_dirs = [
		os.path.join(args.exp_dir, "saved_states"), 
		os.path.join(args.exp_dir, "pretrained_weights")
	]

	for r_dir in required_dirs: 
		if not os.path.isdir(r_dir):
			
			if verbose:
				print(f"... Creating required directory: {r_dir}")
			os.makedirs(r_dir)

	target_length = 1024
	
	if verbose:
		time = datetime.now() - global_start_time
		print(f"\t > Loading Dataset to memory ... [{time}]")
		global_dataset = dataset(args.data_train, verbose=True)
		time = datetime.now() - global_start_time
		print(f"\t > Global Dataset successfully loaded to memory. [{time}]")
	else:
		global_dataset = dataset(args.data_train)
		
	train_ratio, val_ratio = 0.8, 0.2
	train_portion = int(len(global_dataset) * train_ratio)
	val_portion = len(global_dataset) - train_portion
	train_dataset, val_dataset = torch.utils.data.random_split(global_dataset, [train_portion, val_portion])
	
	if verbose:
		time = datetime.now() - global_start_time
		print(f"\t > Dataset of {len(global_dataset)} successfuly split into: [{time}] \n\t\t TrainSet: {len(train_dataset)}\n\t\t ValSet: {len(val_dataset)}")

	train_loader = dataloader(
			  			train_dataset,
			  			batch_size=args.batch_size, 
		  				shuffle=args.data_shuffle, 
		  				num_workers=args.num_workers, 
		  				pin_memory=True
			  		)

	val_loader = dataloader(
						val_dataset,
						batch_size=args.batch_size, 
						shuffle=args.data_shuffle, 
						num_workers=args.num_workers, 
						pin_memory=True
					)

	audio_model = classifier(
	    				label_dim=args.n_class, 
	    				fstride=args.fstride, 
	    				tstride=args.tstride, 
	    				input_fdim=args.fdim,
                    	input_tdim=args.tdim, 
	    				imagenet_pretrain=args.imagenet_pretrain,
                    	audioset_pretrain=args.audioset_pretrain, 
	    				model_size='base384', 
                    	exp_dir=args.exp_dir,
                    	verbose=verbose
                    )
	if verbose:
		time = datetime.now() - global_start_time
		print(f"\t > Successfuly Created Dataloaders. [{time}]")

	arg_file_path = os.path.join(args.exp_dir, "args.pkl")
	with open(arg_file_path, "wb") as f:
	    pickle.dump(args, f)
	
	train_function(audio_model, train_loader, val_loader, vars(args))

	if verbose:
		time = datetime.now() - global_start_time
		print(f"\t > Successfuly Trained Model. [{time}]")