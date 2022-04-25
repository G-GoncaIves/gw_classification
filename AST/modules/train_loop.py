# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

# Built-in:
import os
import sys
import time
import pickle
import datetime

# Installed:
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn import metrics
from torch.cuda.amp import autocast, GradScaler

# Custom:
from .recibo import recibo, progress_csv

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def epoch_progress_bar_generator_verbose(total_epochs):
	return tqdm(range(1, total_epochs+1), desc="Trainig", ncols=90, leave=False, position=0, bar_format="{l_bar}{bar}| eta: {eta}", colour="magenta")

def epoch_progress_bar_generator_silent(total_epochs):
	return range(1, total_epochs+1)
	
def batch_progress_bar_generator_verbose(loader, title):
	return enumerate(tqdm(loader, desc=title, ncols=90, leave=False, position=1, bar_format="{l_bar}{bar}| {remaining}", colour="cyan"))

def batch_progress_bar_generator_silent(loader, title):
	return enumerate(loader)

def train(audio_model, train_loader, val_loader, train_conf):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.set_grad_enabled(True)

	# Initialize all of the statistics we want to keep track of
	batch_time = AverageMeter()
	per_sample_time = AverageMeter()
	data_time = AverageMeter()
	per_sample_data_time = AverageMeter()
	loss_meter = AverageMeter()
	per_sample_dnn_time = AverageMeter()
	progress = []
	# best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
	best_epoch, best_mAP, best_acc, best_train_loss = 0, 0, -np.inf, 1
	global_step = 0
	start_time = time.time()
	exp_dir = train_conf["exp_dir"]
	verbose = True if train_conf["verbose"] else False

	if not isinstance(audio_model, nn.DataParallel):
		audio_model = nn.DataParallel(audio_model)

	audio_model = audio_model.to(device)
	# Set up the optimizer
	trainables = [p for p in audio_model.parameters() if p.requires_grad]
	optimizer = torch.optim.Adam(trainables, train_conf["lr"], weight_decay=5e-7, betas=(0.95, 0.999))

	# dataset specific settings
	#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=train_conf["lr"]_patience, verbose=True)
	
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5,10,20,30], gamma=0.5, last_epoch=-1)
	main_metrics = 'mAP'
	loss_fn = nn.BCEWithLogitsLoss()
	warmup = True
	train_conf["loss_fn"] = loss_fn

	# for amp
	scaler = GradScaler()

	begin_date = datetime.datetime.now()
	audio_model.train()

	if verbose:
		epoch_progress_bar_generator = epoch_progress_bar_generator_verbose 
		batch_progress_bar_generator = batch_progress_bar_generator_verbose
	else:
		epoch_progress_bar_generator = epoch_progress_bar_generator_silent 
		batch_progress_bar_generator = batch_progress_bar_generator_silent
		
	# Enable SpecAug for training:
	train_loader.dataset.spec_aug = True
	# Disable SpecAug for validation:
	val_loader.dataset.spec_aug = False

	for epoch in epoch_progress_bar_generator(total_epochs=train_conf["epochs"]):
		begin_time = time.time()
		end_time = time.time()
		audio_model.train()

		for i, (audio_input, labels) in batch_progress_bar_generator(loader=train_loader, title="train"):

			B = audio_input.size(0)
			audio_input = audio_input.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			data_time.update(time.time() - end_time)
			per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
			dnn_start_time = time.time()

			# first several steps for warm-up
			if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
				warm_lr = (global_step / 1000) * train_conf["lr"]
				for param_group in optimizer.param_groups:
					param_group['lr'] = warm_lr

			with autocast():
				audio_output = audio_model(audio_input)
				if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
					loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
				else:
					loss = loss_fn(audio_output, labels)

			# optimization if amp is not used
			# optimizer.zero_grad()
			# loss.backward()
			# optimizer.step()


			# optimiztion if amp is used
			optimizer.zero_grad()
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			# record loss
			loss_meter.update(loss.item(), B)
			batch_time.update(time.time() - end_time)
			per_sample_time.update((time.time() - end_time)/audio_input.shape[0])
			per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0])

			end_time = time.time()
			global_step += 1

		#
		stats, valid_loss = validate(audio_model, val_loader, train_conf, verbose, bar_generator=batch_progress_bar_generator)
		
		"""
		"""
		"""
		mAP = np.mean([stat['AP'] for stat in stats])
		mAUC = np.mean([stat['auc'] for stat in stats])
		acc = stats[0]['acc']

		middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
		middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
		average_precision = np.mean(middle_ps)
		average_recall = np.mean(middle_rs)
		"""
		"""
		"""
		
		mAP = 0
		mAUC = 0
		acc = 0

		middle_ps = 0
		middle_rs = 0
		average_precision = 0
		average_recall = 0
		
		train_loss = loss_meter.avg

		if train_loss < best_train_loss:
			
			best_epoch = epoch
			best_train_loss = loss_meter.avg
			best_val_loss = valid_loss
			best_mAP = mAP
			best_acc = acc
			best_auc = mAUC
			best_average_precision = average_precision
			best_average_recall = average_recall 

			# Saves Best model state to memory.
			best_model_path = os.path.join(exp_dir, "saved_states", "best_audio_model.pth")
			best_optim_path = os.path.join(exp_dir, "saved_states", "best_optim.pth")
			torch.save(audio_model, best_model_path)
			torch.save(optimizer, best_optim_path)

		# Saves Current model state to memory.
		current_model_path = os.path.join(exp_dir, "saved_states", "latest_audio_model.pth")
		current_optim_path = os.path.join(exp_dir, "saved_states", "latest_optim.pth")
		torch.save(audio_model, current_model_path)
		torch.save(optimizer, current_optim_path)
		
		# Checks memory allocation.
		peak_cuda_memory_allocation = torch.cuda.max_memory_allocated() * 9.3132257461548e-10 

		# Saves current and best metrics to a txt file.
		recibo_path = os.path.join(exp_dir, "recibo.txt")
		print(recibo_path)
		recibo(
			start_time = begin_date,
			current_time = datetime.datetime.now(),
			epoch = epoch,
			train_loss = train_loss,
			val_loss = valid_loss,
			_map = mAP,
			acc = acc,
			auc = mAUC,
			avg_pre = average_precision,
			avg_recall = average_recall,
			best_epoch = best_epoch,
			best_train_loss = best_train_loss,
			best_val_loss = best_val_loss,
			best_map = best_mAP,
			best_acc = best_acc,
			best_auc = best_auc,
			best_avg_pre = best_average_precision,
			best_avg_recall = best_average_recall,
			peak_cuda_mem = peak_cuda_memory_allocation,
			current_state_loc = current_model_path,
			best_state_loc = best_model_path,
			recibo_location = recibo_path
			)
		
		# Saves metrics progression to a csv.
		csv_path = os.path.join(exp_dir, "metrics.csv")
		access_type = "a+" if os.path.isfile(csv_path) else "w+"
		progress_csv(
				epoch,
				train_loss,
				valid_loss,
				mAP,
				acc,
				mAUC,
				average_precision,
				average_recall,
				access_type=access_type,
				csv_path=csv_path
				) 

		scheduler.step()

		finish_time = time.time()

		batch_time.reset()
		per_sample_time.reset()
		data_time.reset()
		per_sample_data_time.reset()
		loss_meter.reset()
		per_sample_dnn_time.reset()
		

def validate(audio_model, val_loader, train_conf, verbose, bar_generator):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	batch_time = AverageMeter()
	if not isinstance(audio_model, nn.DataParallel):
		audio_model = nn.DataParallel(audio_model)
	audio_model = audio_model.to(device)
	# switch to evaluate mode
	audio_model.eval()

	end = time.time()
	A_predictions = []
	A_targets = []
	A_loss = []
	A_max_certainty = []

	with torch.no_grad():
		for i, (audio_input, labels) in bar_generator(loader=val_loader, title="valid"):
			audio_input = audio_input.to(device)

			# compute output
			audio_output = audio_model(audio_input)
			###audio_output = torch.sigmoid(audio_output)
			smax = torch.nn.Softmax()
			audio_output = smax(audio_output)
			predictions = audio_output.detach().to('cpu')

			A_predictions.append(predictions)
			A_targets.append(labels)

			# compute the loss
			labels = labels.to(device)
			if isinstance(train_conf["loss_fn"], torch.nn.CrossEntropyLoss):
				loss = train_conf["loss_fn"](audio_output, torch.argmax(labels.long(), axis=1))
			else:
				loss = train_conf["loss_fn"](audio_output, labels)
			A_loss.append(loss.to('cpu').detach())

			batch_time.update(time.time() - end)
			end = time.time()

		audio_output = torch.cat(A_predictions)
		target = torch.cat(A_targets)
		loss = np.mean(A_loss)
		#stats = calculate_stats(output=audio_output, target=target)
		stats = []

	return stats, loss

def calculate_stats(output, target):
	"""Calculate statistics including mAP, AUC, etc.

	Args:
	  output: 2d array, (samples_num, classes_num)
	  target: 2d array, (samples_num, classes_num)

	Returns:
	  stats: list of statistic of each class.
	"""

	classes_num = target.shape[-1]
	stats = []

	# Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
	# Doc -> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
	acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

	# Class-wise statistics
	for k in range(classes_num):

		# Average precision
		# Doc -> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
		avg_precision = metrics.average_precision_score(
			target[:, k], output[:, k], average=None)

		# AUC
		# Doc -> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
		auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

		# Precisions, recalls
		# Doc -> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
		(precisions, recalls, thresholds) = metrics.precision_recall_curve(
			target[:, k], output[:, k])

		save_every_steps = 1000     # Sample statistics to reduce size
		dict = {'precisions': precisions[0::save_every_steps],
				'recalls': recalls[0::save_every_steps],
				'AP': avg_precision,
				'auc': auc,
				# note acc is not class-wise, this is just to keep consistent with other metrics
				'acc': acc
				}
		stats.append(dict)

	return stats
