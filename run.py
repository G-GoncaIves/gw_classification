import argparse
import os
import sys
import torch
import torch.optim as optim
from jsonparse import json_to_parser
import importlib

class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()  
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


working_dir = os.path.dirname(os.path.realpath(__file__))
model_options= ["AST"]

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
subparsers = parser.add_subparsers(help="SubParsers command help:", dest="command")

for model in model_options:

	parser_json_path = os.path.join(working_dir, model, "parser.json")
	json_to_parser(parser_json_path, subparsers)

args = parser.parse_args()

model = importlib.import_module(args.command)

verbose = True if args.verbose else False

if args.data_type == "waveforms":

	model.train_model(
						dataloader = model.modules.My_DataLoader,
						train_function = model.modules.train,
						classifier = model.modules.My_Classifier,
						dataset = model.modules.WaveForms,
						args = args,
						verbose=verbose
			  		)

else:

	model.train_model(
						dataloader = model.modules.My_DataLoader,
						train_function = model.modules.train,
						classifier = model.modules.My_Classifier,
						dataset = model.modules.Spectrograms,
						args = args,
						verbose=verbose
			  		)
