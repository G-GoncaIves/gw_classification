import argparse
import json


known_types = {
	"int" : int,
	"float" : float,
	"str" : str,
	"bool" : bool,
	"tuple" : tuple,
	"list" : list
	}

def json_to_parser(json_path, subparsers):

	with open(json_path) as jfile:
		data = json.load(jfile)
		
		parser_title = data["name"]
		parser_description = data["description"]
		parser_help = data["help"]
		parser_args = data["args"]

	subparser = subparsers.add_parser(
						name=parser_title,
						description=parser_description,
						help=parser_help,
					   )	

	for arg in parser_args:

		arg["name"] = "--" + arg["name"] # TODO

		if "action" in arg.keys():

			subparser.add_argument(
							arg["name"], 
							action=arg["action"], 
							help=arg["help"]
						)

		else:

			subparser.add_argument(
							arg["name"],
							type=known_types[arg["type"]],
							help=arg["help"],
							metavar=arg["metavar"]
					 	)

