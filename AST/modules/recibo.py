def recibo(
				start_time,
				current_time,
				
				epoch, 
				train_loss, 
				val_loss, 
				_map, 
				acc,
				auc,
				avg_pre,
				avg_recall,

				best_epoch, 
				best_train_loss, 
				best_val_loss, 
				best_map,
				best_acc,
				best_auc,
				best_avg_pre,
				best_avg_recall,

				peak_cuda_mem,

				current_state_loc, 
				best_state_loc,
				side=None,

				recibo_location=None
			):

	
	str_list = []

	str_list.append("")
	str_list.append(" Time:")
	elapsed_time = current_time - start_time
	avg_epoch_time = elapsed_time / epoch
	str_list.append(f"    Start = {start_time}")
	str_list.append(f"    Current = {current_time}")
	str_list.append(f"    Elapsed = {elapsed_time}")
	str_list.append(f"    Avg Time/Epoch = {avg_epoch_time}")

	str_list.append("")
	str_list.append(f" Current Stats:")
	str_list.append(f"    Epoch = {epoch}")
	str_list.append(f"    Train Loss = {train_loss:.4f}")
	str_list.append(f"    Valid Loss = {val_loss:.4f}")
	str_list.append(f"    Mean AP = {_map:.4f}")
	str_list.append(f"    Accurracy = {acc:.4f}")
	str_list.append(f"    AUC = {auc:.4f}")
	str_list.append(f"    Avg Precision = {avg_pre:.4f}")
	str_list.append(f"    Avg Recall = {avg_recall:.4f}")

	str_list.append("")
	str_list.append(f" Best Stats:")
	str_list.append(f"    Epoch = {best_epoch}")
	str_list.append(f"    Train Loss = {best_train_loss:.4f}")
	str_list.append(f"    Valid Loss = {best_val_loss:.4f}")
	str_list.append(f"    Mean AP = {best_map:.4f}")
	str_list.append(f"    Accurracy = {best_acc:.4f}")
	str_list.append(f"    AUC = {best_auc:.4f}")
	str_list.append(f"    Avg Precision = {best_avg_pre:.4f}")
	str_list.append(f"    Avg Recall = {best_avg_recall:.4f}")

	str_list.append("")
	str_list.append(" Resource Usage:")
	str_list.append(f"      Peak CUDA mem (Gloabl)  = {peak_cuda_mem}")

	str_list.append("")
	str_list.append(f" Saved State Locations:")
	str_list.append(f"    Current State Loc = {current_state_loc}")
	str_list.append(f"    Best State Loc = {best_state_loc}")
	
	str_list.append("")

	if not side:

		side = max([len(s) for s in str_list]) + 4

	top = bottom = "="*side

	recibo = "\n\t" + top + "\n" 
	
	for line in str_list:

		line_len = len(line)
		recibo += "\t|" + line + " "*(side - line_len -2) + "|" + "\n"

	recibo += "\t" + bottom + "\n"

	with open(recibo_location, "w") as f:
		f.write(recibo)

def progress_csv(
		*args,
		access_type=None,
		csv_path=None
		):
	
	line_string = ""
	for arg in args[:-1]:
		line_string += f"{arg};"
	line_string += f"{args[-1]}\n"

	with open(csv_path, access_type) as f: 
		f.write(line_string)


if __name__ == "__main__":
	
	import os

	recibo_path = os.path.join("/home/goncalo/", "recibo.txt")
	recibo(
		start_time = 0,
		current_time = 0,
		epoch = 100,
		train_loss = 0,
		val_loss = 0,
		_map = 0,
		acc = 0,
		auc = 0,
		avg_pre = 0,
		avg_recall = 0,
		best_epoch = 0,
		best_train_loss = 0,
		best_val_loss = 0,
		best_map = 0,
		best_acc = 0,
		best_auc = 0,
		best_avg_pre = 0,
		best_avg_recall = 0,
		current_state_loc = 0,
		best_state_loc = "0000000000000000000000000000000000000000000000000000000000000000000000000",
		recibo_location = recibo_path
		)
