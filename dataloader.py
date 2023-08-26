import os
import numpy as np
from multiprocessing import Pool
import shutil

def organize_primus(primus_path, dst):
	if not os.path.exists(dst):
		os.mkdir(dst)
	if not os.path.exists(os.path.join(dst, "images")):
		os.mkdir(os.path.join(dst, "images"))
	if not os.path.exists(os.path.join(dst, "agnostic_labels")):
		os.mkdir(os.path.join(dst, "agnostic_labels"))
	if not os.path.exists(os.path.join(dst, "semantic_labels")):
		os.mkdir(os.path.join(dst, "semantic_labels"))
	for subdir in os.listdir(primus_path):
		if os.path.isdir(os.path.join(primus_path, subdir)):
			for file in os.listdir(os.path.join(primus_path, subdir)):
				foi = os.path.join(primus_path, subdir, file)
				if file[0] != "." and file.split(".")[-1] == "png" and False:
					shutil.copy(foi, os.path.join(dst, "images"))
				elif file.split(".")[-1] == "agnostic":
					shutil.copy(foi, os.path.join(dst, "agnostic_labels"))
				elif file.split(".")[-1] == "semantic" and False: 
					shutil.copy(foi, os.path.join(dst, "semantic_labels"))

def build_alphabet(labels_path, write_alphabet:bool=False):
	if not os.path.exists(labels_path) and not os.path.isdir(labels_path):
		raise SystemExit("Not a valid path.")
	alphabet = []
	for file in os.listdir(labels_path):
		f = open(os.path.join(labels_path, file))
		characters = f.read().split("\t")
		alphabet += list(np.unique(characters))
		f.close()
	alphabet = list(np.unique(alphabet))
	del_idx = 0
	for idx in range(len(alphabet)):
		if alphabet[idx-del_idx] == "" or alphabet[idx-del_idx] == "\t" or alphabet[idx-del_idx] == " ":
			del alphabet[idx - del_idx]
			del_idx += 1
	if write_alphabet:
		f = open(os.path.join(labels_path, "alphabet.txt"), "w")
		f.write("\t".join(alphabet))
		f.close()

	return alphabet

if __name__ == "__main__":
	data_dir = "C:\\Users\\hitts\\Documents\\Github\\piano\\primus_ds\\semantic_labels"
	alphabet = build_alphabet(data_dir, write_alphabet=True)
	print(len(alphabet))