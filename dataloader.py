import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from PIL import Image
import numpy as np
from multiprocessing import Pool
import shutil
import pickle
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

def ping():
	print("Pong")

class DataLoader():

	def __init__(self, path, batch_size:int=0, val_size:float=0.2, shuffle:bool=False, verbose:bool=False):
		self.path = path
		self.batch_size = batch_size
		self.val_size = val_size
		self.shuffle = shuffle
		self.verbose = verbose

		self.unified_heights = False
		self.image_path = os.path.join(self.path, "ds", "images")
		self.label_path = os.path.join(self.path, "ds", "numified_labels")
		self.alph_label_path = os.path.join(self.path, "ds", "labels")
		self.unified_images_path = os.path.join(self.path, "unified_images")
		if not os.path.exists(self.unified_images_path):
			self.unified_images_path = None

	def get_max_image_height(self):
		imgs = os.listdir(os.path.join(self.image_path))
		height = np.array(Image.open(os.path.join(self.image_path, imgs[0]))).shape[0]
		for img in imgs:
			img_height = np.array(Image.open(os.path.join(self.image_path, img))).shape[0]
			if img_height > height:
				height = img_height
		return height

	def pad_image(self, image, max_height):
		opened_image = Image.open(os.path.join(self.image_path, image))
		width, height = opened_image.size
		padded_image = Image.new(opened_image.mode, (width, max_height), (255, 255, 255))
		padded_image.paste(opened_image, (0, (max_height - height) // 2))
		padded_image.save(os.path.join(self.unified_images_path, image))

	def unify_image_heights(self):
		if not os.path.exists(os.path.join(self.unified_images_path)):
			os.mkdir(os.path.join(self.unified_images_path))

		height = self.get_max_image_height()
		for idx, image in enumerate(os.listdir(self.image_path)):
			self.pad_image(image, height)
		print(f"Unified {idx + 1} image heights, {height} pixels tall.")
		self.unified_heights = True

	def unify_image_widths(self):
		if not os.path.exists(os.path.join(self.path, "unified_width")):
			os.mkdir(os.path.join(self.path, "unified_width"))
		widths = [Image.open(os.path.join(self.image_path, image)).size[0] for image in os.listdir(self.image_path)]
		avg_width = round(np.mean(widths))
		for idx, image in enumerate(os.listdir(self.image_path)):
			img = Image.open(os.path.join(self.image_path, image))
			resized_img = img.resize((avg_width, img.size[1]))
			resized_img.save(os.path.join(self.path, "unified_width", image))
		print(f"Reshaped {idx+1} images to ({avg_width}, 226).")

	def organize_primus(self, primus_path, dst):
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

	def build_alphabet(self, labels_path, write_alphabet:bool=False):
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

	def numify_labels(self):
		if not os.path.exists(os.path.join(self.path, "ds", "numified_labels")):
			os.mkdir(os.path.join(self.path, "ds", "numified_labels"))
		
		try:
			alphabet = open(os.path.join(self.path, "alphabet.txt")).read().split("\t")
		except:
			raise SystemExit(f"Could not find alphabet.txt file: {os.path.join(self.path, 'alphabet.txt')}")
		
		for idx, label in enumerate(os.listdir(self.alph_label_path)):
			opened_label = open(os.path.join(self.alph_label_path, label)).read().split("\t")
			numerical_encoding = []
			for note in opened_label:
				for jdx, element in enumerate(alphabet):
					if note == element:
						numerical_encoding.append(jdx)
			f = open(os.path.join(self.label_path, label.split(".")[0] + ".txt"), "w")
			f.write("0 " + " 0 ".join(str(num) for num in numerical_encoding) + " 0")
			f.close()
		if self.verbose:
			print(f"Numerified {idx + 1} labels into text files.")

	def gen_label_image_locs(self):
		image_file = open(os.path.join(self.path, "image_addresses.txt"), "w")
		label_file = open(os.path.join(self.path, "label_addresses.txt"), "w")
		for idx, name in enumerate(os.listdir(self.image_path)):
			image_file.write(os.path.join(self.image_path, name) + "\n")
			label_file.write(os.path.join(self.label_path, name.split(".")[0] + ".txt") + "\n")
		image_file.close()
		label_file.close()
		self.image_loc_path = os.path.join(self.path, "image_addresses.txt")
		self.label_loc_path = os.path.join(self.path, "label_addresses.txt")
		print(f"Wrote {idx + 1} image and label address locations to image_addresses.txt and label_addresses.txt files.")

	def get_ds(self):
		labels = []
		max_label_length = 0
		for name in os.listdir(self.label_path):
			with open(os.path.join(self.label_path, name), "r") as fp:
				label = fp.read().strip().split(" ")
				if len(label) > max_label_length:
					max_label_length = len(label)
				labels.append(np.array(label))
		for idx, label in enumerate(labels):
			amt = max_label_length - len(label)
			labels[idx] = np.pad(label, (0, amt), "constant", constant_values=("0", "0"))
		label_ds = tf.data.Dataset.from_tensor_slices(labels)
		#label_ds = tf.data.Dataset.from_tensor_slices(os.listdir(self.label_path))
		
		def process_path(path):
			return tf.image.decode_png(tf.io.read_file(path), channels=1)

		image_ds = tf.data.Dataset.list_files(str(pathlib.Path(self.image_path + "\\*.png")), shuffle=False)
		image_ds = image_ds.map(process_path)

		dataset = tf.data.Dataset.zip((image_ds, label_ds))
		# for x, y in dataset:
		# 	print(str(x.numpy()).split("\\")[-1].split(".")[0])
		# 	print(str(y.numpy())[2:].split(".")[0])
		# 	print()
		# print("########")
		if self.shuffle:
			dataset = dataset.shuffle(buffer_size=len(dataset))
		val_amt = round(self.val_size * len(dataset))
		train_ds = dataset.skip(val_amt)
		val_ds = dataset.take(val_amt)
		# for x, y in train_ds:
		# 	print(str(x.numpy()).split("\\")[-1].split(".")[0])
		# 	print(str(y.numpy())[2:].split(".")[0])
		# 	print()
		# raise SystemExit("Done.")
		if self.verbose:
			print(f"Loaded {len(train_ds) + len(val_ds)} images.")
		if self.batch_size:
			train_ds = train_ds.batch(self.batch_size, drop_remainder=True)
			val_ds = val_ds.batch(self.batch_size, drop_remainder=True)
			if self.verbose:
				print(f"Training dataset created: {self.batch_size * len(train_ds)} images, {len(train_ds)} batches of size {self.batch_size}.")
				print(f"Validation dataset created: {self.batch_size * len(val_ds)} images, {len(val_ds)} batches of size {self.batch_size}.")
		elif self.verbose:
			print(f"Training dataset created: {len(train_ds)} images.")
			print(f"Validation dataset created: {len(val_ds)} images.")
		
		x_train, y_train = list(map(list, zip(*[(x.numpy(), y.numpy()) for x, y in train_ds])))
		x_test, y_test = list(map(list, zip(*[(x.numpy(), y.numpy()) for x, y in val_ds])))

		x_train = np.array(x_train)
		y_train = np.array(y_train).astype(int)
		x_test = np.array(x_test)
		x_test = np.array(x_test).astype(int)

		return x_train, y_train, x_test, y_test

	def decode_numerical_label(self, labels):
		alphabet = open(os.path.join(self.path, "alphabet.txt")).read().split("\t")
		return [alphabet[int(idx)] for idx in labels if int(idx)]

	def get_alphabet(self):
		return open(os.path.join(self.path, "alphabet.txt")).read().split("\t")

if __name__ == "__main__":
	path = "C:\\Users\\hitts\\Documents\\GitHub\\piano"
	#path = "C:\\Users\\hitts\\Desktop"
	batch_size = 0
	val_size = 0.2
	shuffle = False
	verbose = False
	dl = DataLoader(path, batch_size, val_size, shuffle, verbose)
	train_ds, val_ds = dl.get_ds()
	image, label = next(iter(train_ds))
	print(dl.decode_numerical_label(label.numpy()))
	plt.imshow(image)
	plt.show()