import os
from PIL import Image
import numpy as np
from multiprocessing import Pool
import shutil

class DataLoader():

	def __init__(self, path):
		self.unified_heights = False
		self.path = path
		self.image_path = os.path.join(self.path, "images")
		self.unified_images_path = os.path.join(self.path, "unified_images")

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

	def get_random_image(self):
		images = os.listdir(self.image_path)
		return images[np.random.randint(0, len(images))]

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

if __name__ == "__main__":
	path = "C:\\Users\\hitts\\Desktop\\ds"
	#path = "C:\\Users\\hitts\\Documents\\GitHub\\piano\\primus_ds"
	dl = DataLoader(path)
	dl.unify_image_heights()
