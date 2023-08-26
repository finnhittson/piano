import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import os

class Sound:
	SAMPLE_FREQ = 44100
	THE_BOB_HOPE_FACTOR = 3

	lengths = {
		'half':1/2, 
		'quarter':1/4, 
		'eighth':1/8, 
		'sixteenth':1/16, 
		'whole':1.0, 
		'double':2.0, 
		'quadruple':4.0, 
		'thirty':30.0, 
		'sixty':60.0, 
		'hundred':100.0
	}

	def __init__(self):
		self.present_notes = {}
		if os.path.exists("C:\\Users\\hitts\\Documents\\GitHub\\piano\\sound.wav"):
			os.remove("C:\\Users\\hitts\\Documents\\GitHub\\piano\\sound.wav")
		if os.path.exists("C:\\Users\\hitts\\Documents\\GitHub\\piano\\sound_data.txt"):
			os.remove("C:\\Users\\hitts\\Documents\\GitHub\\piano\\sound_data.txt")

	def get_present_notes(self):
		return self.present_notes

	def add_present_note(self, note, freq):
		self.present_notes[note] = freq

	def get_note_frequency(self, note):
		sharp = 1 if "#" in note else 0
		octave_number = int(note[-1])
		main_note_number = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}[note[0]]
		if octave_number == 0:
			note_difference = 0 + sharp + (2 if "B" in note else 0)
		else:
			note_difference = main_note_number + sharp + 3 + 12 * (octave_number - 1)
		freq = 27.5 * pow(2, note_difference / 12)
		if note not in self.get_present_notes().keys():
			self.add_present_note(note, freq)
		return freq

	def synth_sound(self):
		f = open("sound_data.txt", "r")
		data = np.array([float(val) for val in f.read().split(" ")[:-1]]).astype("float32")
		write("sound.wav", self.SAMPLE_FREQ, data)

	def read_labels(self, path):
		f = open(path, "r")
		labels = f.read().split("\t")
		for label in labels:
			func = label.split("-")[0]
			if func == "barline":
				continue
				print("barline()")
			elif func == "clef":
				continue
				print("clef()")
			elif func == "gracenote":
				continue
				print("gracenote()")
			elif func == "keySignature":
				continue
				print("keySignature()")
			elif func == "multirest":
				continue
				print("multirest()")
			elif func == "note":
				note = label.split("-")[1].split("_")[0]
				length = self.lengths[label.split("-")[1].split("_")[1]]
				self.note(note, length * self.THE_BOB_HOPE_FACTOR)
			elif func == "rest":
				length = self.lengths[label.split("-")[1]]
				self.rest(length * self.THE_BOB_HOPE_FACTOR)
			elif func == "tie":
				continue
				print("tie()")
			elif func == "timeSignature":
				continue
				print("timeSignature()")
		self.synth_sound()

	def barline(self):
		#raise NotImplementedError
		pass

	def clef(self):
		raise NotImplementedError

	def gracenote(self, note, length):
		freq = notes[note]
		note_length = lengths[length] * NOTE_LENGTH
		return [np.sin(2 * np.pi * freq * t) for t in range(note_length)]
		
	def key_signature(self):
		raise NotImplementedError

	def multirest(self, length):
		return np.zeros(lengths[length] * NOTE_LENGTH)

	def note(self, note, length:int=1.0):
		freq = self.get_note_frequency(note)
		times = np.linspace(0, length, int(self.SAMPLE_FREQ * length))
		sound_data = open("sound_data.txt", "a")
		for t in times:
			sound_data.write(f"{np.sin(2 * np.pi * freq * t)} ")
		count = 0
		while round(np.sin(2 * np.pi * freq * t), 2) != 0:
			t += 1 / self.SAMPLE_FREQ
			sound_data.write(f"{np.sin(2 * np.pi * freq * t)} ")
		sound_data.close()

	def rest(self, length):
		times = np.linspace(0, length, int(self.SAMPLE_FREQ * length))
		sound_data = open("sound_data.txt", "a")
		for t in times:
			sound_data.write(f"0 ")
		sound_data.close()

	def tie(self):
		raise NotImplementedError

	def time_signature(self):
		raise NotImplementedError

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument(
		"-note",
		type=str)	
	args = argparser.parse_args()
	sound = Sound()
	path = "C:\\Users\\hitts\\Documents\\GitHub\\piano\\primus_ds\\semantic_labels\\000051650-1_1_1.semantic"
	sound.read_labels(path)