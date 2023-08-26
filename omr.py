import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

WEIGHT_DECAY = 0.001

def max_pooling2d(x):
	return layers.MaxPool2D(
		pool_size = (2, 2),
		strides = (1, 1),
		padding = "same"
	)(x)

def conv2d(x, channels):
	return layers.Conv2D(
		filter = channels,
		kernel = (3, 3),
		padding = "same",
		kernel_regularizer = regularizers.l2(WEIGHT_DECAY)
	)(x)

def lstm(x):
	return layers.LSTM(
		units = 256,
		activation = "relu",
		return_sequences = True,
		kernel_regularizer = regularizers.l2(WEIGHT_DECAY)
	)(x)

def get_model(alphabet_length):
	inputs = keras.Input(shape=(128, None, 1))

	# Convolution block
	x = conv2d(inputs, channels=32)
	x = max_pooling2d(x)
	x = conv2d(x, channels=64)
	x = max_pooling2d(x)
	x = conv2d(x, channels=128)
	x = max_pooling2d(x)
	x = conv2d(x, channels=256)
	x = max_pooling2d(x)

	# Recurrent block
	x = lstm(x)
	x = lstm(x)
	x = layers.Dense(
		units = alphabet_length,
		activation = 'relu',
		kernel_regularizer = regularizers.l2(WEIGHT_DECAY)
	)(x)
	outputs = layers.Softmax()(x)
	model = keras.Model(inputs=inputs, outputs=ouputs)

	# Compile model
	model.compile(
		loss = tf.nn.ctc_loss(),
		optimizer = keras.optimizers.Adam(lr=0.001),
		metrics = ["accuracy"]
	)
