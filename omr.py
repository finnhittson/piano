import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, TimeDistributed, Dense, Softmax, Input, Permute

WEIGHT_DECAY = 0.001
BATCH_SIZE = 32
IMAGE_HEIGHT = 226
IMAGE_WIDTH = 1320
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 1)

def get_model(num_classes):
	# CNN Layers for image processing
	input_img = Input(shape=IMAGE_SHAPE)
	cnn_output = Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(input_img)
	cnn_output = MaxPooling2D(2, strides=1, padding='same')(cnn_output)

	cnn_output = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(cnn_output)
	cnn_output = MaxPooling2D(2, strides=1, padding='same')(cnn_output)

	cnn_output = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(cnn_output)
	cnn_output = MaxPooling2D(2, strides=1, padding='same')(cnn_output)

	cnn_output = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(cnn_output)
	cnn_output = MaxPooling2D(2, strides=1, padding='same')(cnn_output)

	# RNN/LSTM layers for sequence processing
	rnn_output = TimeDistributed(LSTM(256, return_sequences=True))(cnn_output)
	rnn_output = TimeDistributed(LSTM(256, return_sequences=True))(rnn_output)
	rnn_output = TimeDistributed(Dense(num_classes + 1, activation='relu'))(rnn_output)

	output = Softmax()(rnn_output)

	# Define the model
	model = Model(inputs=input_img, outputs=output, name="Optical Music Recognition Model")

	# Define CTC loss function
	def ctc_loss(y_true, y_pred):
		label_length = tf.fill([tf.shape(y_true)[0]], tf.shape(y_pred)[1])
		ctc_loss_value = tf.nn.ctc_loss(
			labels=y_true,
			logits=y_pred,
			label_length=label_length,
			logit_length=label_length
		)
		# Return the mean loss across the batch
		return tf.reduce_mean(ctc_loss_value)

	model.compile(optimizer='adam', loss=ctc_loss)
	print("Model compiled.")

	return model