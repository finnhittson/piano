import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, TimeDistributed, Dense, Softmax, Input, Permute, Reshape

WEIGHT_DECAY = 0.001
BATCH_SIZE = 32
IMAGE_HEIGHT = 226
IMAGE_WIDTH = 1320
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 1)

def get_model(num_classes):
	# CNN Layers for image processing
	input_img = Input(shape=IMAGE_SHAPE)
	conv_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(input_img)
	maxpool_1 = MaxPooling2D(2, padding='same')(conv_1)

	conv_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(maxpool_1)
	maxpool_2= MaxPooling2D(2, padding='same')(conv_2)

	conv_3 = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(maxpool_2)
	maxpool_3 = MaxPooling2D(2, padding='same')(conv_3)

	conv_4 = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(maxpool_3)
	maxpool_4 = MaxPooling2D(2, padding='same')(conv_4)

	reshaped = Reshape((-1, 256))(maxpool_4)

	# RNN/LSTM layers for sequence processing
	lstm_1 = LSTM(256, return_sequences=True)(reshaped)
	lstm_2 = LSTM(256, return_sequences=True)(lstm_1)
	dense_output = Dense(num_classes + 1, activation='relu')(lstm_2)

	output = Softmax()(dense_output)

	# Define the model
	model = Model(inputs=input_img, outputs=output, name="omr_model")

	# Define CTC loss function
	def ctc_loss(y_true, y_pred):
		print(y_true)
		print(y_pred)
		input_length = tf.ones(shape=(tf.shape(y_true)[0],), dtype=tf.int32) * 65
		label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=False, dtype=tf.int32)

		ctc_loss_value = tf.nn.ctc_loss(
			labels=y_true,
			logits=y_pred,
			label_length=label_length,
			logit_length=input_length,
			logits_time_major=False
		)
		# Return the mean loss across the batch
		return tf.reduce_mean(ctc_loss_value)

	model.compile(
		optimizer='adam', 
		loss=ctc_loss, 
	)
	print("Model compiled.")

	return model