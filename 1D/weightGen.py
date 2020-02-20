import tensorflow as tf
import numpy as np

data_type = tf.float32

#-------------------------- Mask Outputs from a to b ------------------------#
def weightCreation(start, end, wN):	# create arrays for frozen and trainable weights
	
	# ensure that start is not larger than end
	if start <= end:

		# if the starting index is greater than the starting index
		if start > 1:

			# create vector of "frozen" zeros representing free-space
			wL = tf.ones(shape = [start-1], dtype = data_type,name = 'left_weights')

		# otherwise if it is 1, there are no frozen weights less than the starting index
		elif start is 1:
			wL = None
		else:
			raise ValueError("Starting index, start, must be between 1 and Ending index, end")

		# if the ending index is less than the number of total weights
		if end < wN:

			# create vector of untrainable zeros to the right of the mask
			wR = tf.ones(shape = [wN - end], dtype = data_type,name = 'right_weights')

		# otherwise if it is the last index, no weights to the right of the trainable weights
		elif end is wN:
			wR = None
		else:
			raise ValueError("Ending index, start, must be between Starting index, start and (# of features)/2")

		# create trainable weights after creating untrainable weights
		#wT = tf.Variable(tf.reshape(tf.convert_to_tensor(starting_weights,dtype=tf.float32),[1,80]))
		wT = tf.Variable(tf.ones(shape = [end - start + 1], dtype = data_type),name = 'trainable_weights')

		return wL, wT, wR
	
	else:
		raise ValueError("Starting index must be smaller than or equal to ending index.")

		#-------------------------- Mask Outputs from a to b ------------------------#

#-------------------------- Create Total Array of Weights (Only the "Middle" is Trainable) ------------------------#
def weightConcat(wL, wT, wR):
	# masking starts at first spatial weight
	if wL is None:

		# masking ends at last spatial weight and starts at first (every weight is trainable)
		if wR is None:
			w = wT

		# masking ends before last spatial weight
		# so we have constant (untrainable) weights after the trainable weights in our row vector
		# i.e.: w = [#, #, #, #, 1, 1] where # is a trainable weight and 1 is the constant untrainable free space
		else:
			w = tf.concat([wT, wR], axis = 1,name =  'weight_tensor')

	# masking does not start at first spatial weight
	else:

		# masking does not start at first spatial weight but ends at last
		if wR is None:
			w = tf.concat([wL, wT], axis = 1,name =  'weight_tensor')

		# masking is somewhere between the first and last spatial weights
		else:
			w = tf.concat([wL, wT, wR], axis = 1,name =  'weight_tensor')

	return w

def weightConcat_static(wL, wT, wR):
	# masking starts at first spatial weight
	if wL is None:

		# masking ends at last spatial weight and starts at first (every weight is trainable)
		if wR is None:
			w = wT

		# masking ends before last spatial weight
		# so we have constant (untrainable) weights after the trainable weights in our row vector
		# i.e.: w = [#, #, #, #, 1, 1] where # is a trainable weight and 1 is the constant untrainable free space
		else:
			w = tf.concat([wT, wR], axis = 0,name =  'weight_tensor')

	# masking does not start at first spatial weight
	else:

		# masking does not start at first spatial weight but ends at last
		if wR is None:
			w = tf.concat([wL, wT], axis = 0,name =  'weight_tensor')

		# masking is somewhere between the first and last spatial weights
		else:
			w = tf.concat([wL, wT, wR], axis = 0,name =  'weight_tensor')

	return w


def weightCreationTimeDep(start, end, wN, timeN):	# create arrays for frozen and trainable weights
	
	# ensure that start is not larger than end
	if start <= end:

		# if the starting index is greater than the starting index
		if start > 1:

			# create vector of "frozen" ones representing free-space
			wL = tf.ones(shape = [timeN, start-1], dtype = data_type)

		# otherwise if it is 1, there are no frozen weights less than the starting index
		elif start is 1:
			wL= None
		else:
			raise ValueError("Starting index, start, must be between 1 and Ending index, end")

		# if the ending index is less than the number of total weights
		if end < wN:

			# create vector of untrainable ones to the right of the mask
			wR = tf.ones(shape = [timeN, wN - end], dtype = data_type)

		# otherwise if it is the last index, no weights to the right of the trainable weights
		elif end is wN:
			wR = None
		else:
			raise ValueError("Ending index, start, must be between Starting index, start and (# of features)/2")

		# create trainable weights after creating untrainable weights
		wT = tf.Variable(tf.ones(shape = [timeN, end - start + 1], dtype = data_type))

		return wL, wT, wR
	
	else:
		raise ValueError("Starting index must be smaller than or equal to ending index.")

