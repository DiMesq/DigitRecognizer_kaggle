from math import exp
import csv
import numpy as np 

MAX_PIXEL_VAL = 255

def sigmoid(z):
	return 1/(1 + exp(-z))

def read_pixels(filedir, n_rows, n_col, has_header, has_labels):
	''' filedir: string, path to csv file. Each element in a row must correspond to the 
				pixels of a single image, where each pixel ranges from 0-255. 
				If this file corresponds to the training data, the labels must be in the 
				first column. The labels range from 0 to 9.

		n_rows: int, number of rows the file has 
				(Important: if file has header, don't include this line in the number of rows)
		n_col: int, number of columns the file has
		has_header: boolean, True if file has header, False otherwise
		has_labels: boolean, True if it has labels according to description in the filedir 
					argument. False otherwise.

		returns: (n_rows, n_col) ndarray, the file content has a numpy array''' 

	# init ndarray to store file data
	data = np.zeros((n_rows, n_col))

	# start reading the file
	with open(filedir, 'rt') as csvfile:
		reader = csv.reader(csvfile)

		# skip header line
		if has_header: reader.next()

		i = 0 #keeps track of which line we are in
		for row in reader:

			# read data
			data[i, :] = [int(ele) for ele in row]

			# normalize pixels to 0-1 range
			data[i, 1:] /=  MAX_PIXEL_VAL if has_labels else data[i, :] /= MAX_PIXEL_VAL

		return data



