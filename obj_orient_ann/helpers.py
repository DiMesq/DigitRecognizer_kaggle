import csv
import numpy as np 
import settings as s

def sigmoid(z):
	return 1/(1 + np.exp(-z))

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
		if has_header: next(reader)

		i = 0 #keeps track of which line we are in
		for row in reader:

			# read data
			data[i, :] = [int(ele) for ele in row]

			# normalize pixels to 0-1 range
			if has_labels:
				data[i, 1:] /= s.MAX_PIXEL_VAL
			else: 
				data[i, :] /= s.MAX_PIXEL_VAL

		return data

def split_data(data):
	''' Splits the data into training examples and labels.

		data: ndarray, first column is the labels in 0-9 range. The other 
			  columns for every row have the pixels

		returns : list, of two ndarrays with shapes (m, n) and (m,1) respectively.
				  The first corresponds to the training pixels and the second to
				  the labels -> [training_pixels, labels]'''

	return [data[:, 1:], data[:, 0:1]]






