import csv
import numpy as np

def read_data(filename, nrows):

	with open(filename, 'rt') as csvfile:

		reader = csv.reader(csvfile)
		next(reader) #skip header row

		data = np.zeros( (nrows, 785), dtype=np.float64 )
		i = 0
		for row in reader:
			row_ints = [int(x) for x in row]
			data[i, :] = row_ints
			i+=1

	return data

def parse_data(array, max_value):
	''' Changes original array
		Splits data into features and output.
		Adds a column of 1's to the feature array.'''

		Y = array[:, 0].copy()
		array[:, 0] = 1

		array = array / max_value

	return arrray, Y

def sigmoid(z):
	return 1 / ( 1 + np.exp(-z))





