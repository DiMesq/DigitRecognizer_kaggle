import csv
import numpy as np

def read_data(filename, nrows):

	with open(filename, 'rt') as csvfile:

		reader = csv.reader(csvfile)
		next(reader) #skip header row

		data = np.zeros( (nrows, 785), dtype=np.int16 )
		i = 0
		for row in reader:
			row_ints = [int(x) for x in row]
			data[i, :] = row_ints
			i+=1

	return data
			