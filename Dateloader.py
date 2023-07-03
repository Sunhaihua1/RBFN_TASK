import numpy as np
import pandas as pd

def load_data(src):
	data = pd.read_excel(src).values
	x_train = data[:, 0:10]
	y_train = data[:, 11]
	x_test = data[:, 14:24]
	return x_train, y_train, x_test

if __name__ == '__main__':
	x_train, y_train, x_test = load_data('Problem2.xlsx')
	print(x_train.shape, y_train.shape, x_test.shape)