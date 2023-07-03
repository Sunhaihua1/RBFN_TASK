import numpy as np
import pandas as pd
from RBFN import RBFN
def load_data(src):
	data = pd.read_excel(src).values
	x_train = data[:, 0:10]
	y_train = data[:, 11]
	x_test = data[:, 14:24]
	return x_train, y_train, x_test

if __name__ == '__main__':
	x_train, y_train, x_test = load_data('Problem1.xlsx')
	model = RBFN(10, 10)
	# print(x_train.shape, y_train.shape, x_test.shape)
	model.fit(x_train, y_train)  
	# print(model.centers)
	final = model.predict(x_test)
	print(final)