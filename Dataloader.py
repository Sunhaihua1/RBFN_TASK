import numpy as np
import pandas as pd
import torch
from RBFN import RBFN
import warnings
warnings.filterwarnings("ignore")

def load_data(src):
	data = pd.read_excel(src).values
	x_train = data[:, 0:10]
	y_train = data[:, 11]
	x_test = data[:, 14:24]
	x_train = torch.tensor(x_train, dtype=torch.float32)
	y_train = torch.tensor(y_train, dtype=torch.float32)
	x_test = torch.tensor(x_test, dtype=torch.float32)
	return x_train, y_train, x_test

