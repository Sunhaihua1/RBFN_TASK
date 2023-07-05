from RBFN import RBFN
from Dataloader import load_data
import pandas as pd
import numpy as np

if __name__ == '__main__':
	x_train, y_train, x_test = load_data('Problem2.xlsx')
	model = RBFN(10, 6)
	model.fit(x_train, y_train)  
	final = model.predict(x_test)
	pd.DataFrame(final).to_excel('Problem2_result.xlsx')
	
	print(final)