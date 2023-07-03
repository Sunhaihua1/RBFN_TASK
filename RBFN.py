import numpy as np
from sklearn.cluster import KMeans

class RBFN:
	def __init__(self,input_size, kernal_size) -> None:
		self.input_size = input_size
		self.kernal_size = kernal_size
		self.sigma = None
		self.w = None
		self.b = None
		self.centers = None
		self._kernal_function = lambda center, x_point: np.exp(-self.sigma * np.linalg.norm(center-x_point)**2) # gaussian kernal

	def _calculate_interpolation_matrix(self, X):
		G = np.zeros((X.shape[0], self.kernal_size))
		for i in range(X.shape[0]):
			for j in range(self.kernal_size):
				G[i, j] = self._kernal_function(self.centers[j], X[i])
		return G

	def fit(self, X, Y) :
     	# step 1: initialize centers
		self.centers = KMeans(n_clusters=self.kernal_size).fit(X).cluster_centers_

		# step 2: calculate sigma
		dmax = max([np.linalg.norm(self.centers[i]-self.centers[j]) for i in range(self.kernal_size) for j in range(self.kernal_size)])
		self.sigma = dmax / np.sqrt(2*self.kernal_size)

		# step 3: calculate interpolation matrix
		G = self._calculate_interpolation_matrix(X)

		# step 4: calculate weights
		self.w = np.dot(np.linalg.pinv(G), Y)
	
	def predict(self, X):
		G = self._calculate_interpolation_matrix(X)
		return np.dot(G, self.w)
