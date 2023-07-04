import numpy as np
from sklearn.cluster import KMeans

class RBFN:
	def __init__(self,input_size, kernal_size) -> None:
		self.input_size = input_size
		self.kernal_size = kernal_size
		self.sigma = None
		self.w = None
		self.b = 0
		self.centers = None
		# self._kernal_function = lambda center, x_point: np.exp(-0.5 * np.power(np.linalg.norm(center-x_point) / self.sigma, 2)) # gaussian kernal
		self._kernal_function = lambda center, x_point: np.exp(-0.5 * np.sqrt(np.sum(np.square(x_point - center))) / np.square(self.sigma)) # gaussian kernal

	def _calculate_interpolation_matrix(self, X):
		G = np.zeros((X.shape[0], self.kernal_size))
		for i in range(X.shape[0]):
			for j in range(self.kernal_size):
				G[i, j] = self._kernal_function(self.centers[j], X[i])
		return G

	def _calculate_sigma(self):
		max_dis = 0
		num = 0
		total = 0
		for i in range(self.kernal_size - 1):
			for j in range(i + 1, self.kernal_size):
				dis = np.linalg.norm(self.centers[i] - self.centers[j])
				total += dis
				max_dis = max(max_dis, dis)
				num += 1
		self.sigma = 2 * total / num
	def fit(self, X, Y) :
     	# step 1: initialize centers
		self.centers = KMeans(n_clusters=self.kernal_size).fit(X).cluster_centers_

		# step 2: calculate sigma
		dmax = max([np.linalg.norm(self.centers[i]-self.centers[j]) for i in range(self.kernal_size) for j in range(self.kernal_size)])
		print(self.centers.shape)
		# self.sigma = dmax / np.sqrt(2*self.centers.shape[0])
		self._calculate_sigma()
		# print(self.sigma)
		# print(1 / (2 * (self.sigma ** 2)))

		# step 3: calculate interpolation matrix
		G = self._calculate_interpolation_matrix(X)
  
		# step 4: calculate weights
		temp = np.ones((len(X)))
		temp = np.column_stack((G, temp))
		temp = np.dot(np.linalg.pinv(temp), Y)
		self.w = temp[:self.kernal_size]
		self.b = temp[self.kernal_size]
		# self.w = np.dot(np.linalg.pinv(G), Y)
	
	def predict(self, X):
		G = self._calculate_interpolation_matrix(X)
		return np.dot(G, self.w) + self.b
