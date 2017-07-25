import numpy as np
import matplotlib.pyplot as plt

x1_min, x1_max, x2_min, x2_max = -10, 10, -10, 10
n = 1000
x1s, x2s = np.linspace(x1_min, x1_max, n), np.linspace(x2_min, x2_max, n)
x1s, x2s = np.meshgrid(x1s, x2s)
X = np.concatenate((np.ones((n, n, 1)), x1s.reshape(n, n, 1), x2s.reshape(n, n, 1)), axis=2)

def binary_classifier():
	w = np.random.rand(3, 1)
	Y = np.tensordot(X, w, axes=(2, 0))
	# Compute sigmoid
	Y = 1 / (1+np.exp(-Y))
	# Get the class
	Y = Y >= 0.5
	# Plot
	Y = Y.reshape(Y.shape[0], Y.shape[1])
	plt.imshow(Y, extent=[x1_min, x1_max, x2_min, x2_max], vmin=0, vmax=1, origin='lower', cmap='jet')
	plt.colorbar()
	plt.show()

def multiclass_classifier(nb_classes=7):
	W = np.random.rand(3, nb_classes)
	Y = np.tensordot(X, W, axes=(2, 0))
	# Compute softmax
	exp_Y = np.exp(Y)
	sum_exp_Y = np.sum(exp_Y, axis=2).reshape(exp_Y.shape[0], exp_Y.shape[1], 1)
	Y = exp_Y / sum_exp_Y
	# Get the class
	Y = np.argmax(Y, axis=2)
	# Plot
	plt.imshow(Y, extent=[x1_min, x1_max, x2_min, x2_max], vmin=0, vmax=nb_classes-1, origin='lower', cmap='jet')
	plt.colorbar()
	plt.show()

binary_classifier()
multiclass_classifier()