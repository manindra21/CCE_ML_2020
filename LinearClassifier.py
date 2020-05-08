"""
 Logistic Regression for Iris Datset.

 There are three classes, Setosa class is linearly separable from the other two classes.
 Versicolor and Virginica classes are not linearly separable.
"""

import numpy
import csv
import pandas as pd
import matplotlib.pyplot as plt

numpy.seterr(all='ignore')


def loadCSV(filename):
	"""
	function to load dataset from the rgument.

	Args:
		filename: Take input as string which is a pth to the location where dataset located.

	Returns:
		Numpy Array.
	"""
	with open(filename, "r") as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for i in range(len(dataset)):
			dataset[i] = [float(x) for x in dataset[i]]
	return numpy.array(dataset)


def softmax(weightvector):
	"""
	Softmax function is an activation function which provides the aprior probability.

	Args:
		weightvector: A vector which consists the sum of all weight for each node.

	Returns:
		the aprioir probability of the node.

	"""
	e = numpy.exp(weightvector - numpy.max(weightvector))  # prevent overflow
	if e.ndim == 1:
		return e / numpy.sum(e, axis=0)
	else:
		return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2


"""
These are two global variables:
	objectStoreClassifier: It Stores the classifier class object
	classStoreClassifier: it stores the dictionary for the classes of the Dataset.
"""
objectStoreClassifier = None
classStoreClassifier = None

"""
LogisticRegression Class for Training and predicting the model.
"""


class LogisticRegression(object):
	def __init__(self, input, label, n_in, n_out):
		"""
		It initializes the class variable once object is created.
		:param input: The actual inputs or features of the model. Example: [5.9, 3.0, 5.1, 1.8].
		:param label: The target value corresponds to the input given to  model, which is as given:
						{0: 'versicolor', 1: 'virginica', 2: 'setosa'}.
		:param n_in: Number of features in the model. Iris Dataset has 4 features.
		:param n_out: Number of classes its mapping. Iris Dataset has 3 Classes.
		"""
		self.x = input
		self.y = label
		self.W = numpy.zeros((n_in, n_out))  # initialize W 0
		self.b = numpy.zeros(n_out)  # initialize bias 0

	def train(self, lr=0.005, input=None, L2_reg=0.01):
		"""
		It trains the model.
		:param lr: Learning rate for the model.
		:param input:  Model Input
		:param L2_reg: Regularization parameter
		"""
		if input is not None:
			self.x = input

		p_y_given_x = softmax(numpy.dot(self.x, self.W) + self.b)
		d_y = self.y - p_y_given_x

		self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
		self.b += lr * numpy.mean(d_y, axis=0)

	def negative_log_likelihood(self):
		"""
		It calculates the Cross Entropy, the loss function.
		:return: The cross entropy.
		"""
		softmax_activation = softmax(numpy.dot(self.x, self.W) + self.b)

		cross_entropy = - numpy.mean(
			numpy.sum(
				self.y * numpy.log(softmax_activation) + (1 - self.y) * numpy.log(1 - softmax_activation), axis=1))

		return cross_entropy

	def predict(self, weightedSum):
		"""
		This method predicts the probability of the class based on the weights calculated during training Process.
		:param weightedSum: The sum of all waits to apply the Activation function.
		:return: The probability for each class.
		"""
		return softmax(numpy.dot(weightedSum, self.W) + self.b)


def test_lr(x, y, learning_rate=0.005, n_epochs=500):
	"""
	This function trains the model based on number of epochs. Which is 500 for this model.
	:param x: The actual inputs or features of the model. Example: [5.9, 3.0, 5.1, 1.8].
	:param y: The target value corresponds to the input given to  model, which is as given:
						{0: 'versicolor', 1: 'virginica', 2: 'setosa'}.
	:param learning_rate: Learning rate which effects the gradient movement on the curve that how fast or slow
							the gradient moves towards the minima.
	:param n_epochs: Number of times the model go for forward and backward propagation. In each epoch the learning rate
					is also decreasing so that the gradient doesn't jump side by side instead of moving different
					slopes. the gradient converges fastly to the global minima.
	:return: None
	"""
	# construct LogisticRegression
	classifier = LogisticRegression(input=x, label=y, n_in=4, n_out=3)
	global objectStoreClassifier
	objectStoreClassifier = classifier

	# train
	for epoch in range(n_epochs):
		classifier.train(lr=learning_rate)
		cost = classifier.negative_log_likelihood()
		# print(sys.stderr, 'Training epoch %d, cost is ' % epoch, cost)
		learning_rate *= 0.95

	# predict(classifier)
	x = numpy.array([6.2, 2.8, 4.8, 1.8])
	values = classifier.predict(x)
	print(classStoreClassifier)
	print(classStoreClassifier[values.argmax()])


def predict(x):
	values = objectStoreClassifier.predict(x)
	print(classStoreClassifier)
	print(classStoreClassifier[values.argmax()])


if __name__ == "__main__":
	# Reading the dataset from the given location.
	iris = pd.read_csv(r'C:\Manindra\iris2.csv')

	# Here we are plotting petal_length vs petal_width
	# setosa - 'red'; versicolor - 'blue'; virginica - 'green'
	for n in range(0, 150):
		if iris['species'][n] == 'setosa':
			plt.scatter(iris['petal_length'][n], iris['petal_width'][n], color='red', label="Setosa")
			plt.xlabel('petal_length')
			plt.ylabel('petal_width')
		elif iris['species'][n] == 'versicolor':
			plt.scatter(iris['petal_length'][n], iris['petal_width'][n], color='blue', label="Versicolor")
			plt.xlabel('petal_length')
			plt.ylabel('petal_width')
		elif iris['species'][n] == 'virginica':
			plt.scatter(iris['petal_length'][n], iris['petal_width'][n], color='green', label="Virginia")
			plt.xlabel('petal_length')
			plt.ylabel('petal_width')

	# plt.show()
	# Here we are plotting Sepal_length vs Sepal_width
	# setosa - 'red'; versicolor - 'blue'; virginica - 'green'

	for n in range(0, 150):
		if iris['species'][n] == 'setosa':
			plt.scatter(iris['sepal_length'][n], iris['sepal_width'][n], color='red', label="Setosa")
			plt.xlabel('sepal_length')
			plt.ylabel('sepal_width')
		elif iris['species'][n] == 'versicolor':
			plt.scatter(iris['sepal_length'][n], iris['sepal_width'][n], color='blue', label="Versicolor")
			plt.xlabel('sepal_length')
			plt.ylabel('sepal_width')
		elif iris['species'][n] == 'virginica':
			plt.scatter(iris['sepal_length'][n], iris['sepal_width'][n], color='green', label="Virginia")
			plt.xlabel('sepal_length')
			plt.ylabel('sepal_width')

	# plt.show()

	features = numpy.array(iris.iloc[:, :4])
	label = iris['species'].tolist()

	# Finding number of classes in the Iris Dataset.
	x = set(iris['species'].tolist())
	classes = len(x)

	# Creating a dictionary for the Class and Integer values.
	res = {key: value for value, key in enumerate(x)}

	# Storing the dictionary in a global object.
	classStoreClassifier = {key: value for value, key in res.items()}

	# Creating one-hot encoder for the Dataset.
	onehotencodedvector = [[0 for x in range(classes)] for y in range(150)]
	for i in range(150):
		onehotencodedvector[i][res[label[i]]] = 1
	label = numpy.array(onehotencodedvector)

	# Calling the train function for training the model.
	test_lr(features, label)

	# Predicting the output of the model based on input vector.
	x = numpy.array([5.2, 3.5, 1.5, 0.2])
	predict(x)

