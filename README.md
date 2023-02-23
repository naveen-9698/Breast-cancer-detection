# Breast-cancer-detection

The training data comprises 480 data points and the test set is 31 points.
The feature vector has dimension D=30. The data has 31 dimensions, with the first component being the label: 1 = Malignant, 2 = Benign.

A 2-class perceptron learning algorithm and classifier is used to train the model with sequential as well as stochiastic gradient descent.
The test data is trained using the trained weights from the classifier. Logistic regression is added as a comparison.

The challenging part of the project was to implement the entire learning algorithm using only numpy.( pandas for only reading the file)
