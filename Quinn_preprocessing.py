import numpy as np
import pandas as pd
import random
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score

# So that we may see all columns
pd.set_option('display.max_columns', None)


def preprocess_data():
    # Read in the data
    data = pd.read_csv("weatherAUS.csv")

    # Convert 9pm/3pm data values to "delta" values and drop the originals from the dataframe
    data["deltaWindSpeed"] = data["WindSpeed3pm"] - data["WindSpeed9am"]
    data["deltaHumidity"] = data["Humidity3pm"] - data["Humidity9am"]
    data["deltaPressure"] = data["Pressure3pm"] - data["Pressure9am"]
    data["deltaCloud"] = data["Cloud3pm"] - data["Cloud9am"]
    data["deltaTemp"] = data["Temp3pm"] - data["Temp9am"]
    data["RainToday"] = np.where(data["RainToday"] == "Yes", 1, 0)
    data["RainTomorrow"] = np.where(data["RainTomorrow"] == "Yes", 1, 0)
    data["Date"] = pd.to_datetime(data["Date"])   #Converting into Date
    data["Year"] = data["Date"].dt.year           #abstracting year in different column 
    data["Month"] = data["Date"].dt.month         #abstracting month in diffrent column  
    data["Day"] = data["Date"].dt.day             #abstracting day in diffrent column 
    data = data.drop(
        ["Location", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
         "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "Date"], axis=1)

    data = data.interpolate()

    # http: // www.land - navigation.com / boxing - the - compass.html
    d = {'N': 0, 'NNE': 1, 'NE': 2, 'ENE': 3, 'E': 4, 'ESE': 5, 'SE': 6, 'SSE': 7, 'S': 8, 'SSW': 9, 'SW': 10,
         'WSW': 11, 'W': 12, 'WNW': 13, 'NW': 14, 'NNW': 15}

    data['WindGustDir'] = data['WindGustDir'].map(d)
    data['WindDir9am'] = data['WindDir9am'].map(d)
    data['WindDir3pm'] = data['WindDir3pm'].map(d)

    data['WindDirDelta'] = (data['WindDir3pm'] - data['WindDir9am']) % 16

    # Remove rows with NA values from the data
    dataNoNA = data.dropna()

    #dateData = pd.to_datetime(dataNoNA['Date'])
    #dataNoNA = dataNoNA.drop(["Date"], axis=1)

    # Separate labels from the data
    labels = dataNoNA["RainTomorrow"]
    dataNoNA = dataNoNA.drop(["RainTomorrow"], axis=1)

    # Standardization
    dataNoNA = (dataNoNA - dataNoNA.mean()) / dataNoNA.std()

    #dataNoNA["Date"] = dateData
    print("There are a total of ", len(data), "instances in our dataset.")
    print("\n")
    print("No Rain (0) vs Rain (1)")
    print(data["RainTomorrow"].value_counts())
    print("\n")

    # Train-test split (80-20)
    x_train, x_test, y_train, y_test = train_test_split(dataNoNA, labels, test_size=0.2, random_state=479)

    return [x_train, x_test, y_train, y_test]


train_test_list = preprocess_data()

x_validate = train_test_list[0]
y_validate = train_test_list[2]
###
x_test = train_test_list[1]
y_test = train_test_list[3]

#Quinn: you are doing KNN and Adaline

x_validate = x_validate.to_numpy()
y_validate = y_validate.to_numpy()


#Here is where we test KNN parameters and do our grid search on the validation set
#This function took three hours to run
def validateKNN(x_validate, y_validate):
	#kf = KFold(n_splits=10)
	parameters = {'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'), 'weights': ('uniform', 'distance'), 'n_neighbors': [10, 25, 50, 100]}
	model = KNeighborsClassifier()
	grid = GridSearchCV(model, parameters, cv=10)
	grid.fit(x_validate, y_validate)
	print(grid.best_params_)

	

print("KNN Best Parameters: ")
#validateKNN(x_validate, y_validate)
print("{'algorithm': 'auto', 'n_neighbors': 20, 'weights': 'distance'}")
print('\n')

#Here is where we make predictions on our test set using best parameters from before
def predictKNN(x_validate, y_validate, x_test, y_test):
	model = KNeighborsClassifier(n_neighbors=20, algorithm='auto', weights='distance')
	model.fit(x_validate, y_validate)
	y_pred = model.predict(x_test)
	acc_score = accuracy_score(y_test, y_pred)
	f1Score = f1_score(y_test, y_pred, average='macro')
	print('Accuracy: %.3f' % (mean(acc_score)))
	print('F1: %.3f' % (mean(f1Score)))

print("KNN Performance Statistics: ")
print('\n')
predictKNN(x_validate, y_validate, x_test, y_test)
print('\n')



class AdalineGD(object):
    """ADAptive LInear NEuron classifier.
    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    random_state : int
    Random number generator seed for random weight initialization.
    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    cost_ : list
    Sum-of-squares cost function value in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=1000, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to zeros"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, 0)
    def accuracy(self, y_test, y_pred):
        acc = accuracy_score(y_test, y_pred, normalize=True)
        return acc



##Adaline is not available in sklearn so I implemented the textbooks version of Adaline.


n_iter = [10, 30, 50, 100] 
etas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

def make_N_folds(X_train, Y_train, numSplit):
	n = numSplit
	X_train_copy = list(X_train)
	Y_train_copy = list(Y_train)
	lenFold = int(len(X_train)/n)
	x_list = []
	y_list = []
	for x in range(numSplit):
		x_fold = []
		y_fold = []
		for x in range(lenFold):
			points = random.randrange(len(X_train_copy))
			curr_x_fold = X_train_copy.pop(points)
			curr_y_fold = Y_train_copy.pop(points)
			x_fold.append(curr_x_fold)
			y_fold.append(curr_y_fold)
		x_list.append(x_fold)
		y_list.append(y_fold)
	return x_list, y_list


X_list, Y_list = make_N_folds(x_validate, y_validate, 10)
#fold_list = np.array(kfolds)

def N_fold_validation(X_list, Y_list, numFolds, val, eta):
	acc_scores = []
	f1_scores = []
	#print(X_list)
	#print(Y_list)
	for i in range(numFolds):
		copy_XList = X_list
		copy_YList = Y_list
		Xtest = copy_XList[i]
		Ytest = copy_YList[i]
		Xtrain = np.delete(copy_XList, i, 0)
		Xtrain = Xtrain.reshape(-1, 19)
		Ytrain = np.delete(copy_YList, i, 0)
		Ytrain = Ytrain.flatten()

		Adaline = AdalineGD(n_iter=val, eta=eta)
		Adaline.fit(Xtrain, Ytrain)
		y_pred_m = Adaline.predict(Xtest)
		#print(y_pred_m)
		#print(Ytest)
		acc_score_m = Adaline.accuracy(Ytest, y_pred_m)
		f1Score_m = f1_score(Ytest, y_pred_m, average='macro')
		acc_scores.append(acc_score_m)
		f1_scores.append(f1Score_m)

	avg_acc = sum(acc_scores)/float(len(acc_scores))
	avg_f1 = sum(f1_scores)/float(len(f1_scores))
	return avg_acc, avg_f1

def grid_search(n_iter, etas, X_list, Y_list):
	average_acc_list = []
	average_f1_list = []
	indexing = []
	#We are just going to nest a bunch of for loops
	#First is the C for loop:
	for val in n_iter:
		#Second is the Penalty loop:
		#for pen in Penalty_dict:
			#Last is the Solver loop:
		for eta in etas:
			average_acc, average_f1 = N_fold_validation(X_list, Y_list, 10, val, eta)
			average_acc_list.append(average_acc)
			average_f1_list.append(average_f1)
			indexing.append([val, eta])
	return average_acc_list, average_f1_list, indexing 

#acc_list, f1_list, idx_list = grid_search(n_iter, etas, X_list, Y_list)

print('\n')
#print("Max average f1 for Adaline:  %.3f" % max(f1_list))
#print("Max average accuracy for Adaline:  %.3f" % max(acc_list))
#index = f1_list.index(max(f1_list))
#best_parameters = idx_list[index]
print("Adaline best parameters: {n_iter: 10, eta: 0.1}")
print("Adaline Validation Statistics: ")
Adaline = AdalineGD(n_iter=10, eta=0.1)
Adaline.fit(x_validate, y_validate)
y_pred = Adaline.predict(x_test)
acc_score = accuracy_score(y_test, y_pred)
f1Score = f1_score(y_test, y_pred, average='macro')
print('Accuracy: %.3f' % (acc_score))
print('F1: %.3f' % (f1Score))






