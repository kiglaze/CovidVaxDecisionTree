# https://www.geeksforgeeks.org/decision-tree-implementation-python/

# Run this program on your local python
# interpreter, provided you have installed
# the required libraries.

# how to read a .pkl file: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html

# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Function importing Dataset
def import_data():
	imported_data = pd.read_pickle("./all_agg_df.pkl")

	# Prints all available column names for use in imported data.
	for col in imported_data.columns:
		print(col)
	subset_data = imported_data[["State", "Sentiment", "Day", "Dist_Per_100K", "Admin_Per_100K", "new_case_per_100K", "cum_tot_cases_per_100K"]]

	# Dataset obseravtions
	# int(re.findall('^\\d+|$', subset_data["Day"])[0])
	# subset_data['Day'] = subset_data['Day'].transform(transform_days_data)
	subset_data = subset_data.loc[subset_data["State"] != "Total"]
	subset_data = subset_data.loc[subset_data["State"] != "District of Columbia"]
	subset_data_time_cross_section = subset_data.loc[622 == subset_data['Day'].transform(transform_days_data)]
	standardized_y = StandardScaler().fit_transform(subset_data_time_cross_section[['Admin_Per_100K']])
	standardized_y_arr = []
	y_category_arr = []
	index = 0
	for val in standardized_y:
		standardized_val = standardized_y[index][0]
		standardized_y_arr.append(standardized_val)
		category = None
		if(standardized_val >= 0):
			category = 'H'
		else:
			category = 'L'
		y_category_arr.append(category)
		index = index + 1

	subset_data_time_cross_section.insert(0, "standardized_y", standardized_y_arr, True)
	subset_data_time_cross_section.insert(0, "y_category", y_category_arr, True)

	# subset_data_time_cross_section.shape[0] gives # states represented
	result = subset_data_time_cross_section[["y_category", "Sentiment", "cum_tot_cases_per_100K"]]
	return result


def transform_days_data(days_string):
	return days_string.days


# Function to split the dataset
def splitdataset(balance_data):
	# Separating the target variable
	# X = balance_data.values[:, 0:2]
	#X = balance_data[["Sentiment", "cum_tot_cases_per_100K"]]
	X = balance_data.values[:, 1:3]
	# Y = balance_data.values[:, 4]
	Y = balance_data.values[:, 0]

	# Splitting the dataset into train and test
	X_train, X_test, y_train, y_test = train_test_split(
		X, Y, test_size=0.3, random_state=100)

	return X, Y, X_train, X_test, y_train, y_test


# Function to perform training with giniIndex.
def train_using_gini(X_train, Y_train):
	# Creating the classifier object
	#clf = DecisionTreeClassifier(max_depth=4)
	clf = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)

	# Performing training
	clf = clf.fit(X_train, Y_train)
	tree.plot_tree(clf)
	plt.show()
	return clf


# Driver code
def main():
	# Building Phase
	data = import_data()
	X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

	clf = train_using_gini(X_train, y_train)
	print("end")


# Calling main function
if __name__=="__main__":
	main()

# Suspected issue: Y has a ton of different values, rather than being categorized.
# Number of values in value array is large because of the number of demonstrated values for Y.
# All the 1's you see actually add up to the number of samples.
# Possible fix: Need to make Y have fewer possible values.
# Perhaps standardization and splitting into thirds afterwards (0-.33, .33-.66, .66-1).
