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

# Series_Complete_Pop_Pct, ['Sentiment'],['Age'],['Education'],['Income']

# Function importing Dataset
def import_data():
	imported_data = pd.read_pickle("./all_agg_df.pkl")

	# Prints all available column names for use in imported data.
	for col in imported_data.columns:
		print(col)
	subset_data = imported_data[["State", "Sentiment", "Families-Median income (dollars)", "Total-AGE BY EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Bachelor's degree or higher", "Total-AGE BY EDUCATIONAL ATTAINMENT!!Population 18 to 24 years!!Bachelor's degree or higher", "SEX AND AGE!!Total population", "Series_Complete_Pop_Pct", "Day", "Date", "Dist_Per_100K", "Admin_Per_100K", "new_case_per_100K", "cum_tot_cases_per_100K"]]

	# Dataset obseravtions
	# int(re.findall('^\\d+|$', subset_data["Day"])[0])
	# subset_data['Day'] = subset_data['Day'].transform(transform_days_data)
	subset_data = subset_data.loc[subset_data["State"] != "Total"]
	subset_data = subset_data.loc[subset_data["State"] != "District of Columbia"]
	bachelor_pct = (subset_data["Total-AGE BY EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Bachelor's degree or higher"] + subset_data["Total-AGE BY EDUCATIONAL ATTAINMENT!!Population 18 to 24 years!!Bachelor's degree or higher"]) / subset_data["SEX AND AGE!!Total population"]
	subset_data.insert(0, "bachelor_pct", bachelor_pct, True)
	subset_data_time_cross_section = subset_data.loc[622 == subset_data['Day'].transform(transform_days_data)] # Jan 2020
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
	result = subset_data_time_cross_section[["y_category", "bachelor_pct", "Families-Median income (dollars)", "Sentiment"]]
	# 	result = subset_data_time_cross_section[["y_category", "bachelor_pct", "Families-Median income (dollars)", "Sentiment"]]
	return result


def transform_days_data(days_string):
	return days_string.days


# Function to split the dataset
def splitdataset(balance_data, featuresArray):
	# Separating the target variable
	# X = balance_data.values[:, 0:2]
	#X = balance_data[["Sentiment", "cum_tot_cases_per_100K"]]
	X = balance_data.values[:, 1:(len(featuresArray)+1)]
	# Y = balance_data.values[:, 4]
	Y = balance_data.values[:, 0]

	# Splitting the dataset into train and test
	X_train, X_test, y_train, y_test = train_test_split(
		X, Y, test_size=0.3, random_state=100)

	return X, Y, X_train, X_test, y_train, y_test


# Function to perform training with giniIndex.
def train_using_gini(X_train, Y_train, featuresArray):
	# Creating the classifier object
	#clf = DecisionTreeClassifier(max_depth=4)
	clf = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=6, min_samples_leaf=5)

	# Performing training
	clf = clf.fit(X_train, Y_train)

	plt.figure(figsize=(10,10), dpi=100)
	tree.plot_tree(clf, fontsize=10, feature_names=featuresArray, class_names=["H", "L"])
	tree.export_graphviz(clf, out_file="tree.dot", feature_names=featuresArray, class_names=["H", "L"], filled=True)
	# , feature_names=["Sentiment", "cum_tot_cases_per_100K", "bachelor_pct"]
	# feature_names=["Sentiment", "% Bachelor's", "Families Median Income"]
	plt.show()
	return clf


# Function to make predictions
def prediction(X_test, clf_object):
	# Predicton on test with giniIndex
	y_pred = clf_object.predict(X_test)
	print("Predicted values:")
	print(y_pred)
	return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
	print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))

	print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)

	print("Report : ", classification_report(y_test, y_pred))


# /Users/irisglaze/Documents/CSC522/decision_tree_implementation/venv/bin/python decision_tree_implementation.py
# run to get decision tree image: dot -Tpng -Gdpi=300 tree.dot -o tree.png
# must install graphviz first to run in Mac

# Driver code
def main():
	# Building Phase
	data = import_data()
	featuresArray = ["% Bachelor's", "Families Median Income"];
	X, Y, X_train, X_test, y_train, y_test = splitdataset(data, featuresArray)

	clf = train_using_gini(X_train, y_train, featuresArray)
	# Operational Phase
	print("Results Using Gini Index:")

	# Prediction using gini
	y_pred_gini = prediction(X_test, clf)
	cal_accuracy(y_test, y_pred_gini)

	print("end")


# Calling main function
if __name__=="__main__":
	main()

# Suspected issue: Y has a ton of different values, rather than being categorized.
# Number of values in value array is large because of the number of demonstrated values for Y.
# All the 1's you see actually add up to the number of samples.
# Possible fix: Need to make Y have fewer possible values.
# Perhaps standardization and splitting into thirds afterwards (0-.33, .33-.66, .66-1).
