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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re

# Function importing Dataset
def import_data():
	imported_data = pd.read_pickle("./all_agg_df.pkl")

	# Prints all available column names for use in imported data.
	for col in imported_data.columns:
		print(col)
	subset_data = imported_data[["State", "Sentiment", "Day", "Dist_Per_100K", "Admin_Per_100K", "new_case_per_100K", "cum_tot_cases_per_100K"]]

	# Printing the dataset shape
	print("Dataset Length: ", len(subset_data))
	print("Dataset Shape: ", subset_data.shape)

	# Printing the dataset obseravtions
	print("Dataset: ", subset_data.head())
	# re.fullmatch('\\d+', '13 days 00:00:00')
	# int(re.findall('^\\d+|$', '13 days 00:00:00')[0])
	# int(re.findall('^\\d+|$', subset_data["Day"])[0])
	subset_data['Day_int'] = subset_data['Day'].transform(transform_days_data)
	subset_data = subset_data.loc[subset_data["State"] != "Total"]
	subset_data_time_cross_section = subset_data.loc[subset_data["Day_int"] == 622]
	# subset_data_time_cross_section.shape[0] gives # states represented
	return subset_data


def transform_days_data(days_string):
	return days_string.days


# Driver code
def main():
	# Building Phase
	data = import_data()
	print(data.head())

	
# Calling main function
if __name__=="__main__":
	main()
