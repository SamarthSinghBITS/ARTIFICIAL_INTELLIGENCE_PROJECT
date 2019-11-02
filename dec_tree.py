import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz

# Load data file
data = pd.read_csv('bank.csv')
data.head()
# print(data)

# print(data.describe())

# Make a copy for parsing
bank_data = data.copy()
# print(bank_data)

# Combine similar jobs into categiroes
bank_data['job'] = bank_data['job'].replace(['management', 'admin.'], 'white-collar')
bank_data['job'] = bank_data['job'].replace(['services', 'housemaid'], 'pink-collar')
bank_data['job'] = bank_data['job'].replace(['retired', 'student', 'unemployed', 'unknown'], 'other')
# print(bank_data)
# New value counts
# print(bank_data.job.value_counts())

# Combine 'unknown' and 'other' as 'other' isn't really match with either 'success' or 'failure'
bank_data.drop('poutcome', axis=1, inplace=True)

# Drop 'contact', as every participant has been contacted.
bank_data.drop('contact', axis=1, inplace=True)
# print(bank_data)

# values for "default" : yes/no
# bank_data["default"]
bank_data['default_cat'] = bank_data['default'].map({'yes': 1, 'no': 0})
bank_data.drop('default', axis=1, inplace=True)

# values for "housing" : yes/no
bank_data["housing_cat"] = bank_data['housing'].map({'yes': 1, 'no': 0})
bank_data.drop('housing', axis=1, inplace=True)

# values for "loan" : yes/no
bank_data["loan_cat"] = bank_data['loan'].map({'yes': 1, 'no': 0})
bank_data.drop('loan', axis=1, inplace=True)

# day  : last contact day of the month
# month: last contact month of year
# Drop 'month' and 'day' as they don't have any intrinsic meaning
bank_data.drop('month', axis=1, inplace=True)
bank_data.drop('day', axis=1, inplace=True)

# values for "deposit" : yes/no
bank_data["deposit_cat"] = bank_data['deposit'].map({'yes': 1, 'no': 0})
bank_data.drop('deposit', axis=1, inplace=True)

# Map padys=-1 into a large value (10000 is used) to indicate that it is so far in the past that it has no effect
bank_data.loc[bank_data['pdays'] == -1, 'pdays'] = 10000

# Create a new column: recent_pdays
bank_data['recent_pdays'] = np.where(bank_data['pdays'], 1/bank_data.pdays, 1/bank_data.pdays)

# Drop 'pdays'
bank_data.drop('pdays', axis=1, inplace=True)

# Drop 'previous'
bank_data.drop('previous', axis=1, inplace=True)

for col in bank_data.columns:
    print(col)
# Convert categorical variables to dummies
bank_with_dummies = pd.get_dummies(data=bank_data, columns=['job', 'marital', 'education'],
                                   prefix=['job', 'marital', 'education'])
print(bank_with_dummies)

for col in bank_with_dummies.columns:
    print(col)
# make a copy
bankcl = bank_with_dummies

# corr = bankcl.corr()
# print(corr)

# Train-Test split: 20% test data
data_drop_deposite = bankcl.drop('deposit_cat', 1)
label = bankcl.deposit_cat
data_train, data_test, label_train, label_test = train_test_split(data_drop_deposite, label,
                                                                  test_size=0.2, random_state=50)
# print(data_train)
# print(data_test)
# print(label_train)
# print(label_test)


def train_using_gini(x_train, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=1, max_depth=2)
    # Performing training
    clf_gini.fit(x_train, y_train)
    return clf_gini


gini = train_using_gini(data_train, label_train)
# print(gini)


def train_using_entropy(x_train, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=2)
    # Performing training
    clf_entropy.fit(x_train, y_train)
    return clf_entropy


Info_gain = train_using_entropy(data_train, label_train)
# print(Info_gain)


# Function to make predictions
def prediction(x_test, clf_object):
    # Predicton
    y_pred = clf_object.predict(x_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : ",
          classification_report(y_test, y_pred))


y_pred_gini = prediction(data_test, gini)
cal_accuracy(label_test, y_pred_gini)


y_pred_entropy = prediction(data_test, Info_gain)
cal_accuracy(label_test, y_pred_entropy)

export_graphviz(gini, out_file='tree.dot')
