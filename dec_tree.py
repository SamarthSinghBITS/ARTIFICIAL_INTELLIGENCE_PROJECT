import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import tkinter as tk

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

# for col in bank_with_dummies.columns:
#     print(col)
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
# for col in data_test.columns:
#     print(col)


# Function to make predictions
def prediction(x_test, clf_object):
    # Prediction
    y_pred = clf_object.predict(x_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))

    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)

    print(classification_report(y_test, y_pred))


# Cart Algorithm
def train_using_gini(x_train, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=1, max_depth=2)
    # Performing training
    clf_gini.fit(x_train, y_train)
    return clf_gini


# Training Using Cart Algorithm
gini = train_using_gini(data_train, label_train)
# print(gini)


# ID3 Algorithm
def train_using_entropy(x_train, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=2)
    # Performing training
    clf_entropy.fit(x_train, y_train)
    return clf_entropy


# Training Using ID3 Algorithm
Info_gain = train_using_entropy(data_train, label_train)
# print(Info_gain)

# Predict on data set which model has not seen before
y_pred_gini = prediction(data_test, gini)

# Calculate Precision, Recall, F-measure and Accuracy of the model
cal_accuracy(label_test, y_pred_gini)

# Predict on data set which model has not seen before
y_pred_entropy = prediction(data_test, Info_gain)

# Calculate Precision, Recall, F-measure and Accuracy of the model
cal_accuracy(label_test, y_pred_entropy)

# Drawing Graphs
export_graphviz(gini, out_file='gini.dot')
export_graphviz(Info_gain, out_file='Info_gain.dot')


# KNN Algorithm
def train_using_knn(x_train, y_train):
    knn1 = KNeighborsClassifier(n_neighbors=49)
    # Performing training
    knn1.fit(x_train, y_train)
    return knn1


# Training Using KNN Algorithm
knn = train_using_knn(data_train, label_train)

# Predict on data set which model has not seen before
predict_knn = prediction(data_test, knn)

# Calculate Precision, Recall, F-measure and Accuracy of the model
cal_accuracy(label_test, predict_knn)


# MLP Algorithm
def train_using_mlp(x_train, y_train):
    mlp1 = MLPClassifier(solver='adam', activation='logistic', alpha=1e-5, hidden_layer_sizes=(7),
                         random_state=1, epsilon=1e-9)
    # Performing training
    mlp1.fit(x_train, y_train)
    return mlp1


# Training Using MLP Algorithm
mlp = train_using_mlp(data_train, label_train)

# Predict on data set which model has not seen before
predict_mlp = mlp.predict(data_test)

# Calculate Precision, Recall, F-measure and Accuracy of the model
cal_accuracy(label_test, predict_mlp)


# SVM Algorithm
def train_using_svm(x_train, y_train):
    svclf1 = svm.LinearSVC(penalty='l2', dual=False, C=2.5, random_state=1)
    # Performing training
    svclf1.fit(x_train, y_train)
    return svclf1


# Training Using Linear SVC Algorithm
svclf = train_using_svm(data_train, label_train)

# Predict on data set which model has not seen before
predict_svm = svclf.predict(data_test)

# Calculate Precision, Recall, F-measure and Accuracy of the model
cal_accuracy(label_test, predict_svm)

main_input = []


def decision():
    try:
        main_input.append(int(age.get("1.0", 'end-1c')))
        main_input.append(job_var.get())
        main_input.append(mar_var.get())
        main_input.append(edu_var.get())
        main_input.append(int(balance.get("1.0", 'end-1c')))
        main_input.append(int(duration.get("1.0", 'end-1c')))
        main_input.append(int(campaign.get("1.0", 'end-1c')))
        main_input.append(option_var_default.get())
        main_input.append(option_var_housing.get())
        main_input.append(option_var_loan.get())
        main_input.append(int(recent_pdays.get("1.0", 'end-1c')))
        if main_input[7] == 'Yes':
            main_input[7] = 1
        else:
            main_input[7] = 0

        if main_input[8] == 'Yes':
            main_input[8] = 1
        else:
            main_input[8] = 0

        if main_input[9] == 'Yes':
            main_input[9] = 1
        else:
            main_input[9] = 0

        if main_input[0] < 0 or main_input[5] < 0 or main_input[6] < 0:
            raise ValueError

        print(main_input[0], main_input[1])
        bank = pd.DataFrame([main_input])
        bank.columns = ['age', 'job', 'marital', 'education', 'balance', 'duration', 'campaign', 'default_cat',
                        'housing_cat', 'loan_cat', 'recent_pdays']
        print(bank)
        bank["job_blue-collar"] = bank['job'].map({'Blue Collar': 1, 'Entrepreneur': 0, 'Self Employed': 0,
                                                   'Pink Collar': 0, 'Technician': 0, 'White Collar': 0, 'Others': 0})
        bank["job_entrepreneur"] = bank['job'].map({'Blue Collar': 0, 'Entrepreneur': 1, 'Self Employed': 0,
                                                    'Pink Collar': 0, 'Technician': 0, 'White Collar': 0, 'Others': 0})
        bank["job_other"] = bank['job'].map({'Blue Collar': 0, 'Entrepreneur': 0, 'Self Employed': 0,
                                             'Pink Collar': 0, 'Technician': 0, 'White Collar': 0, 'Others': 1})
        bank["job_pink-collar"] = bank['job'].map({'Blue Collar': 0, 'Entrepreneur': 0, 'Self Employed': 0,
                                                   'Pink Collar': 1, 'Technician': 0, 'White Collar': 0, 'Others': 0})
        bank["job_self-employed"] = bank['job'].map({'Blue Collar': 0, 'Entrepreneur': 0, 'Self Employed': 1,
                                                     'Pink Collar': 0, 'Technician': 0, 'White Collar': 0, 'Others': 0})
        bank["job_technician"] = bank['job'].map({'Blue Collar': 0, 'Entrepreneur': 0, 'Self Employed': 0,
                                                  'Pink Collar': 0, 'Technician': 1, 'White Collar': 0, 'Others': 0})
        bank["job_white-collar"] = bank['job'].map({'Blue Collar': 0, 'Entrepreneur': 0, 'Self Employed': 0,
                                                    'Pink Collar': 0, 'Technician': 0, 'White Collar': 1, 'Others': 0})
        bank.drop('job', axis=1, inplace=True)
        bank["marital_divorced"] = bank['marital'].map({'Divorced': 1, 'Married': 0, 'Single': 0})
        bank["marital_married"] = bank['marital'].map({'Divorced': 0, 'Married': 1, 'Single': 0})
        bank["marital_single"] = bank['marital'].map({'Divorced': 0, 'Married': 0, 'Single': 1})
        bank.drop('marital', axis=1, inplace=True)
        bank["education_primary"] = bank['education'].map({'Primary': 1, 'Secondary': 0, 'Tertiary': 0, 'Unknown': 0})
        bank["education_secondary"] = bank['education'].map({'Primary': 0, 'Secondary': 1, 'Tertiary': 0, 'Unknown': 0})
        bank["education_tertiary"] = bank['education'].map({'Primary': 0, 'Secondary': 0, 'Tertiary': 1, 'Unknown': 0})
        bank["education_unknown"] = bank['education'].map({'Primary': 0, 'Secondary': 0, 'Tertiary': 0, 'Unknown': 1})
        bank.drop('education', axis=1, inplace=True)
        print(bank)
        for col in bank.columns:
            print(col)
        predict_knn_f = prediction(bank, knn)
        predict_cart_f = prediction(bank, gini)
        predict_id_f = prediction(bank, Info_gain)
        predict_mlp_f = prediction(bank, mlp)
        predict_svm_f = prediction(bank, svclf)
        sum_f = predict_knn_f + predict_cart_f + predict_id_f + predict_mlp_f + predict_svm_f
        if sum_f < 3:
            predict = 0
        else:
            predict = 1
        print(predict)

    except ValueError:
        print("Invalid input")


master = tk.Tk()
# master.geometry("1366x768")
master.title("Enter the client's details:")
tk.Label(master, text='Age').grid(row=0)

Jobs = ['Blue Collar', 'Entrepreneur', 'Pink Collar', 'Self Employed', 'Technician', 'White Collar', 'Others']
job_var = tk.StringVar(master)
job_var.set(Jobs[0])

Marital = ['Divorced', 'Married', 'Single']
mar_var = tk.StringVar(master)
mar_var.set(Marital[0])

Education = ['Primary', 'Secondary', 'Tertiary', 'Unknown']
edu_var = tk.StringVar(master)
edu_var.set(Education[0])

Yes_No_default = ['Yes', 'No']
option_var_default = tk.StringVar(master)
option_var_default.set(Yes_No_default[0])

Yes_No_housing = ['Yes', 'No']
option_var_housing = tk.StringVar(master)
option_var_housing.set(Yes_No_housing[0])

Yes_No_loan = ['Yes', 'No']
option_var_loan = tk.StringVar(master)
option_var_loan.set(Yes_No_loan[0])

tk.Label(master, text='Job').grid(row=1)
tk.Label(master, text='Marital Status').grid(row=2)
tk.Label(master, text='Education').grid(row=3)
tk.Label(master, text='Balance').grid(row=4)
tk.Label(master, text='Duration').grid(row=5)
tk.Label(master, text='Campaign').grid(row=6)
tk.Label(master, text='Default').grid(row=7)
tk.Label(master, text='Housing').grid(row=8)
tk.Label(master, text='Loan').grid(row=9)
tk.Label(master, text='Recent_Pdays').grid(row=10)

age = tk.Text(master, height=1, width=10)

# job = tk.Entry(master)
job = tk.OptionMenu(master, job_var, *Jobs)

# marital = tk.Entry(master)
marital = tk.OptionMenu(master, mar_var, *Marital)

# education = tk.Entry(master)
education = tk.OptionMenu(master, edu_var, *Education)

balance = tk.Text(master, height=1, width=10)
duration = tk.Text(master, height=1, width=10)
campaign = tk.Text(master, height=1, width=10)

# default = tk.Entry(master)
default = tk.OptionMenu(master, option_var_default, *Yes_No_default)

# housing = tk.Entry(master)
housing = tk.OptionMenu(master, option_var_housing, *Yes_No_housing)

# loan = tk.Entry(master)
loan = tk.OptionMenu(master, option_var_loan, *Yes_No_loan)

recent_pdays = tk.Text(master, height=1, width=10)

age.grid(row=0, column=1)
job.grid(row=1, column=1)
marital.grid(row=2, column=1)
education.grid(row=3, column=1)
balance.grid(row=4, column=1)
duration.grid(row=5, column=1)
campaign.grid(row=6, column=1)
default.grid(row=7, column=1)
housing.grid(row=8, column=1)
loan.grid(row=9, column=1)
recent_pdays.grid(row=10, column=1)
submit = tk.Button(master, text='Submit', width=25, command=decision)
submit.grid(row=11, column=1)

tk.mainloop()
