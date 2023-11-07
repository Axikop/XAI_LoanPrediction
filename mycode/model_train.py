from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
import pandas as pd
import os


from data_preprocess import preprocess_data
from feature_selection import select_features


preprocess_data()

train_data = pd.read_csv('data/processed/loan-train-processed.csv')
test_data = pd.read_csv('data/processed/loan-test-processed.csv')


train_data['Dependents'].replace('3+', 3, inplace=True)
test_data['Dependents'].replace('3+', 3, inplace=True)

train_data['Dependents'] = pd.to_numeric(train_data['Dependents'])
test_data['Dependents'] = pd.to_numeric(test_data['Dependents'])

X_train = train_data.drop(['Loan_Status', 'Loan_ID'], axis=1)
y_train = train_data['Loan_Status']
X_test = test_data.drop(['Loan_Status', 'Loan_ID'], axis=1)


X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)


X_train, X_val, y_train, y_val = train_test_split(X_train_fs, y_train, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_leaf=1, min_samples_split=2)


model.fit(X_train, y_train)


y_pred = model.predict(X_val)
print(f"Validation accuracy: {accuracy_score(y_val, y_pred)}")


if not os.path.exists('models'):
    os.makedirs('models')


dump(model, 'models/decision_tree.joblib')
