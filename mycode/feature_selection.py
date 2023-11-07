import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


train_data = pd.read_csv('data/processed/loan-train-processed.csv')
test_data = pd.read_csv('data/processed/loan-test-processed.csv')


train_data['Dependents'].replace('3+', 3, inplace=True)
test_data['Dependents'].replace('3+', 3, inplace=True)

train_data['Dependents'] = pd.to_numeric(train_data['Dependents'])
test_data['Dependents'] = pd.to_numeric(test_data['Dependents'])



X_train = train_data.drop(['Loan_Status', 'Loan_ID'], axis=1)
y_train = train_data['Loan_Status']
X_test = test_data.drop(['Loan_Status', 'Loan_ID'], axis=1)
y_test = test_data['Loan_Status']


X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)


for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))

