import pandas as pd

def preprocess_data():
    
    train_data = pd.read_csv('C:\\Users\\adi20\\Desktop\\XAI-Loan\\data\\raw\\loan-train.csv')
    test_data = pd.read_csv('C:\\Users\\adi20\\Desktop\\XAI-Loan\\data\\raw\\loan-test.csv')

    
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    
    categorical_variables = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

    
    train_data = pd.get_dummies(train_data, columns=categorical_variables)
    test_data = pd.get_dummies(test_data, columns=categorical_variables)

    
    test_data = test_data.reindex(columns = train_data.columns, fill_value=0)


    train_data.to_csv('C:\\Users\\adi20\\Desktop\\XAI-Loan\\data\\processed\\loan-train-processed.csv', index=False)
    test_data.to_csv('C:\\Users\\adi20\\Desktop\\XAI-Loan\\data\\processed\\loan-test-processed.csv', index=False)

if __name__ == "__main__":
    preprocess_data()