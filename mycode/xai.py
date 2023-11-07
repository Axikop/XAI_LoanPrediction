import shap
import pandas as pd
from joblib import load

model = load('models/decision_tree.joblib')


test_data = pd.read_csv('data/processed/loan-test-processed.csv')


test_data['Dependents'].replace('3+', 3, inplace=True)


test_data['Dependents'] = pd.to_numeric(test_data['Dependents'])


X_test = test_data.drop(['Loan_Status', 'Loan_ID'], axis=1)

xplain = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

for i in range(len(X_test)):
    print(f"Prediction: {model.predict(X_test.iloc[i, :].values.reshape(1, -1))}")
    print(f"SHAP values: {shap_values[0][i]}")
    shap.force_plot(xplain.expected_value[0], shap_values[0][i], X_test.iloc[i,:])

shap.summary_plot(shap_values, X_test)    
