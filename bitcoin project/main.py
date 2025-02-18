import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/shahnawazaadil/Desktop/Github/ADVI/bitcoin project/bitcoin_transactions_confirmed.csv', delimiter='\t')

df.fillna(0, inplace=True)

df['Transaction_Date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str))
df['Transaction_Fee_Ratio'] = df['fee'] / df['input_value']
df['Value_Difference'] = df['output_value'] - df['input_value']

X = df[['transactions', 'input_value', 'output_value', 'fee', 'Transaction_Fee_Ratio', 'Value_Difference']]
y = (df['fee'] > df['fee'].median()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Report": classification_report(y_test, y_pred, output_dict=True)
    }

output_dir = '/Users/shahnawazaadil/Desktop/Github/ADVI/bitcoin project/output'
os.makedirs(output_dir, exist_ok=True)

for name, model in models.items():
    with open(os.path.join(output_dir, f'{name}_report.txt'), 'w') as f:
        f.write(f"Accuracy: {results[name]['Accuracy']:.2f}\n")
        f.write(classification_report(y_test, model.predict(X_test)))

plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=[metrics['Accuracy'] for metrics in results.values()])
plt.title('Model Accuracy')
plt.savefig(os.path.join(output_dir, 'model_accuracy.png'))

plt.figure(figsize=(10, 6))
df.set_index('Transaction_Date')['transactions'].plot()
plt.title('Transactions Over Time')
plt.savefig(os.path.join(output_dir, 'transactions_over_time.png'))

plt.figure(figsize=(10, 6))
df.set_index('Transaction_Date')['input_value'].plot()
plt.title('Input Value Over Time')
plt.savefig(os.path.join(output_dir, 'input_value_over_time.png'))

plt.figure(figsize=(10, 6))
df.set_index('Transaction_Date')['fee'].plot()
plt.title('Fees Over Time')
plt.savefig(os.path.join(output_dir, 'fees_over_time.png'))

st.title('Bitcoin Blockchain De-Anonymizer')
for name, metrics in results.items():
    st.subheader(f'Model: {name}')
    st.write(f"Accuracy: {metrics['Accuracy']:.2f}")
    st.text(classification_report(y_test, models[name].predict(X_test)))

st.bar_chart(pd.DataFrame({m: [r['Accuracy']] for m, r in results.items()}))

st.write("Dataset Insights")
st.line_chart(df[['Transaction_Date', 'transactions']].set_index('Transaction_Date'))
st.line_chart(df[['Transaction_Date', 'input_value']].set_index('Transaction_Date'))
st.line_chart(df[['Transaction_Date', 'fee']].set_index('Transaction_Date'))