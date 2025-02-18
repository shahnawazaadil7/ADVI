import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# Load dataset (replace with actual path)
df = pd.read_csv('/Users/shahnawazaadil/Desktop/Github/ADVI/bitcoin project/bitcoin_transactions_confirmed.csv')

# Preprocessing
df.fillna(0, inplace=True)

# Feature engineering for Bitcoin dataset
df['Transaction_Date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str))
df['Transaction_Fee_Ratio'] = df['Fee'] / df['Input Value']
df['Value_Difference'] = df['Output Value'] - df['Input Value']

# Define features and target
X = df[['Number of Transactions', 'Input Value', 'Output Value', 'Fee', 'Transaction_Fee_Ratio', 'Value_Difference']]
y = (df['Fee'] > df['Fee'].median()).astype(int)  # Binary classification based on median fee

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
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
        "Report": classification_report(y_test, y_pred)
    }

# Streamlit GUI
st.title('Bitcoin Blockchain De-Anonymizer')
for model, metrics in results.items():
    st.subheader(f'Model: {model}')
    st.write(f"Accuracy: {metrics['Accuracy']:.2f}")
    st.text(metrics['Report'])

st.bar_chart(pd.DataFrame({m: [r['Accuracy']] for m, r in results.items()}))

st.write("Dataset Insights")
st.line_chart(df[['Transaction_Date', 'Number of Transactions']].set_index('Transaction_Date'))
st.line_chart(df[['Transaction_Date', 'Input Value']].set_index('Transaction_Date'))
st.line_chart(df[['Transaction_Date', 'Fee']].set_index('Transaction_Date'))