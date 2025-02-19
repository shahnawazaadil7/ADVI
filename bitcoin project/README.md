# Bitcoin Blockchain De-Anonymizer

This project aims to analyze Bitcoin transaction trends using machine learning techniques to classify blockchain data. The analysis focuses on transaction fees, input values, and transaction volumes while evaluating model performance for fee-level predictions.

## Table of Contents

- [Introduction](#introduction)
- [Data Analysis](#data-analysis)
  - [Transaction Metrics Over Time](#transaction-metrics-over-time)
- [Model Performance Evaluation](#model-performance-evaluation)
- [Methodology](#methodology)
- [Results and Discussion](#results-and-discussion)
- [Conclusion](#conclusion)
- [Setup](#setup)
- [Usage](#usage)

## Introduction

Bitcoin transactions are inherently pseudonymous, making it challenging to trace the origins and destinations of funds. This study analyzes Bitcoin transaction trends using machine learning techniques to classify blockchain data. The analysis focuses on transaction fees, input values, and transaction volumes while evaluating model performance for fee-level predictions.

## Data Analysis

### Transaction Metrics Over Time

#### Fees Over Time
- **Observation**: The fee trend shows significant volatility with sharp spikes, particularly between 2014 and 2018.
- **Implication**: Transaction fees have been highly dynamic, possibly due to network congestion and Bitcoin's increasing adoption.

#### Input Value Over Time
- **Findings**: Input values display a growing trend with exponential surges, particularly around 2016 and 2021.
- **Implication**: This indicates an increase in Bitcoin transaction sizes over time.

#### Transactions Over Time
- **Observation**: The number of transactions has increased consistently over the years, peaking in 2022.
- **Implication**: Bitcoin network activity has steadily grown, reinforcing its increasing adoption.

## Model Performance Evaluation

### Accuracy Comparison
The model performances for transaction fee classification are as follows:
- Logistic Regression Accuracy: 75%
- Random Forest Accuracy: 100%
- XGBoost Accuracy: 100%

### Classification Report Summary
**Logistic Regression**
- Class 0: Precision: 94%, Recall: 53%
- Class 1: Precision: 68%, Recall: 96%

**Random Forest & XGBoost**
- Accuracy: 100%
- Concern: The perfect accuracy indicates potential overfitting in these models.

## Methodology

### Data Preprocessing
- Missing values were replaced with zeros.
- `Transaction_Date` feature was created using year, month, and day.
- New features derived:
  - `Transaction_Fee_Ratio = fee / input_value`
  - `Value_Difference = output_value - input_value`

### Model Training
- **Features**: Transactions, input value, output value, fee, Transaction_Fee_Ratio, Value_Difference.
- **Target**: Binary classification (fee higher/lower than median value).
- **Algorithms Used**:
  - Logistic Regression
  - Random Forest
  - XGBoost
- **Evaluation Metrics**: Accuracy, Classification Report.

## Results and Discussion

### Model Performance Insights
- The high accuracy of Random Forest and XGBoost suggests possible overfitting.
- Logistic Regression provides more realistic classification results.

### Transaction Trends
- Input values have increased significantly over time.
- Transaction counts have increased, confirming Bitcoin's growing adoption.

## Conclusion

- Machine learning models require further validation to prevent overfitting.
- Bitcoin transaction trends suggest an evolving landscape, necessitating deeper analysis.
- Feature engineering and data consistency improvements are necessary for better predictive accuracy.

## Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/shahnawazaadil7/bitcoin-project.git
   cd bitcoin-project
   ```

5. The application will perform the analysis and generate reports in the `output` directory.

## Visualizing Results

To visualize the results using Streamlit, run the following command: