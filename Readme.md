# Telekom Customer Churn Prediction

Author: **Mokhichekhra Makhmudova**

***

## Project Overview

This is a small end‑to‑end machine learning project:

- You explore and clean Telekom customer data in a Jupyter Notebook. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- You train and save a machine learning model and a scaler. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- You build a simple Streamlit web app to make churn predictions for new customers. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/03646449-f97f-43ee-acab-095dbd7680ab/app.py)

The goal is to give a clear, beginner‑friendly example of a churn prediction workflow that can later be extended with more features.

***

## Dataset

The dataset is stored in `customer_churn_data.csv` and contains 1,000 rows and 10 columns. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/93ca33bc-4cf0-4af0-9f8e-f8dee9ca832e/customer_churn_data.csv)

Main columns:

- **CustomerID**: Unique customer identifier. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/93ca33bc-4cf0-4af0-9f8e-f8dee9ca832e/customer_churn_data.csv)
- **Age**: Customer age (integer). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/93ca33bc-4cf0-4af0-9f8e-f8dee9ca832e/customer_churn_data.csv)
- **Gender**: Male or Female. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- **Tenure**: Number of months the customer has stayed with the company. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/93ca33bc-4cf0-4af0-9f8e-f8dee9ca832e/customer_churn_data.csv)
- **MonthlyCharges**: Monthly bill amount. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- **ContractType**: Month‑to‑Month, One‑Year, or Two‑Year. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/93ca33bc-4cf0-4af0-9f8e-f8dee9ca832e/customer_churn_data.csv)
- **InternetService**: None, DSL, or Fiber Optic. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/93ca33bc-4cf0-4af0-9f8e-f8dee9ca832e/customer_churn_data.csv)
- **TotalCharges**: Total amount charged during the contract. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/93ca33bc-4cf0-4af0-9f8e-f8dee9ca832e/customer_churn_data.csv)
- **TechSupport**: Yes or No. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/93ca33bc-4cf0-4af0-9f8e-f8dee9ca832e/customer_churn_data.csv)
- **Churn**: Target variable, Yes if the customer churned, otherwise No. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)

In this version of the model, only a subset of features is used:

- **Age**  
- **Gender** (encoded as 1 = Female, 0 = Male) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- **Tenure**  
- **MonthlyCharges** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)

***

## Exploratory Data Analysis (EDA)

EDA is done in `notebook.ipynb`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)

Key steps:

- **Data inspection**: `df.head()`, `df.info()`, and `df.describe()` to understand basic structure and summary statistics. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- **Missing values**: `InternetService` missing values are filled before modeling. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- **Duplicates**: Checked with `df.duplicated().sum()`; none are kept. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- **Correlation analysis**: A correlation matrix is created for numeric columns (CustomerID, Age, Tenure, MonthlyCharges, TotalCharges). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- **Churn distribution**: Pie chart of churn vs non‑churn customers. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- **Group statistics**:  
  - Mean MonthlyCharges by Churn. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
  - Mean MonthlyCharges by Churn and Gender. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
  - Mean Tenure by Churn. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
  - Mean Age by Churn. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- **Visualizations**:  
  - Histogram of Tenure. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
  - Histogram of MonthlyCharges. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
  - Bar chart of average MonthlyCharges by ContractType. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)

These steps help to understand how churn is related to age, monthly charges, and tenure.

***

## Modeling

All modeling code is in `notebook.ipynb`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)

### Feature Engineering

- Input features **X**: `Age`, `Gender`, `Tenure`, `MonthlyCharges`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- Target **y**: `Churn`, encoded as 1 for Yes and 0 for No. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- `Gender` is converted to numeric: 1 = Female, 0 = Male. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)

### Train–Test Split

- Train–test split with 80% training and 20% test data using `train_test_split` from scikit‑learn. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)

### Scaling

- Numerical features are standardized using `StandardScaler`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- The scaler is fitted on the training data and saved using `joblib` to `scaler.pkl`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)

### Model

- The main classifier is **Logistic Regression** (`sklearn.linear_model.LogisticRegression`). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- The model is trained on the scaled training data. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- Model performance is evaluated using accuracy on the test set. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- The trained model is saved using `joblib` to `model.pkl`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)

***

## Streamlit Application

The interactive web app is defined in `app.py` and uses the saved scaler and model. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/03646449-f97f-43ee-acab-095dbd7680ab/app.py)

### App Features

- Loads `scaler.pkl` and `model.pkl` with `joblib`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/03646449-f97f-43ee-acab-095dbd7680ab/app.py)
- Shows the title “Churn Prediction App”. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/03646449-f97f-43ee-acab-095dbd7680ab/app.py)
- Asks the user to input:  
  - **Age** (number input between 10 and 100). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/03646449-f97f-43ee-acab-095dbd7680ab/app.py)
  - **Tenure** (number input between 0 and 130). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/03646449-f97f-43ee-acab-095dbd7680ab/app.py)
  - **MonthlyCharges** (number input between 30 and 150). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/03646449-f97f-43ee-acab-095dbd7680ab/app.py)
  - **Gender** (selectbox: Male, Female). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/03646449-f97f-43ee-acab-095dbd7680ab/app.py)

### Prediction Logic

- Gender is converted to numeric: 1 if Female, otherwise 0. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/03646449-f97f-43ee-acab-095dbd7680ab/app.py)
- The feature vector `[Age, Gender, Tenure, MonthlyCharges]` is created and reshaped to a NumPy array. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/03646449-f97f-43ee-acab-095dbd7680ab/app.py)
- The array is scaled with the loaded `scaler`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/03646449-f97f-43ee-acab-095dbd7680ab/app.py)
- The scaled features are passed to the loaded `model` for prediction. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/03646449-f97f-43ee-acab-095dbd7680ab/app.py)
- The app outputs “Yes” if churn is predicted (class 1), otherwise “No”. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/03646449-f97f-43ee-acab-095dbd7680ab/app.py)
- On a successful prediction, Streamlit balloons are shown, and the predicted value is displayed. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/03646449-f97f-43ee-acab-095dbd7680ab/app.py)

***

## How to Run the Project

### 1. Clone or Download

Place the following files in the same project folder:

- `customer_churn_data.csv`  
- `notebook.ipynb`  
- `app.py`  
- `model.pkl` (saved from the notebook) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
- `scaler.pkl` (saved from the notebook) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

Install required libraries (for example):

```bash
pip install pandas numpy scikit-learn matplotlib streamlit joblib
```

(You can also export an exact `requirements.txt` from your environment.)

### 4. Train and Save the Model (Optional)

If you want to retrain the model:

1. Open `notebook.ipynb` in Jupyter Notebook or JupyterLab.  
2. Run all cells to:
   - Load and clean the dataset. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
   - Create features and target. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
   - Split, scale, and train the model. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)
   - Save `scaler.pkl` and `model.pkl`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/923d9701-a0d5-4d3a-a1a8-02683b1b7ccb/notebook.ipynb)

### 5. Run the Streamlit App

In the project folder, run:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) and:

1. Enter Age, Tenure, MonthlyCharges.  
2. Choose Gender.  
3. Click **Predict!** to see whether the model predicts churn or not. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/105503230/03646449-f97f-43ee-acab-095dbd7680ab/app.py)

***

## File Structure

A simple recommended structure:

```text
telekom-churn/
├─ app.py
├─ notebook.ipynb
├─ customer_churn_data.csv
├─ model.pkl
├─ scaler.pkl
└─ README.md
```