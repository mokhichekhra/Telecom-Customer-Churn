# Telekom Customer Churn Prediction

Author: **Mokhichekhra Makhmudova**

***

## Project Overview

This is a small end‑to‑end machine learning project:

- You explore and clean Telekom customer data in a Jupyter Notebook. 
- You train and save a machine learning model and a scaler. 
- You build a simple Streamlit web app to make churn predictions for new customers. 

The goal is to give a clear, beginner‑friendly example of a churn prediction workflow that can later be extended with more features.

***

## Dataset

The dataset is stored in `customer_churn_data.csv` and contains 1,000 rows and 10 columns.

Main columns:

- **CustomerID**: Unique customer identifier. 
- **Age**: Customer age (integer). 
- **Gender**: Male or Female. 
- **Tenure**: Number of months the customer has stayed with the company. 
- **MonthlyCharges**: Monthly bill amount. 
- **ContractType**: Month‑to‑Month, One‑Year, or Two‑Year. 
- **InternetService**: None, DSL, or Fiber Optic. 
- **TotalCharges**: Total amount charged during the contract. 
- **TechSupport**: Yes or No. 
- **Churn**: Target variable, Yes if the customer churned, otherwise No. 

In this version of the model, only a subset of features is used:

- **Age**  
- **Gender** (encoded as 1 = Female, 0 = Male)
- **Tenure**  
- **MonthlyCharges** 

***

## Exploratory Data Analysis (EDA)

EDA is done in `notebook.ipynb`. 

Key steps:

- **Data inspection**: `df.head()`, `df.info()`, and `df.describe()` to understand basic structure and summary statistics. 
- **Missing values**: `InternetService` missing values are filled before modeling. 
- **Duplicates**: Checked with `df.duplicated().sum()`; none are kept. 
- **Correlation analysis**: A correlation matrix is created for numeric columns (CustomerID, Age, Tenure, MonthlyCharges, TotalCharges). 
- **Churn distribution**: Pie chart of churn vs non‑churn customers. 
- **Group statistics**:  
  - Mean MonthlyCharges by Churn. 
  - Mean MonthlyCharges by Churn and Gender. 
  - Mean Tenure by Churn. 
  - Mean Age by Churn. 
- **Visualizations**:  
  - Histogram of Tenure. 
  - Histogram of MonthlyCharges. 
  - Bar chart of average MonthlyCharges by ContractType. 

These steps help to understand how churn is related to age, monthly charges, and tenure.

***

## Modeling

All modeling code is in `notebook.ipynb`. 

### Feature Engineering

- Input features **X**: `Age`, `Gender`, `Tenure`, `MonthlyCharges`. 
- Target **y**: `Churn`, encoded as 1 for Yes and 0 for No. 
- `Gender` is converted to numeric: 1 = Female, 0 = Male. 

### Train–Test Split

- Train–test split with 80% training and 20% test data using `train_test_split` from scikit‑learn. 
### Scaling

- Numerical features are standardized using `StandardScaler`. 
- The scaler is fitted on the training data and saved using `joblib` to `scaler.pkl`. 

### Model

- The main classifier is **Logistic Regression** (`sklearn.linear_model.LogisticRegression`). 
- The model is trained on the scaled training data. 
- Model performance is evaluated using accuracy on the test set. 
- The trained model is saved using `joblib` to `model.pkl`.

***

## Streamlit Application

The interactive web app is defined in `app.py` and uses the saved scaler and model. 

### App Features

- Loads `scaler.pkl` and `model.pkl` with `joblib`. 
- Shows the title “Churn Prediction App”. 
- Asks the user to input:  
  - **Age** (number input between 10 and 100). 
  - **Tenure** (number input between 0 and 130).
  - **MonthlyCharges** (number input between 30 and 150). 
  - **Gender** (selectbox: Male, Female).

### Prediction Logic

- Gender is converted to numeric: 1 if Female, otherwise 0. 
- The feature vector `[Age, Gender, Tenure, MonthlyCharges]` is created and reshaped to a NumPy array. 
- The array is scaled with the loaded `scaler`. 
- The scaled features are passed to the loaded `model` for prediction. 
- The app outputs “Yes” if churn is predicted (class 1), otherwise “No”. 
- On a successful prediction, Streamlit balloons are shown, and the predicted value is displayed. 

***

## How to Run the Project

### 1. Clone or Download

Place the following files in the same project folder:

- `customer_churn_data.csv`  
- `notebook.ipynb`  
- `app.py`  
- `model.pkl` 
- `scaler.pkl` 

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
   - Load and clean the dataset.
   - Create features and target. 
   - Split, scale, and train the model. 
   - Save `scaler.pkl` and `model.pkl`. 

### 5. Run the Streamlit App

In the project folder, run:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) and:

1. Enter Age, Tenure, MonthlyCharges.  
2. Choose Gender.  
3. Click **Predict!** to see whether the model predicts churn or not. 
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
