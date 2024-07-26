# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import shap
import warnings
import missingno as msno
from ydata_profiling import ProfileReport
from scipy import stats
from yellowbrick.cluster import KElbowVisualizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFE
from sklearn.ensemble import VotingClassifier

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot styles
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Title of the app
st.title('Customer Segmentation Analysis')

# Generate synthetic data
@st.cache_data
def load_data():
    return pd.read_csv('customer_segmentation_data.csv')

data = load_data()

# Display the data
st.subheader('Data')
st.write(data.head())

# Profile Report
st.subheader('Data Profile Report')
profile = ProfileReport(data, title="Customer Segmentation Data Profile", explorative=True)
st_profile_report = st.checkbox('Display Profile Report')
if st_profile_report:
    st_profile_report(profile)

# Missing values plot
st.subheader('Missing Values Matrix')
fig, ax = plt.subplots()
msno.matrix(data, ax=ax)
st.pyplot(fig)

# Check for missing values
st.subheader('Missing Values Count')
missing_values = data.isnull().sum()
st.write(missing_values)

# Basic statistics
st.subheader('Basic Statistics')
st.write(data.describe())

# Display columns
st.subheader('Columns')
st.write(data.columns)

# Data types
st.subheader('Data Types')
st.write(data.dtypes)

# Unique values in each column
st.subheader('Unique Values in Each Column')
st.write(data.nunique())

# Display categorical column unique values
st.subheader('Categorical Columns Unique Values')
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    st.write(f"{column} Values:")
    st.write(data[column].unique())

# Data preprocessing and model training
st.subheader('Model Training')
target = st.selectbox('Select Target Variable', data.columns)
features = st.multiselect('Select Features', data.columns)

if target and features:
    # Train-test split
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RandomForest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Predictions and evaluation
    y_pred = rf_model.predict(X_test_scaled)
    st.write('Classification Report:')
    st.text(classification_report(y_test, y_pred))
    st.write('Confusion Matrix:')
    st.write(confusion_matrix(y_test, y_pred))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test_scaled)[:,1])
    roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:,1])
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # SHAP values
    st.subheader('SHAP Values')
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_scaled)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    st.pyplot(fig)