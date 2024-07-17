
# EDA and Visualization

This Jupyter notebook performs exploratory data analysis (EDA) and visualization on a Customer Segmentation Dataset for Marketing Analysis. Here is the link for dataset : 
https://www.kaggle.com/datasets/fahmidachowdhury/customer-segmentation-data-for-marketing-analysis/code

## About the dataset

This dataset contains simulated customer data that can be used for segmentation analysis. It includes demographic and behavioral information about customers, which can help in identifying distinct segments within the customer base. This can be particularly useful for targeted marketing strategies, improving customer satisfaction, and increasing sales.

## Columns:

#### id: 
Unique identifier for each customer.

#### age: 
Age of the customer.

#### gender: 
Gender of the customer (Male, Female, Other).

#### income: 
Annual income of the customer (in USD).

#### spending_score: 
Spending score (1-100), indicating the customer's spending behavior and loyalty.

#### membership_years: 
Number of years the customer has been a member.

#### purchase_frequency: 
Number of purchases made by the customer in the last year.

#### preferred_category: 
Preferred shopping category (Electronics, Clothing, Groceries, Home & Garden, Sports).

#### last_purchase_amount: 
Amount spent by the customer on their last purchase (in USD).

## Potential Uses:
#### Customer Segmentation: 
Identify different customer segments based on their demographic and behavioral characteristics.

#### Targeted Marketing: 
Develop targeted marketing strategies for different customer segments.

#### Customer Loyalty Programs: 
Design loyalty programs based on customer spending behavior and preferences.

#### Sales Analysis: 
Analyze sales patterns and predict future trends.

## Installation and Usage
Open a command prompt or terminal.

Install Jupyter using pip by running:
```python
pip install jupyter
```
Open a jupyter notebook
```python
jupyter notebook
```
## Necessary Imports
```python
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFE
from sklearn.ensemble import VotingClassifier
```
## Install all the imports with a single command :
```python
pip install -r requirements.txt
```