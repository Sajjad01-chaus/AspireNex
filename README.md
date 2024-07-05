# AspireNex
The Repo for ML task provided by AspireNex 

Overview
This repository contains the code and documentation for two machine learning tasks completed as part of the AspireNex Machine Learning internship. The tasks include:

Credit Card Fraud Detection
SMS Spam Detection
Table of Contents
Credit Card Fraud Detection
Overview
Dataset
Preprocessing
Model Training
Evaluation
SMS Spam Detection
Overview
Dataset
Preprocessing
Model Training
Evaluation
Installation
Usage.

Sure, let's make the README file shorter while still covering the essential points.

---

# AspireNex Internship Projects

## 1. Credit Card Fraud Detection

### Dataset Extraction and Loading
```python
from zipfile import ZipFile
import pandas as pd

# Extract and load the dataset
with ZipFile('/content/creditcardfraud.zip', 'r') as zip_ref:
    zip_ref.extractall()
crd_data = pd.read_csv('/content/creditcard.csv')
```

### Data Preprocessing
```python
# Check for missing values and class distribution
print(crd_data.isnull().sum())
print(crd_data['Class'].value_counts())
```

### Splitting Data
```python
from sklearn.model_selection import train_test_split
X = crd_data.drop('Class', axis=1)
Y = crd_data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```

### Balancing Data with SMOTE
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority', random_state=2)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
```

### Model Training and Evaluation
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

model = LogisticRegression(random_state=2, max_iter=1000)
model.fit(X_train_resampled, Y_train_resampled)
Y_pred = model.predict(X_test)

# Evaluate model
print("Confusion Matrix:", confusion_matrix(Y_test, Y_pred))
print("Accuracy Score:", accuracy_score(Y_test, Y_pred))
```

## 2. SMS Spam Detection

### Dataset Extraction and Loading
```python
!kaggle datasets download -d uciml/sms-spam-collection-dataset
from zipfile import ZipFile

# Extract and load the dataset
with ZipFile('/content/sms-spam-collection-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall()
sms_data = pd.read_csv('/content/spam.csv', encoding='latin1')
```

### Data Preprocessing
```python
# Clean and prepare the dataset
sms_data = sms_data.dropna(axis=1, how='all').rename(columns={'v1': 'Category', 'v2': 'Message'})
sms_data['Category'] = sms_data['Category'].map({'ham': 1, 'spam': 0})
X_sms = sms_data['Message']
Y_sms = sms_data['Category']
```

### Model Training and Evaluation (example with logistic regression)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Vectorize text data
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_sms_tfidf = tfidf.fit_transform(X_sms)

# Split data
X_sms_train, X_sms_test, Y_sms_train, Y_sms_test = train_test_split(X_sms_tfidf, Y_sms, test_size=0.2, random_state=2)

# Train and evaluate model
model_sms = LogisticRegression()
model_sms.fit(X_sms_train, Y_sms_train)
Y_sms_pred = model_sms.predict(X_sms_test)

print("Confusion Matrix:", confusion_matrix(Y_sms_test, Y_sms_pred))
print("Accuracy Score:", accuracy_score(Y_sms_test, Y_sms_pred))
```

---


Usage
Credit Card Fraud Detection:
Run the Jupyter notebook or Python script for the credit card fraud detection task.
SMS Spam Detection:
Run the Jupyter notebook or Python script for the SMS spam detection task.
