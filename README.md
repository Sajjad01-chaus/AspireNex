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

# 1. Credit Card Fraud Detection

## Dataset Extraction and Loading
```python
from zipfile import ZipFile
import pandas as pd

# Extract and load the dataset
with ZipFile('/content/creditcardfraud.zip', 'r') as zip_ref:
    zip_ref.extractall()
crd_data = pd.read_csv('/content/creditcard.csv')
```

## Data Preprocessing
```python
# Check for missing values and class distribution
print(crd_data.isnull().sum())
print(crd_data['Class'].value_counts())
```

## Splitting Data
```python
from sklearn.model_selection import train_test_split
X = crd_data.drop('Class', axis=1)
Y = crd_data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```

## Balancing Data with SMOTE
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority', random_state=2)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
```

## Model Training and Evaluation
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

## Dataset Extraction and Loading
```python
!kaggle datasets download -d uciml/sms-spam-collection-dataset
from zipfile import ZipFile

# Extract and load the dataset
with ZipFile('/content/sms-spam-collection-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall()
sms_data = pd.read_csv('/content/spam.csv', encoding='latin1')
```

## Data Preprocessing
```python
# Clean and prepare the dataset
sms_data = sms_data.dropna(axis=1, how='all').rename(columns={'v1': 'Category', 'v2': 'Message'})
sms_data['Category'] = sms_data['Category'].map({'ham': 1, 'spam': 0})
X_sms = sms_data['Message']
Y_sms = sms_data['Category']
```

## Model Training and Evaluation (example with logistic regression)
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


Here's a detailed README file for your GitHub repository, covering both the credit card fraud detection and SMS spam detection tasks:

AspireNex Internship Tasks
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
Usage
Contributing
License
Credit Card Fraud Detection
Overview
The credit card fraud detection task aims to build a model that can distinguish between fraudulent and non-fraudulent transactions using a dataset of anonymized credit card transactions.

Dataset
The dataset used for this task is provided as a compressed zip file, creditcardfraud.zip, which contains the creditcard.csv file.

Preprocessing
Extract the dataset:

python
Copy code
from zipfile import ZipFile
Dataset= '/content/creditcardfraud.zip'
with ZipFile(Dataset, 'r') as zip:
   zip.extractall()
   print("Extracted")
Load the dataset:

python
Copy code
import pandas as pd
crd_data = pd.read_csv('/content/creditcard.csv')
Check for missing values:

python
Copy code
crd_data.isnull().sum()
Separate the dataset into legitimate and fraudulent transactions:

python
Copy code
legit = crd_data[crd_data.Class == 0]
fraud = crd_data[crd_data.Class == 1]
Split the data into training and testing sets:

python
Copy code
from sklearn.model_selection import train_test_split
X = crd_data.drop('Class', axis=1)
Y = crd_data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
Apply SMOTE to balance the training data:

python
Copy code
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority', random_state=2)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
Model Training
Train a Logistic Regression model on the resampled training data:

python
Copy code
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=2, max_iter=1000)
model.fit(X_train_resampled, Y_train_resampled)
Evaluation
Evaluate the model on the test data:

python
Copy code
from sklearn.metrics import accuracy_score, confusion_matrix

Y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

print("Accuracy Score:")
print(accuracy_score(Y_test, Y_pred))
Evaluate the model on the training data:

python
Copy code
Y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, Y_train_pred)
print("Training Accuracy:", train_accuracy)
SMS Spam Detection
Overview
The SMS spam detection task aims to classify SMS messages as either spam or ham (not spam) using a dataset of labeled SMS messages.

Dataset
The dataset used for this task is available on Kaggle as the "SMS Spam Collection Dataset."

Preprocessing
Download and extract the dataset:

python
Copy code
!kaggle datasets download -d uciml/sms-spam-collection-dataset
from zipfile import ZipFile
dataset= '/content/sms-spam-collection-dataset.zip'
with ZipFile(dataset, 'r') as zip:
   zip.extractall()
   print('Dataset is extracted')
Load the dataset:

python
Copy code
sms_data = pd.read_csv('/content/spam.csv', encoding='latin1')
Drop unnecessary columns:

python
Copy code
sms_data = sms_data[['v1', 'v2']]
Rename columns:

python
Copy code
sms_data.rename(columns={'v1': 'Category', 'v2': 'Message'}, inplace=True)
Replace null values and encode labels:

python
Copy code
new_data = sms_data.where((pd.notnull(sms_data)), '')
new_data.loc[new_data['Category'] == 'spam', 'Category'] = 0
new_data.loc[new_data['Category'] == 'ham', 'Category'] = 1
Separate features and labels:

python
Copy code
X = new_data['Message']
Y = new_data['Category']
Model Training
Train a Logistic Regression model using TF-IDF features:

python
Copy code
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, Y, test_size=0.2, random_state=3)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)
Evaluation
Evaluate the model on the test data:


Copy code
Y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

print("Accuracy Score:")
print(accuracy_score(Y_test, Y_pred))
Installation
Clone the repository:

Copy code
git clone https://github.com/yourusername/aspirenex.git
cd aspirenex
Install the required packages:

Copy code
pip install -r requirements.txt
```

Usage
Credit Card Fraud Detection:
Run the Jupyter notebook or Python script for the credit card fraud detection task.
SMS Spam Detection:
Run the Jupyter notebook or Python script for the SMS spam detection task.
