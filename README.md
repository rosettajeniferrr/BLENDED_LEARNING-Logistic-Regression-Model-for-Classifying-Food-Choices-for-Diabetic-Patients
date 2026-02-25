# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the food items dataset and separate features (X) and target variable (y).
2. Normalize the input features using MinMaxScaler and encode the target labels using LabelEncoder.
3. Split the dataset into training and testing sets with stratified sampling.
4. Train a Logistic Regression model with L2 regularization on the training data.
5. Predict test results and evaluate performance using accuracy, confusion matrix, precision, recall, and F1-score.

## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: Rosetta Jenifer.C
RegisterNumber:212225230230
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('food_items (1).csv')
print('Name: Rosetta Jenifer.C')
print('Reg.No: 212225230230')
print('Dataset Overview')
print(df.head())
print("\nDataset Info:")
print(df.info())
X_raw=df.iloc[:,:-1]
y_raw=df.iloc[:,-1:]
scaler=MinMaxScaler()
X=scaler.fit_transform(X_raw)
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y_raw.values.ravel())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)
penalty='l2'
multi_class='multinomial'
solver='lbfgs'
max_iter=1000
l2_model=LogisticRegression(random_state=123,penalty=penalty,multi_class=multi_class,solver=solver,max_iter=max_iter)
l2_model.fit(X_train,y_train)
y_pred=l2_model.predict(X_test)
print('Name: Rosetta Jenifer.C')
print('Reg.No: 212225230230')
print("\nModel Evaluation:")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred))
conf_matrix=confusion_matrix(y_test,y_pred)
print(conf_matrix)
print('Name: Rosetta Jenifer.C')
print('Reg.No: 212225230230')
```



## Output:
<img width="636" height="647" alt="image" src="https://github.com/user-attachments/assets/5996bde5-6fbe-44c8-8f8a-10f1c0a55a73" />
<img width="641" height="425" alt="image" src="https://github.com/user-attachments/assets/f3c42bcf-6844-4f62-92b4-a8babc8dee38" />
<img width="572" height="346" alt="image" src="https://github.com/user-attachments/assets/747c316f-2d21-4809-ba8c-2862274435a0" />



## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
