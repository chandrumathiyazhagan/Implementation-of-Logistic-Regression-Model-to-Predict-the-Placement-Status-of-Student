# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.  

## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: M.CHANDRU

RegisterNumber:  212222230026

```python
import pandas as pd
```
```python
data=pd.read_csv('/content/Placement_Data.csv')
data.head()
```
```python
data1=data.copy()
data1.head()
```
```python
data1=data1.drop(['sl_no','salary'],axis=1)
```
```python
data1.duplicated().sum()
```
```python
data1
```
```python
x=data1.iloc[:,:-1]
x
```
```python
y=data1["status"]
y
```
```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```python
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='liblinear')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
```
```python
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("AccuracyScore:",accuracy)
print("\nConfusion Matrix:\n",confusion)
print("\nCLassification Report:\n",cr)
```
```python
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
```

## Output:

## Accuracy Score and Classification Report:
![Screenshot 2024-03-12 214636](https://github.com/chandrumathiyazhagan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393023/5dde5294-482e-4c0c-9a77-7ec1b0d8dfeb)

## Displaying:
![Screenshot 2024-03-12 214648](https://github.com/chandrumathiyazhagan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393023/6d1510ce-7ce5-4aba-9f1d-f69df520a346)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
