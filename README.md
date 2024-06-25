# Decision Tree Classifier on Titanic Dataset

This project demonstrates the use of a Decision Tree Classifier on the Titanic dataset to predict the survival of passengers based on various features.

## Output


https://github.com/sarvesh-2109/Descision_Tree/assets/113255836/bf001390-cbbc-4829-806b-8aea0e464c5f



## Overview

The Titanic dataset provides information on the passengers, including whether they survived or not, their age, sex, and other features. This project involves data preprocessing, training a Decision Tree model, and evaluating its performance.

## Project Steps

1. **Importing Libraries**: Necessary libraries such as pandas and scikit-learn are imported.
2. **Loading the Dataset**: The Titanic dataset is loaded into a pandas DataFrame.
3. **Data Preprocessing**: The dataset is cleaned by dropping unnecessary columns and handling missing values.
4. **Encoding Categorical Variables**: The 'Sex' column is encoded to numeric values using LabelEncoder.
5. **Model Training**: A Decision Tree Classifier is trained on the preprocessed data.
6. **Model Evaluation**: The accuracy of the model is evaluated on the test set.

## Libraries Used

- pandas
- scikit-learn

## Dataset

The dataset used is the Titanic dataset, which can be found on [Kaggle](https://www.kaggle.com/c/titanic/data).

## Code

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree

# Loading the dataset
df = pd.read_csv('/content/titanic.csv')
df.head()

# Data Preprocessing
data = df.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'])
data.head()

inputs = data.drop(columns=['Survived'])
target = data['Survived']

inputs.isnull().sum()

inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.isnull().sum()

# Encoding Categorical Variables
le_sex = LabelEncoder()

inputs['Sex_n'] = le_sex.fit_transform(inputs['Sex'])
inputs = inputs.drop(columns=['Sex'])
inputs.head()

# Model Training
model = tree.DecisionTreeClassifier()

X_train , X_test , y_train , y_test = train_test_split(inputs, target, test_size=0.2)

model.fit(X_train, y_train)

# Model's Accuracy
model.score(X_test, y_test)
```

## Results

The trained Decision Tree Classifier is evaluated on the test set, and its accuracy is obtained using the `score` method.

## Conclusion

This project demonstrates a basic implementation of a Decision Tree Classifier for binary classification on the Titanic dataset. The preprocessing steps and model training provide a foundation for more complex machine learning workflows.

