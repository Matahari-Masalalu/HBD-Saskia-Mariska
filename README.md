# Predictive Analytics on the Influence of Various Factors on Exam Scores

## Business Understanding

In the field of education, understanding the factors that influence student performance is crucial for developing effective teaching strategies and interventions. The objective of this project is to analyze various factors affecting students' exam scores. By identifying the key predictors of academic success, we can provide insights that help educators and policymakers make informed decisions. This analysis aims to enhance educational outcomes and better support students in their learning journeys.

## Data Understanding

The dataset used in this project contains various features that may influence students' exam scores. These features include both numerical and categorical variables, such as:

- **Numerical Features**:
  - `Hours_Studied`: The number of hours a student studies.
  - `Attendance`: The percentage of classes attended.
  - `Sleep_Hours`: Average hours of sleep per night.
  - `Previous_Scores`: Scores from previous assessments.
  - `Tutoring_Sessions`: Number of tutoring sessions attended.
  - `Physical_Activity`: Level of physical activity (e.g., hours per week).
  - `Exam_Score`: The score achieved in the exam.

- **Categorical Features**:
  - `Parental_Involvement`: Level of parental involvement in education.
  - `Access_to_Resources`: Availability of educational resources.
  - `Extracurricular_Activities`: Participation in extracurricular activities.
  - `Motivation_Level`: Self-reported motivation level.
  - `Internet_Access`: Access to the internet for educational purposes.
  - `Family_Income`: Income level of the family.
  - `Teacher_Quality`: Assessment of teacher quality.
  - `School_Type`: Type of school (public/private).
  - `Peer_Influence`: Influence of peers on academic performance.
  - `Learning_Disabilities`: Presence of any learning disabilities.
  - `Parental_Education_Level`: Education level of parents.
  - `Distance_from_Home`: Distance from home to school.
  - `Gender`: Gender of the student.

## Data Preparation

### 1. Importing Libraries

We will begin by importing the necessary libraries for our analysis.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
```

### 2. Data Loading
To load the dataset, you can either download it from Google Drive or upload it from your local directory.

```python
from google.colab import files
files.upload()

StudentPerformanceFactors = pd.read_csv('/content/StudentPerformanceFactors.csv')
```

### 3. Data Cleaning
Data cleaning is an essential step to ensure the quality of our dataset. This involves:

## 1. Removing Duplicates 
Duplicates in the dataset can cause bias in the analysis. Therefore, we need to remove rows that have the same values across all columns.

```python
StudentPerformanceFactors_cleaned = StudentPerformanceFactors.drop_duplicates()
```

## 2. Removing Missing Values (NaN)
Missing values can disrupt statistical analysis and visualization. Therefore, rows containing NaN values should be removed.

```python
StudentPerformanceFactors_cleaned = StudentPerformanceFactors_cleaned.dropna()
```

## 3.Handling Outliers using IQR
Outliers can skew your analysis and lead to misleading results. One common method to detect and handle outliers is the Interquartile Range (IQR) method.

```python
numerical_cols = StudentPerformanceFactors_cleaned.select_dtypes(include=[np.number])
Q1 = numerical_cols.quantile(0.25)
Q3 = numerical_cols.quantile(0.75)
IQR = Q3 - Q1
StudentPerformanceFactors_cleaned = StudentPerformanceFactors_cleaned[~((numerical_cols < (Q1 - 1.5 * IQR)) | (numerical_cols > (Q3 + 1.5 * IQR))).any(axis=1)]
```

### 4. Univariate Analysis
Univariate analysis is a statistical technique that involves the examination of a single variable in a dataset. The primary goal is to summarize and find patterns within that variable without considering relationships with other variables.

To facilitate data analysis and visualization, we categorize the variables in the dataset into two main types: numerical variables and categorical variables.

python

Verify

Open In Editor
Edit
Copy code
numerical_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']
categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Motivation_Level', '

### 5. Multivariate Analysis
Multivariate analysis involves examining the relationships between multiple variables. This helps in understanding how different factors interact with each other and their combined effect on exam scores.

```python
# Correlation matrix for numerical features
plt.figure(figsize=(12, 8))
correlation_matrix = StudentPerformanceFactors_cleaned[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()
```

### 6. Model Training
Setelah memahami data, langkah selanjutnya adalah melatih model prediktif menggunakan algoritma regresi. Kami akan menggunakan tiga model: K-Nearest Neighbors, Random Forest, dan AdaBoost.

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error

# Splitting the data into training and testing sets
X = StudentPerformanceFactors_cleaned.drop('Exam_Score', axis=1)
y = StudentPerformanceFactors_cleaned['Exam_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-Nearest Neighbors
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
train_mse_knn = mean_squared_error(y_true=y_train, y_pred=knn.predict(X_train))

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
train_mse_rf = mean_squared_error(y_true=y_train, y_pred=rf.predict(X_train))

# AdaBoost
boosting = AdaBoostRegressor()
boosting.fit(X_train, y_train)
train_mse_boosting = mean_squared_error(y_true=y_train, y_pred=boosting.predict(X_train))
```

### 7. Model Evaluation
Setelah melatih model, kita akan mengevaluasi kinerjanya menggunakan Mean Squared Error (MSE) untuk dataset pelatihan dan pengujian.
```Python
# Create a DataFrame to store MSE results
mse_results = pd.DataFrame(columns=['Train MSE', 'Test MSE'], index=['KNN', 'Random Forest', 'AdaBoost'])

# Evaluate models on training data
mse_results.loc['KNN', 'Train MSE'] = train_mse_knn
mse_results.loc['Random Forest', 'Train MSE'] = train_mse_rf
mse_results.loc['AdaBoost', 'Train MSE'] = train_mse_boosting

# Evaluate models on test data
mse_results.loc['KNN', 'Test MSE'] = mean_squared_error(y_true=y_test, y_pred=knn.predict(X_test))
mse_results.loc['Random Forest', 'Test MSE'] = mean_squared_error(y_true=y_test, y_pred=rf.predict(X_test))
mse_results.loc['AdaBoost', 'Test MSE'] = mean_squared_error(y_true=y_test, y_pred=boosting.predict(X_test))

print(mse_results)
```
