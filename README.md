# README.md for Predictive Analytics on the Influence of Various Factors on Exam Scores

## Business Understanding

In the educational sector, understanding the factors that influence student performance is crucial for improving academic outcomes. This project aims to analyze various factors affecting exam scores, providing insights that can help educators and policymakers enhance student learning experiences. By identifying key determinants of academic success, institutions can implement targeted interventions to support students, ultimately leading to improved educational outcomes.

## Data Understanding

The dataset used in this project,
StudentPerformanceFactors.csv
, contains various features related to student performance, including:

- Numerical Features:
-
Hours_Studied
: The number of hours a student studies.
-
Attendance
: The percentage of classes attended.
-
Sleep_Hours
: The average hours of sleep per night.
-
Previous_Scores
: Scores from previous assessments.
-
Tutoring_Sessions
: The number of tutoring sessions attended.
-
Physical_Activity
: The level of physical activity.
-
Exam_Score
: The score achieved in the exam.

- Categorical Features:
-
Parental_Involvement
: Level of parental involvement in education.
-
Access_to_Resources
: Availability of educational resources.
-
Extracurricular_Activities
: Participation in extracurricular activities.
-
Motivation_Level
: Student's motivation level.
-
Internet_Access
: Access to the internet for educational purposes.
-
Family_Income
: Family income level.
-
Teacher_Quality
: Quality of teaching received.
-
School_Type
: Type of school attended (public/private).
-
Peer_Influence
: Influence of peers on academic performance.
-
Learning_Disabilities
: Presence of learning disabilities.
-
Parental_Education_Level
: Education level of parents.
-
Distance_from_Home
: Distance from home to school.
-
Gender
: Gender of the student.

## Project Structure

The project is structured into several key sections:

1. Importing Libraries: Essential libraries for data manipulation, visualization, and machine learning are imported.
2. Data Loading: The dataset is loaded from a specified path or uploaded directly.
3. Data Cleaning: This section involves:
- Removing duplicates.
- Handling missing values.
- Identifying and removing outliers using the Interquartile Range (IQR) method.
4. Univariate Analysis: Analyzing individual features to summarize and find patterns.
5. Data Preparation: Preparing the data for modeling, including encoding categorical variables and scaling numerical features.
6. Modeling: Implementing various regression models (KNN, Random Forest, and Boosting) to predict exam scores.
7. Evaluation: Assessing model performance using Mean Squared Error (MSE) on training and testing datasets.

## Code Implementation

### 1. Importing Libraries


python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



### 2. Data Loading


python
# Load dataset
StudentPerformanceFactors = pd.read_csv('path/to/StudentPerformanceFactors.csv')



### 3. Data Cleaning


python
# Remove duplicates
StudentPerformanceFactors_cleaned = StudentPerformanceFactors.drop_duplicates()

# Remove missing values
StudentPerformanceFactors_cleaned = StudentPerformanceFactors_cleaned.dropna()

# Handle outliers using IQR
numerical_cols = StudentPerformanceFactors_cleaned.select_dtypes(include=[np.number])
Q1 = numerical_cols.quantile(0.25)
Q3 = numerical_cols.quantile(0.75)
IQR = Q3 - Q1
StudentPerformanceFactors_cleaned = StudentPerformanceFactors_cleaned[~((numerical_cols < (Q1 - 1.5  IQR)) | (numerical_cols > (Q3 + 1.5  IQR))).any(axis=1)]



### 4. Univariate Analysis


python
# Visualizing distributions
sns.boxplot(x=StudentPerformanceFactors_cleaned['Hours_Studied'])
plt.show()



### 5. Data Preparation


python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Encoding categorical features
categorical_features = ['Parental_Involvement', 'Access_to_Resources', ...]
label_encoders = {}
for column in categorical_features:
    le = LabelEncoder()
    StudentPerformanceFactors_cleaned[column] = le.fit_transform(StudentPerformanceFactors_cleaned[column])
    label_encoders[column] = le

# Splitting the dataset
y = StudentPerformanceFactors_cleaned["Exam_Score"]
X = StudentPerformanceFactors_cleaned.drop(["Exam_Score"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

# Scaling features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



### 6. Modeling


python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error

# KNN Model
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

# Random Forest Model
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55)
RF.fit(X_train, y_train)

# Boosting Model
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(X_train, y_train)



### 7. Evaluation


python
# Calculate MSE for each model
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN', 'RF', 'Boosting'])
mse.loc['KNN', 'train'] = mean_squared_error(y_train, knn.predict(X_train))
mse.loc['RF', 'train'] = mean_squared_error(y_train, RF.predict(X_train))
mse.loc['Boosting', 'train'] = mean_squared_error(y_train, boosting.predict(X_train))

mse.loc['KNN', 'test'] = mean_squared_error(y_test, knn.predict(X_test))
mse.loc['RF', 'test'] = mean_squared_error(y_test, RF.predict(X_test))
mse.loc['Boosting', 'test'] = mean_squared_error(y_test, boosting.predict(X_test))

# Visualize MSE
mse.sort_values(by='test', ascending=False).plot(kind='barh')
plt.title('Mean Squared Error for Each Model')
plt.show()



## Conclusion

This project provides a comprehensive analysis of the factors influencing exam scores. By employing various machine learning models, we can predict student performance based on multiple features. The insights gained from this analysis can guide educational strategies and interventions aimed at improving student outcomes.

For further inquiries or contributions, please feel free to reach out.
