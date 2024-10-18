# Predictive Analytics on the Influence of Various Factors on Exam Scores

## Business Understanding

In the realm of education, understanding the factors that influence student performance is crucial for developing effective teaching strategies and interventions. This project aims to analyze various factors affecting students' exam scores, providing insights that can help educators and policymakers make informed decisions. By identifying key predictors of academic success, we can enhance educational outcomes and better support students in their learning journeys.

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

### Importing Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

Data Loading
To load the dataset, you can either download it from Google Drive or upload it from your local directory.

python

Verify

Open In Editor
Edit
Copy code
from google.colab import files
files.upload()

StudentPerformanceFactors = pd.read_csv('/content/StudentPerformanceFactors.csv')
Data Cleaning
Removing Duplicates: Duplicates in the dataset can cause bias in the analysis. Therefore, we need to remove rows that have the same values across all columns.

python

Verify

Open In Editor
Edit
Copy code
StudentPerformanceFactors_cleaned = StudentPerformanceFactors.drop_duplicates()
Removing Missing Values (NaN): Missing values can disrupt statistical analysis and visualization. Therefore, rows containing NaN values should be removed.

python

Verify

Open In Editor
Edit
Copy code
StudentPerformanceFactors_cleaned = StudentPerformanceFactors_cleaned.dropna()
Handling Outliers using IQR: Outliers can skew your analysis and lead to misleading results. One common method to detect and handle outliers is the Interquartile Range (IQR) method.

python

Verify

Open In Editor
Edit
Copy code
numerical_cols = StudentPerformanceFactors.select_dtypes(include=[np.number])
Q1 = numerical_cols.quantile(0.25)
Q3 = numerical_cols.quantile(0.75)
IQR = Q3 - Q1
StudentPerformanceFactors = StudentPerformanceFactors[~((numerical_cols < (Q1 - 1.5 * IQR)) | (numerical_cols > (Q3 + 1.5 * IQR))).any(axis=1)]
Univariate Analysis
We categorize the variables into numerical and categorical features to facilitate analysis.

python

Verify

Open In Editor
Edit
Copy code
numerical_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']
categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home', 'Gender']
Data Visualization
Boxplots and histograms are used to visualize the distribution of numerical features.

python

Verify

Open In Editor
Edit
Copy code
sns.boxplot(x=StudentPerformanceFactors['Hours_Studied'])
plt.show()

StudentPerformanceFactors.hist(bins=20, figsize=(20,15))
plt.show()

### Data Visualization (continued)

In addition to boxplots, we can also visualize the distribution of each feature using histograms.

```python
# Visualizing the distribution of numerical features
StudentPerformanceFactors.hist(bins=20, figsize=(20, 15))
plt.show()
Data Preparation for Modeling
Label Encoding for categorical features: Categorical features need to be converted into numerical format for modeling. We use label encoding to achieve this.

python

Verify

Open In Editor
Edit
Copy code
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for column in categorical_features:
    le = LabelEncoder()
    StudentPerformanceFactors[column] = le.fit_transform(StudentPerformanceFactors[column])
    label_encoders[column] = le
Train-Test Split: We split the dataset into training and testing sets to evaluate the performance of our models.

python

Verify

Open In Editor
Edit
Copy code
from sklearn.model_selection import train_test_split

y = StudentPerformanceFactors["Exam_Score"]
X = StudentPerformanceFactors.drop(["Exam_Score"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

print(f'Total # of samples in whole dataset: {len(X)}')
print(f'Total # of samples in train dataset: {len(X_train)}')
print(f'Total # of samples in test dataset: {len(X_test)}')
Feature Scaling: Standardizing the features to have a mean of 0 and a variance of 1 helps improve model performance.

python

Verify

Open In Editor
Edit
Copy code
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
Model Training
We will evaluate three different regression algorithms: K-Nearest Neighbors (KNN), Random Forest, and AdaBoost.

K-Nearest Neighbors (KNN):

python

Verify

Open In Editor
Edit
Copy code
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

train_mse_knn = mean_squared_error(y_true=y_train, y_pred=knn.predict(X_train))
print(f'Train MSE for KNN: {train_mse_knn}')
Random Forest:

python

Verify

Open In Editor
Edit
Copy code
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
rf.fit(X_train, y_train)

train_mse_rf = mean_squared_error(y_true=y_train, y_pred=rf.predict(X_train))
print(f'Train MSE for Random Forest: {train_mse_rf}')
AdaBoost:

python

Verify

Open In Editor
Edit
Copy code
from sklearn.ensemble import AdaBoostRegressor

boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(X_train, y_train)

train_mse_boosting = mean_squared_error(y_true=y_train, y_pred=boosting.predict(X_train))
print(f'Train MSE for AdaBoost: {train_mse_boosting}')
Model Evaluation
We will evaluate the Mean Squared Error (MSE) for both the training and testing datasets for each model.

python

Verify

Open In Editor
Edit
Copy code
# Create a DataFrame to store MSE results
mse_results = pd.DataFrame(columns=['Train MSE', 'Test MSE'], index=['KNN', 'Random Forest', 'AdaBoost'])

# Evaluate models on training data
mse_results.loc['KNN', 'Train MSE'] = train_mse_knn
mse_results.loc['Random Forest', 'Train MSE'] = train_mse_rf
mse_results.loc['AdaBoost', 'Train MSE'] = train_mse_boosting

# Evaluate models on test data
mse_results.loc['KNN', 'Test MSE'] = mean_squared_error(y_true=y_test, y_pred=knn.predict(X_test))
mse_results.loc['Random Forest', 'Test MSE'] = mean_squared_error(y_true=y_test, y_pred=rf.predict(X_test))
mse_results.loc['AdaBoost', 'Test MSE'] = mean_squared_error(y_true=y_tes
