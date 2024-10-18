# Predictive Analytics on the Influence of Various Factors on Exam Scores

## Table of Contents
1. [Business Understanding](#business-understanding)
2. [Data Understanding](#data-understanding)
3. [Data Loading](#data-loading)
4. [Data Cleaning](#data-cleaning)
5. [Univariate Analysis](#univariate-analysis)
6. [Data Preparation](#data-preparation)
7. [Modeling](#modeling)
8. [Results](#results)

---

## Business Understanding

### Overview
The education sector is increasingly leveraging data analytics to improve student performance and educational outcomes. Understanding the various factors that influence exam scores can help educators, policymakers, and stakeholders make informed decisions to enhance learning environments and student support.

### Objectives
- **Identify Key Factors:** Analyze which variables most significantly affect student exam scores.
- **Improve Student Performance:** Provide insights that can lead to targeted interventions for students who may be struggling.
- **Resource Allocation:** Help educational institutions allocate resources more effectively based on data-driven insights.

### Stakeholders
- **Educators:** Teachers and academic staff looking to improve teaching methods and student support.
- **Administrators:** School and district administrators aiming to enhance overall student performance.
- **Policymakers:** Government and educational policymakers interested in data to support funding and program development.
- **Students and Parents:** Individuals seeking to understand the factors that contribute to academic success.

---

## Data Understanding

### Dataset Description
The dataset used in this project contains various factors that may influence student performance, specifically their exam scores. The features include both numerical and categorical variables, reflecting different aspects of students' academic and personal lives.

### Features
- **Numerical Features:**
  - `Hours_Studied`: The number of hours a student studies per week.
  - `Attendance`: The percentage of classes attended by the student.
  - `Sleep_Hours`: Average hours of sleep per night.
  - `Previous_Scores`: Scores from previous exams.
  - `Tutoring_Sessions`: Number of tutoring sessions attended.
  - `Physical_Activity`: Level of physical activity (e.g., low, medium, high).
  - `Exam_Score`: The score obtained in the exam (target variable).

- **Categorical Features:**
  - `Parental_Involvement`: Level of involvement from parents in the student's education.
  - `Access_to_Resources`: Availability of educational resources at home.
  - `Extracurricular_Activities`: Participation in extracurricular activities.
  - `Motivation_Level`: Self-reported motivation level of the student.
  - `Internet_Access`: Availability of internet access at home.
  - `Family_Income`: Family income level.
  - `Teacher_Quality`: Perceived quality of teaching.
  - `School_Type`: Type of school attended (e.g., public, private).
  - `Peer_Influence`: Influence of peers on academic performance.
  - `Learning_Disabilities`: Any learning disabilities reported.
  - `Parental_Education_Level`: Education level of parents.
  - `Distance_from_Home`: Distance from home to school.
  - `Gender`: Gender of the student.

### Data Quality
Before proceeding with analysis, it is essential to assess the quality of the data, including checking for missing values, duplicates, and outliers. This will ensure that the analysis is based on reliable and accurate information.

---

## Data Loading

To load the dataset, you can either download it from Google Drive or use the provided link directly in your code.

- **Link:** [Download Dataset](https://drive.google.com/file/d/1AtAF_wqfwG_fkAc5ChzRPENdkiKjWOa1/view?usp=sharing)

### Uploading and Loading the Dataset
Make sure to replace `'path/to/StudentPerformanceFactors.csv'` with the actual path where you saved the file on your machine.

```python
from google.colab import files
files.upload()

import pandas as pd

# Load the dataset
StudentPerformanceFactors = pd.read_csv('/content/StudentPerformanceFactors.csv')
print(StudentPerformanceFactors.head())
