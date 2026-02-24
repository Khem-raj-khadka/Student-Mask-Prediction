# Student Marks Prediction System
# This project predicts a student's marks based on the number of hours they study.
# It uses a simple Machine Learning algorithm called "Linear Regression".

# ==========================================
# 1. IMPORTING LIBRARIES
# ==========================================
import pandas as pd  # Used for handling dataset (loading and viewing data)
import numpy as np   # Used for numerical calculations
import matplotlib.pyplot as plt  # Used for data visualization (graphs)
from sklearn.model_selection import train_test_split  # Used to split data into training and testing sets
# from sklearn.linear_model import LinearRegression  # REMOVED: implementation reference
from sklearn.metrics import mean_absolute_error, r2_score  # Metrics to check how good our model is

print("Libraries imported successfully!")

# ==========================================
# 2. LOADING THE DATASET
# ==========================================
# Reading data from the CSV file
# Ensure 'student_scores.csv' is in the same folder as this script
url = "student_scores_extended.csv"
try:
    s_data = pd.read_csv(url)
    print("\nData imported successfully!")
    
    # Displaying the first 5 rows of the data to understand its structure
    print("\nFirst 5 rows of the dataset:")
    print(s_data.head())
except FileNotFoundError:
    print(f"Error: '{url}' file not found. Please run 'generate_data.py' first.")
    exit()

# ==========================================
# 3. DATA VISUALIZATION
# ==========================================
# Since we have multiple inputs (3 dimensions), a simple 2D plot isn't enough to show everything.
# We can check correlation matrix.
print("\nCorrelation Matrix:")
print(s_data.corr())


# ==========================================
# 4. DATA PREPROCESSING
# ==========================================
# We need to separate the data into:
# Inputs (Attributes) -> 'Hours' (X)
# Outputs (Labels) -> 'Scores' (y)

# Inputs (Attributes) -> 'Hours', 'Attendance', 'Assignment'
# Outputs (Labels) -> 'Scores'

X = s_data.iloc[:, :-1].values  # Select all columns except the last one (Scores)
y = s_data.iloc[:, -1].values   # Select only the last column (Scores)

# ==========================================
# 5. SPLITTING DATA INTO TRAINING AND TESTING SETS
# ==========================================
# We split the data: 
# 80% for Training (teaching the model)
# 20% for Testing (checking if the model learned correctly)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

print("\nData split done!")
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

from linearregression import MyLinearRegression
from knn import MyKNNRegressor

# ==========================================
# 6. TRAINING THE ALGORITHM
# ==========================================
# We create a Linear Regression model and train it with our training data.

regressor = MyLinearRegression()  
regressor.fit(X_train, y_train)

print("\nTraining complete.")

# ==========================================
# 7. MAKING PREDICTIONS
# ==========================================
# Now that the model is trained, let's test it using the testing data (X_test)

print("\nMaking predictions on test data...")
y_pred = regressor.predict(X_test) # Predicting the scores

# Comparing Actual vs Predicted Scores
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print("\nComparing Actual vs Predicted Scores (First 5):")
print(df.head())

# ==========================================
# 8. EVALUATING THE MODEL
# ==========================================
# We verify how accurate our model is using "Mean Absolute Error".
# Lower error means better accuracy.

print("\nEvaluating Model Performance:")
print(f"--- Linear Regression ---")
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print(f'Model Accuracy (R2 Score): {r2_score(y_test, y_pred) * 100:.2f}%')

# ==========================================
# NEW MODEL: K-NEAREST NEIGHBORS (MANUAL)
# ==========================================
        
# ==========================================
# TRAINING & EVALUATING KNN MODEL
# ==========================================
knn_regressor = MyKNNRegressor(k=5) # Increased k slightly for more data
knn_regressor.fit(X_train, y_train)

print("\nMaking predictions with KNN...")
knn_pred = knn_regressor.predict(X_test)

print(f"\n--- KNN Regression ---")
print('Mean Absolute Error:', mean_absolute_error(y_test, knn_pred))
print(f'Model Accuracy (R2 Score): {r2_score(y_test, knn_pred) * 100:.2f}%')


# ==========================================
# 9. PREDICT FOR YOUR OWN DATA
# ==========================================
# Let's test with a custom input
# Format: [Hours, Attendance, Assignment]

hours = 5
attendance = 90
assignment = 8

input_data = [[hours, attendance, assignment]] 
own_pred = regressor.predict(input_data)

print(f"\n--- Custom Prediction ---")
print(f"Hours: {hours}, Attendance: {attendance}%, Assignment: {assignment}")
print(f"Predicted Score = {own_pred[0]:.2f}")

print("\nProject execution finished successfully!")
