import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class MyLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.theta_ = None # Combined parameters

    def fit(self, X_train, y_train):
        # Implementation of Multiple Linear Regression using Normal Equation
        # Formula: theta = (X_transpose * X)^(-1) * X_transpose * y
        
        # 1. Prepare X matrix
        # Add a column of ones to X_train for the intercept term (bias)
        m_samples = len(X_train)
        X_b = np.c_[np.ones((m_samples, 1)), X_train]  # Add x0 = 1 to each instance
        
        # 2. Compute Normal Equation
        # theta_best = inv(X.T dot X) dot X.T dot y
        X_transpose = X_b.T
        
        # We use np.linalg.pinv (Pseudo Inverse) for stability or np.linalg.inv
        # (X^T * X) might not be invertible if features are redundant
        try:
            self.theta_ = np.linalg.inv(X_transpose.dot(X_b)).dot(X_transpose).dot(y_train)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            self.theta_ = np.linalg.pinv(X_b).dot(y_train)
            
        # 3. Extract Intercept and Coefficients
        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]
        
        print(f"Model trained internally! Intercept: {self.intercept_:.4f}")
        print(f"Coefficients: {self.coef_}")

    def predict(self, X_test):
        # Prepare X_test
        m_samples = len(X_test)
        X_b = np.c_[np.ones((m_samples, 1)), X_test]
        
        # Returns inputs * theta
        return X_b.dot(self.theta_)

if __name__ == "__main__":
    print("--- Linear Regression Model ---")
    
    # Load data
    try:
        df = pd.read_csv('student_scores_extended.csv')
    except FileNotFoundError:
        print("Error: 'student_scores_extended.csv' not found. Please run generate_data.py first.")
        exit(1)
        
    print(f"Loaded dataset with {len(df)} records.")
    
    # Features and Target
    X = df[['Hours', 'Attendance', 'Assignment']].values
    y = df['Scores'].values
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    model = MyLinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R2 Score: {r2:.4f}")
