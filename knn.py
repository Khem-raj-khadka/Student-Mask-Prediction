import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class MyKNNRegressor:
    def __init__(self, k=5): # Increased K slightly for larger dataset
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        
        # Ensure 2D arrays for consistency
        if len(self.X_train.shape) == 1:
            self.X_train = self.X_train.reshape(-1, 1)
            
        print(f"KNN Model trained (data stored)! K={self.k}")

    def predict(self, X_test):
        X_test = np.array(X_test)
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(-1, 1)
            
        predictions = []
        for x in X_test:
            # 1. Calculate Euclidean distances from x to all points in X_train
            # distance = sqrt(sum((x1 - x2)^2))
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            
            # 2. Get indices of the k nearest neighbors
            # argsort returns indices that would sort the array
            k_indices = np.argsort(distances)[:self.k]
            
            # 3. Get the values (y) of these neighbors
            k_nearest_values = self.y_train[k_indices]
            
            # 4. Predict by averaging the fit values
            predictions.append(np.mean(k_nearest_values))
            
        return np.array(predictions)

if __name__ == "__main__":
    print("--- KNN Regressor Model ---")
    
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
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    model = MyKNNRegressor(k=5)
    model.fit(X_train, y_train)
    
    # Predict
    print("Predicting... (this might take a moment with 1500 records)")
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R2 Score: {r2:.4f}")
