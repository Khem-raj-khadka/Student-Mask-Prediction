# Student Marks Prediction - Comprehensive Project Report

## 1. Executive Summary
This project implements a Machine Learning system to predict student marks based on study habits and other factors. It features a dual-model approach using **Linear Regression** and **K-Nearest Neighbors (KNN)**, allowing for performance comparison. The system includes a synthetic data generator, standalone model implementations, and an interactive web dashboard built with Streamlit.

---

## 2. Project Architecture & File Structure

```text
d:\student_mask_prediction\
│
├── 📄 app.py                     # Main Web Application (Streamlit)
├── 📄 generate_data.py           # Data Generation Script
├── 📄 knn.py                     # KNN Model Implementation & Logic
├── 📄 linearregression.py        # Linear Regression Model Implementation & Logic
├── 📄 marks_prediction.py        # Command-Line Interface (CLI) Version
├── 📄 PROJECT_REPORT.md          # This Documentation File
├── 📄 student_scores_extended.csv # The Dataset (1500 records)
└── 📄 requirements.txt           # Python Dependencies
```

---

## 3. Detailed File Analysis

### 3.1 `generate_data.py`
**Purpose**: Creates the dataset used for training the models.
**Key Logic**:
- **Random Seed**: Uses `np.random.seed(42)` to ensure data is consistent every time it's run.
- **Features Generated**:
    - `Hours`: Random value between 1 and 10.
    - `Attendance`: Random value between 60% and 100%.
    - `Assignment`: Random integer between 0 and 10.
- **Target Calculation**:
    - Formula: `Score = 10 + (5 * Hours) + (0.3 * Attendance) + (2 * Assignment) + Noise`
    - Adds random noise ("Noise") to make the data realistic and not perfectly linear.
- **Output**: Saves 1500 records to `student_scores_extended.csv`.

### 3.2 `student_scores_extended.csv`
**Purpose**: The "fuel" for our Machine Learning models.
**Format**: CSV (Comma Separated Values).
**Columns**:
1.  **Hours**: Number of hours studied daily.
2.  **Attendance**: Percentage of class attendance.
3.  **Assignment**: Marks obtained in assignments (out of 10).
4.  **Scores**: The final exam percentage (Target Variable).

### 3.3 `linearregression.py`
**Purpose**: Implements the Multiple Linear Regression algorithm from scratch.
**Class `MyLinearRegression`**:
- **`fit(X, y)`**: Trains the model using the **Normal Equation**: $\theta = (X^T X)^{-1} X^T y$. This mathematically finds the "line of best fit" without needing iterative loops (like Gradient Descent).
- **`predict(X)`**: Calculates output using $y = \theta \cdot X$.
**Execution**:
- When run directly, it loads data, splits it (80% train, 20% test), trains the model, and prints the Mean Squared Error (MSE) and R2 Score (~96%).

### 3.4 `knn.py`
**Purpose**: Implements the K-Nearest Neighbors algorithm from scratch.
**Class `MyKNNRegressor`**:
- **`__init__(k=5)`**: Initializes the model with `k=5` neighbors.
- **`fit(X, y)`**: Simply stores the training data (KNN is a "lazy learner").
- **`predict(X)`**: For every new data point:
    1.  Calculates the **Euclidean Distance** to all training points.
    2.  Sorts them and picks the closest `k` neighbors.
    3.  Averages their scores to give the final prediction.
**Execution**:
- When run directly, performs the same training/testing loop as Linear Regression. Accuracy is typically slightly lower (~93%) for this linear dataset.

### 3.5 `app.py`
**Purpose**: The interactive user interface for the project.
**Framework**: Streamlit.
**Features**:
- **Data Loading**: Caches the dataset for speed.
- **Model Training**: Retrains both Linear Regression and KNN models in real-time when the app starts.
- **Interactive Sidebar**: Displays this project report.
- **Prediction Interface**: Sliders for users to input their own Hours, Attendance, and Assignment marks.
- **Comparison**: Shows predictions from both models side-by-side for comparison.

### 3.6 `marks_prediction.py`
**Purpose**: A legacy/alternative Command-Line Interface (CLI) script.
**Functionality**:
- Performs similar duties to `app.py` but prints results to the terminal console instead of a web page.
- Useful for quick debugging or running on servers without a display.

---

## 4. Algorithms Explained

### Multiple Linear Regression
Think of this as finding a "weighted recipe".
- **Goal**: Find best weights ($w_1, w_2, w_3$) such that:
  $$ Score \approx w_1 \cdot Hours + w_2 \cdot Attendance + w_3 \cdot Assignment + Bias $$
- **Pros**: Very fast, easy to interpret, works great when relationships are simple/linear.
- **Cons**: Can't capture complex, curvy patterns well.

### K-Nearest Neighbors (KNN)
Think of this as "ask your neighbors".
- **Goal**: To predict a student's score, look at the 5 students who are most similar to them (similar hours, attendance, etc.) and average their scores.
- **Pros**: Can capture complex patterns, no math formulas to learn during training.
- **Cons**: Slower on huge datasets because it has to calculate distances to everyone.

---

## 5. How to Run

1.  **Generate Data** (Optional, if you want fresh data):
    ```bash
    python generate_data.py
    ```

2.  **Run Models Individually** (To check accuracy):
    ```bash
    python linearregression.py
    python knn.py
    ```

3.  **Run Web Application** (For the full experience):
    ```bash
    python -m streamlit run app.py
    ```
    (This will open a browser tab at `http://localhost:8501`)

---
**Author**: [Your Name/Antigravity]
**Date**: 2026-02-08
