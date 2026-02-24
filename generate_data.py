
import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# define number of students
# define number of students
n_students = 1500

# 1. Study Hours (Random between 1 and 10)
hours = np.random.uniform(1, 10, n_students)

# 2. Attendance (Random between 60% and 100%)
# There might be a slight correlation: students who study more might attend more, but let's keep it random for now or add slight bias.
attendance = np.random.uniform(60, 100, n_students)

# 3. Assignment Marks (Random between 0 and 10)
assignment = np.random.randint(0, 11, n_students)

# 4. Calculate Scores
# Let's define a known relationship + noise
# Formula: Score = 10 + (5 * Hours) + (0.3 * Attendance) + (2 * Assignment) + Noise
# Max possible approx: 10 + 50 + 30 + 20 = 110 (cap at 100)
scores = 10 + (5 * hours) + (0.3 * attendance) + (2 * assignment) + np.random.normal(0, 3, n_students)

# Clip scores to be between 0 and 100
scores = np.clip(scores, 0, 100)

# Create DataFrame
df = pd.DataFrame({
    'Hours': np.round(hours, 1),
    'Attendance': np.round(attendance, 1),
    'Assignment': assignment,
    'Scores': np.round(scores, 1)
})

# Save to CSV
output_file = 'student_scores_extended.csv'
df.to_csv(output_file, index=False)

print(f"Generated {n_students} records and saved to '{output_file}'")
print(df.head())
