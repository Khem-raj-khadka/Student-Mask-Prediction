
import streamlit as st
import pandas as pd
import numpy as np
import time
from linearregression import MyLinearRegression
from knn import MyKNNRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Page Config
st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for "Themed" Look ---
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        # background-image: linear-gradient(to right bottom, #ffffff, #f0f2f6);
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    h1 {
        color: #2e86de;
        text-align: center;
        font-family: 'Helvetica', sans-serif;
    }
    h2, h3 {
        color: #34495e;
    }
    .stButton>button {
        width: 100%;
        background-color: #2e86de;
        color: white;
        border-radius: 5px;
        height: 50px;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #1b6ca8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1>🎓 Student Marks Prediction AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 1.2em;'>Predict your potential exam score based on your study habits!</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4762/4762311.png", width=100)
    st.title("Project Info")
    st.info("This AI uses **Linear Regression** and **KNN** to predict scores.")
    
    st.markdown("### 📂 Documentation")
    try:
        with open("PROJECT_REPORT.md", "r") as f:
            report_content = f.read()
        with st.expander("Read Project Report"):
            st.markdown(report_content)
    except FileNotFoundError:
        st.warning("⚠️ PROJET_REPORT.md not found.")

# --- Data Loading & Training ---
@st.cache_data
def load_and_train():
    try:
        df = pd.read_csv("student_scores_extended.csv")
        
        # Prepare Data
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Linear Regression
        lin_model = MyLinearRegression()
        lin_model.fit(X_train, y_train)
        lin_pred = lin_model.predict(X_test)
        lin_r2 = r2_score(y_test, lin_pred)

        # Train KNN
        knn_model = MyKNNRegressor(k=5)
        knn_model.fit(X_train, y_train)
        knn_pred = knn_model.predict(X_test)
        knn_r2 = r2_score(y_test, knn_pred)

        return lin_model, knn_model, lin_r2, knn_r2, y_test, lin_pred, knn_pred

    except FileNotFoundError:
        return None, None, 0, 0, None, None, None

lin_model, knn_model, lin_r2, knn_r2, y_test, lin_pred, knn_pred = load_and_train()

if lin_model is None:
    st.error("🚨 Data file not found! Please run `generate_data.py` first.")
    st.stop()

# --- Main Interface ---

# 1. Model Performance Section
st.subheader("📊 Model Performance (Accuracy)")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Linear Regression</h3>
        <h1 style="color: #27ae60;">{lin_r2*100:.2f}%</h1>
        <p>R2 Score</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>KNN Regressor</h3>
        <h1 style="color: #e67e22;">{knn_r2*100:.2f}%</h1>
        <p>R2 Score</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 2. Prediction Section
st.subheader("🔮 Make a Prediction")

# Input Column
input_col, result_col = st.columns([1, 1])

with input_col:
    st.markdown("### Enter Your Details")
    hours = st.slider("📚 Daily Study Hours", 0.0, 10.0, 5.0, help="How many hours do you study per day?")
    attendance = st.slider("📅 Attendance (%)", 60.0, 100.0, 85.0, help="Your class attendance percentage.")
    assignment = st.slider("📝 Assignment Marks (0-10)", 0, 10, 8, help="Average marks in assignments.")
    
    predict_btn = st.button("🚀 Predict Score")

# Result Column
with result_col:
    st.markdown("### Prediction Results")
    
    if predict_btn:
        with st.spinner("🤖 AI is calculating your score..."):
            time.sleep(1.5) # Artificial delay for effect
            
            # Predict
            input_data = [[hours, attendance, assignment]]
            lin_res = lin_model.predict(input_data)[0]
            knn_res = knn_model.predict(input_data)[0]
            
            # Clip results to 0-100
            lin_res = max(0, min(100, lin_res))
            knn_res = max(0, min(100, knn_res))
            
            st.balloons()
            
            # Display Results
            st.success("Prediction Complete!")
            
            tab1, tab2 = st.tabs(["Linear Regression Result", "KNN Result"])
            
            with tab1:
                st.markdown(f"<h1 style='text-align: center; color: #27ae60;'>{lin_res:.2f}%</h1>", unsafe_allow_html=True)
                st.progress(int(lin_res))
                st.caption("Based on linear relationships in the data.")
                
            with tab2:
                st.markdown(f"<h1 style='text-align: center; color: #e67e22;'>{knn_res:.2f}%</h1>", unsafe_allow_html=True)
                st.progress(int(knn_res))
                st.caption("Based on similar students' performance.")

    else:
        st.info("👈 Adjust the sliders and click 'Predict' to see your result!")
        st.image("https://cdn.dribbble.com/users/2069402/screenshots/5649479/education.gif", width=400) # Placeholder animation

# --- Comparison Chart ---
st.markdown("---")
st.warning("📉 Real-time Comparison (First 20 Test Samples)")

chart_df = pd.DataFrame({
    'Actual': y_test[:20],
    'Linear Prediction': lin_pred[:20],
    'KNN Prediction': knn_pred[:20]
})

st.line_chart(chart_df)
