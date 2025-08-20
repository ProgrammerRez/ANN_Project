import streamlit as st
import app_regressor
import app_classifier

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Salary Prediction", "Balance Prediction"])

if page == "Salary Prediction":
    app_regressor.app()
elif page == "Balance Prediction":
    app_classifier.app()