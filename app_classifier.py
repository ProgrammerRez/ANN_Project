import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

# --- Load trained model ---
model = load_model('Classifier_Training/models/model.h5')

# --- Load encoders and scaler ---
with open('Classifier_Training/pipelines/label_encoder.pkl','rb') as f:
    label_encoder_gender = pickle.load(f)

with open('Classifier_Training/pipelines/one_hot_encoding.pkl','rb') as f:
    geo_columns = pickle.load(f)   # OneHotEncoder instance

with open('Classifier_Training/pipelines/scaler.pkl','rb') as f:
    scaler = pickle.load(f)

# --- Streamlit UI setup ---
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction App")
st.markdown("### Enter customer details in the sidebar to predict if they are likely to churn.")

# Sidebar input section
st.sidebar.header("🔧 Input Features")

geo_options = [g.replace("Geography_", "") for g in geo_columns.get_feature_names_out(['Geography'])]
gender_options = label_encoder_gender.classes_

# Grouped inputs
st.sidebar.subheader("🌍 Demographics")
geography = st.sidebar.selectbox("Geography", geo_options)
gender = st.sidebar.selectbox("Gender", gender_options)
age = st.sidebar.slider("Age", 18, 100, 40)

st.sidebar.subheader("💳 Account Info")
credit_score = st.sidebar.slider("Credit Score", 300, 900, 600)
balance = st.sidebar.slider("Balance", 0, 1_000_000, 60000, step=1000)
num_products = st.sidebar.slider("Number of Products", 1, 4, 2)
estimated_salary = st.sidebar.slider("Estimated Salary", 0, 200_000, 50000, step=1000)

st.sidebar.subheader("📊 Membership Details")
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 3)
has_cr_card = st.sidebar.radio("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
is_active = st.sidebar.radio("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

# Collect input data
input_data = {
    "CreditScore": credit_score,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": estimated_salary,
    "Geography": geography
}

# Prediction function
def predict_churn(model, label_encoder, one_hot_encoder, scaler, input_data):
    input_df = pd.DataFrame([input_data])
    geo_one = one_hot_encoder.transform(input_df[["Geography"]])
    if hasattr(geo_one, "toarray"):
        geo_one = geo_one.toarray()
    geo_df = pd.DataFrame(geo_one, columns=one_hot_encoder.get_feature_names_out(["Geography"]))
    
    # Encode Gender
    input_df["Gender"] = label_encoder.transform(input_df["Gender"])
    
    # Merge final dataframe
    final_df = pd.concat([input_df.drop(["Geography"], axis=1), geo_df], axis=1)
    
    # Scale
    scaled_df = scaler.transform(final_df)
    
    # Predict probability
    prediction = model.predict(scaled_df)
    proba = prediction[0][0]
    
    if proba > 0.5:
        result = "⚠️ The customer is **likely to exit** the bank."
    else:
        result = "✅ The customer is **likely to stay** with the bank."
    
    return proba, result

# Button for prediction
if st.sidebar.button("🔮 Predict Churn"):
    proba, result = predict_churn(
        model=model,
        label_encoder=label_encoder_gender,
        one_hot_encoder=geo_columns,
        scaler=scaler,
        input_data=input_data
    )
    
    st.success("✅ Prediction Complete")
    
    # Show styled result
    st.markdown(
        f"""
        <div style="padding:20px; border-radius:15px; background-color:#f0000; text-align:center;">
            <h2>📌 Prediction Result</h2>
            <h3>{result}</h3>
            <p style="font-size:20px;">Probability of exit: <b>{proba:.2%}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("ℹ️ Adjust the sidebar inputs and click **Predict Churn** to see the result.")
