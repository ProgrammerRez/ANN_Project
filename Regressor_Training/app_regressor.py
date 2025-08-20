import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

# --- Load trained model ---
model = load_model('Regressor_Training/models/best_model.keras')

# --- Load encoders and scaler ---
with open('Regressor_Training/pipelines/labelencoder.pkl','rb') as f:
    label_encoder_gender = pickle.load(f)

with open('Regressor_Training/pipelines/onehotencoder.pkl','rb') as f:
    geo_columns = pickle.load(f)   # OneHotEncoder instance

with open('Regressor_Training/pipelines/scaler.pkl','rb') as f:
    scaler = pickle.load(f)

# --- Streamlit UI ---
st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="wide")

st.title("💼 Salary Prediction App")
st.markdown("### Enter employee details in the sidebar to estimate salary.")

# Sidebar input section
st.sidebar.header("🔧 Input Features")

geo_options = [g.replace("Geography_", "") for g in geo_columns.get_feature_names_out(['Geography'])]
gender_options = label_encoder_gender.classes_

# Grouped inputs
st.sidebar.subheader("🌍 Demographics")
geography = st.sidebar.selectbox('Geography', geo_options)
gender = st.sidebar.selectbox('Gender', gender_options)
age = st.sidebar.slider('Age', 18, 100, 30)

st.sidebar.subheader("💳 Account Info")
credit_score = st.sidebar.slider('Credit Score', min_value=300, max_value=900, value=650)
balance = st.sidebar.slider('Balance',min_value=0,max_value=1_000_000,value=50000,step=1000)
num_products = st.sidebar.slider('Number of Products', 1, 4, 2)

st.sidebar.subheader("📊 Employment Details")
tenure = st.sidebar.slider('Tenure (years)', 0, 10, 5)
has_cr_card = st.sidebar.radio('Has Credit Card?', [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
is_active = st.sidebar.radio('Is Active Member?', [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
is_exited = st.sidebar.radio('Has Exited?', [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

# Collect input data
input_data = {
    'CreditScore': credit_score,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': float(balance),
    'NumOfProducts': num_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active,
    'Exited': is_exited,
    'Geography': geography
}

# Prediction function
def predict(input_data, model, le, one, scaler):
    input_df = pd.DataFrame([input_data])
    # One-hot encode Geography
    geo_vals = one.transform(input_df[['Geography']])
    geo_cols = one.get_feature_names_out(['Geography'])
    geo_df = pd.DataFrame(geo_vals, columns=geo_cols)
    # Merge encoded columns
    input_df = pd.concat([input_df, geo_df], axis=1)
    input_df = input_df.drop(['Geography'], axis=1)
    # Encode Gender
    input_df['Gender'] = le.transform(input_df['Gender'])
    # Reorder columns
    input_df = input_df[scaler.feature_names_in_]
    # Scale
    input_df_scaled = scaler.transform(input_df)
    # Predict
    prediction = model.predict(input_df_scaled)
    return prediction[0][0]

# Button to predict
if st.sidebar.button("💰 Predict Salary"):
    result = predict(model=model,
                    le=label_encoder_gender,
                    one=geo_columns,
                    scaler=scaler,
                    input_data=input_data)

    st.success("✅ Prediction Complete")
    
    # Show styled result
    st.markdown(
        f"""
        <div style="padding:20px; border-radius:15px; background-color:#f1001; text-align:center;">
            <h2>📌 Predicted Salary</h2>
            <h1 style="color:#2ecc71;">${result:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("ℹ️ Adjust the sidebar inputs and click **Predict Salary** to see the result.")
