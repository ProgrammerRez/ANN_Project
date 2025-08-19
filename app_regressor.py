import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

# Load trained model
model = load_model('models/model.h5')

# Load encoders and scaler
with open('pipelines/label_encoder.pkl','rb') as f:
    label_encoder_gender = pickle.load(f)

with open('pipelines/one_hot_encoding.pkl','rb') as f:
    geo_columns = pickle.load(f)   # list of dummy column names

with open('pipelines/scaler.pkl','rb') as f:
    scaler = pickle.load(f)

# --- Streamlit UI ---
st.title('🧑‍💼 Customer Churn Prediction App')

# Extract geography options (remove "Geo_" prefix)
geo_options = geo_columns.get_feature_names_out(['Geography'])
gender_options = label_encoder_gender.classes_
# Widgets for user input
geography = st.selectbox('Geography', geo_options)
gender = st.selectbox('Gender',gender_options)
age = st.slider('Age', 18, 100, 40)
tenure = st.slider('Tenure (years)', 0, 10, 3)
balance = st.number_input('Balance', min_value=0.0, value=60000.0, step=1000.0)
credit_score = st.slider('Credit Score', min_value=300, max_value=900, value=600)
num_products = st.slider('Number of Products', 1, 4, 2)
has_cr_card = st.selectbox('Has Credit Card?', [0,1])
is_active = st.selectbox('Is Active Member?', [0,1])
is_Exited = st.selectbox('Has Exited?',[0,1])


input_data = {
    'CreditScore': credit_score,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active,
    'Exited':is_Exited,
    'Geography': geography
}

def predict(input_data,model,le,one,scaler):
    input_df = pd.DataFrame([input_data])
    geo_vals = one.transform(input_df[['Geography']])
    geo_cols = one.get_feature_names_out(['Geography'])
    geo_df = pd.DataFrame(geo_vals,columns=geo_cols)
    input_df = pd.concat([input_df,geo_df],axis=1)
    input_df = input_df.drop(['Geography'],axis=1)
    input_df['Gender'] = le.transform(input_df[['Gender']])
    prediction = model.predict(input_df)
    return prediction[0][0]

result = st.title(f'Result: {predict(model=model,
                                label_encoder=label_encoder_gender,
                                one_hot_encoder=geo_columns,
                                scaler=scaler,
                                input_data=input_data)[1]}')



