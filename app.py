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
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0, step=1000.0)

input_data = {
    'CreditScore': credit_score,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active,
    'EstimatedSalary': estimated_salary,
    'Geography': geography
}

def predict_churn(model, label_encoder, one_hot_encoder, scaler, input_data):

    input_df = pd.DataFrame([input_data])
    geo_one = one_hot_encoder.transform(input_df[['Geography']])
    if hasattr(geo_one, "toarray"):
        geo_one = geo_one.toarray()
    geo_df = pd.DataFrame(geo_one, columns=one_hot_encoder.get_feature_names_out(['Geography']))
    input_df['Gender'] = label_encoder.transform(input_df['Gender'])
    final_df = pd.concat([input_df.drop(['Geography'],axis=1),geo_df],axis=1)

    scaled_df = scaler.transform(final_df)
    prediction = model.predict(scaled_df)
    proba = prediction[0][0]
    if proba > 0.5:
        result='The person will stay in the bank'
        print(result)
    else:
        result='The person will not stay in the bank'
        print(result)
    
    return (proba, result)

result = st.write(predict_churn(model=model,
                                label_encoder=label_encoder_gender,
                                one_hot_encoder=geo_columns,
                                scaler=scaler,
                                input_data=input_data))



# st.dataframe(geo_encoded_df)



# Assignment ANN :

# Your Project is to create end to end ANN Model on Churn Dataset but you have to predict the salary of customers.

# Outcome I am expecting :

# Share your live environment ANN project Deployment. 

# Deaadline : by next week (Saturday)