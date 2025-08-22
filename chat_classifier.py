import pickle
import pandas as pd
from tensorflow.keras.models import load_model
import json
import logging
from datetime import datetime
import streamlit as st
# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory


# --- LOGGERS SETUP ---
logging.basicConfig(level=logging.INFO)

data_logger = logging.getLogger("DataLogger")
timing_logger = logging.getLogger("TimingLogger")

data_handler = logging.FileHandler("Classifier_Training/logs/logs_classifier_chatbot_data_log.txt")
timing_handler = logging.FileHandler("Classifier_Training/logs/logs_classifier_chatbot_timing_log.txt")

data_logger.addHandler(data_handler)
timing_logger.addHandler(timing_handler)


# --- 1. LOAD PREDICTIVE MODELS AND ENCODERS ---
def load_prediction_assets():
    """Loads the Keras model, encoders, and scaler."""
    try:
        model = load_model('Classifier_Training/models/model.h5')
        with open('Classifier_Training/pipelines/label_encoder.pkl', 'rb') as f:
            label_encoder_gender = pickle.load(f)
        with open('Classifier_Training/pipelines/one_hot_encoding.pkl', 'rb') as f:
            one_hot_encoder = pickle.load(f)
        with open('Classifier_Training/pipelines/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, label_encoder_gender, one_hot_encoder, scaler
    except FileNotFoundError as e:
        st.error(f"Fatal Error: Could not load model or pipeline files: {e}. The app cannot proceed.")
        return None, None, None, None


keras_model, label_encoder_gender, one_hot_encoder, scaler = load_prediction_assets()


# --- 2. DEFINE FALLBACK FEATURES ---
DEFAULT_FALLBACKS = {
    'CreditScore': 650, 'Age': 38, 'Tenure': 5, 'Balance': 75000.0,
    'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 1,
    'EstimatedSalary': 100000.0, 'Geography': 'France', 'Gender': 'Male'
}


# --- 3. PREDICTION FUNCTION ---
def predict_churn(model, label_encoder, one_hot_encoder, scaler, input_data):
    """Preprocesses user input and returns the churn prediction probability and result message."""
    try:
        # Ensure all values are single scalars, not lists
        for key in input_data:
            if isinstance(input_data[key], list) and len(input_data[key]) > 0:
                input_data[key] = input_data[key][0]

        input_df = pd.DataFrame([input_data])

        # Ensure numeric fields are numeric
        numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Encode categorical features
        input_df['Gender'] = label_encoder.transform(input_df['Gender'])
        geo_one_hot = one_hot_encoder.transform(input_df[['Geography']])
        geo_df = pd.DataFrame(
            geo_one_hot, 
            columns=one_hot_encoder.get_feature_names_out(['Geography'])
        )

        # Merge features
        final_df = input_df.drop(['Geography'], axis=1)
        final_df_unordered = pd.concat([final_df, geo_df], axis=1)

        # Match scaler’s expected feature order
        final_df = final_df_unordered[scaler.feature_names_in_]

        # Scale
        scaled_df = scaler.transform(final_df)

        # Predict
        prediction = model.predict(scaled_df)
        proba = float(prediction[0][0])

        if proba > 0.5:
            result = 'The customer is likely to **STAY** with the bank.'
        else:
            result = 'The customer is at high risk of **CHURNING** (leaving the bank).'
        return (proba, result)

    except Exception as e:
        return (None, f"An error occurred during prediction: {e}")


# --- 4. LLM AND CHATBOT LOGIC ---
def run_chatbot(api_key, user_message):
    if not (api_key and all([keras_model, label_encoder_gender, one_hot_encoder, scaler])):
        return "Error: Missing API key or model assets."

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.0
    )

    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert data extraction assistant. Extract customer details into JSON. Capitalize first character of each key. Good, poor, mid and mediocre are estimates for high magnitude, not values—translate them to numeric. 30's means any thirty; early 30's means <35; late 30's means >35. Give exact numbers for everything numerical.
        """),
        ("human", "{input}")
    ])
    extractor_chain = extraction_prompt | llm

    conversational_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert banking AI assistant and analyst. Keep it simple and give a **bold heading** stating the **churn risk**. Then provide a concise summary for each assessment point. Be ready for follow-up questions."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    conversational_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7
    )

    history = ChatMessageHistory()
    conversational_chain = RunnableWithMessageHistory(
        conversational_prompt | conversational_llm,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="history",
    )

    try:
        # Step 1: Extract features
        t1_start = datetime.now()
        raw_json_response = extractor_chain.invoke({"input": user_message})
        t1_end = datetime.now()

        response_content = raw_json_response.content.strip()
        if response_content.startswith("```"):
            response_content = response_content.strip("`")
            if response_content.lower().startswith("json"):
                response_content = response_content[4:].strip()
            if response_content.endswith("```"):
                response_content = response_content[:-3].strip()

        try:
            extracted_data = json.loads(response_content)
        except json.JSONDecodeError:
            return "Invalid response after cleaning."

        data_logger.info(f"[PROFILE UPDATE INPUT] {user_message}")
        timing_logger.info(f"Profile update latency: {(t1_end - t1_start).total_seconds()}s")

        # Step 2: Merge with fallbacks
        final_data = DEFAULT_FALLBACKS.copy()
        for key, value in extracted_data.items():
            if value is not None:
                final_data[key] = value

        # Step 3: Prediction
        proba, result_text = predict_churn(
            model=keras_model,
            label_encoder=label_encoder_gender,
            one_hot_encoder=one_hot_encoder,
            scaler=scaler,
            input_data=final_data
        )

        if proba is None:
            return result_text

        churn_risk_prob = 1 - proba if "STAY" in result_text else proba
        analysis_summary_input = f"""
        **Prediction Result:** {result_text}
        **Churn Risk Score:** {churn_risk_prob:.2f}
        **Current Customer Profile:**
        {json.dumps(final_data, indent=2)}
        """

        # Step 4: Conversational response
        t2_start = datetime.now()
        initial_response = conversational_chain.invoke(
            {"input": analysis_summary_input},
            config={"configurable": {"session_id": "any"}}
        )
        t2_end = datetime.now()

        # Logging
        data_logger.info(f"[CONVERSATION INPUT] {analysis_summary_input}")
        data_logger.info(f"[CONVERSATION OUTPUT] {initial_response.content}")
        timing_logger.info(f"Conversation API latency: {(t2_end - t2_start).total_seconds()}s")

        return initial_response.content

    except Exception as e:
        return f"Unexpected error: {e}"


# --- STREAMLIT UI ---
st.set_page_config(page_title='Churn Prediction Chatbot',
                initial_sidebar_state='expanded',
                layout='wide')

st.title('💬 Churn Prediction Chatbot')

with st.sidebar:
    st.markdown('## Configuration')
    st.info("This chatbot uses Google's Gemini LLM with a churn prediction model")
    api_key = st.text_input(label='API Key', type='password')

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Describe a customer... (e.g. A female in her 30's with a poor credit score and balance)"):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if api_key:
        # Run chatbot
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = run_chatbot(api_key, user_input)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("⚠️ Please enter your API key in the sidebar.")
