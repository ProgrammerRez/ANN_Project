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

data_handler = logging.FileHandler("Regressor_Training/logs/logs_classifier_chatbot_data_log.txt")
timing_handler = logging.FileHandler("Regressor_Training/logs/logs_classifier_chatbot_timing_log.txt")

data_logger.addHandler(data_handler)
timing_logger.addHandler(timing_handler)


# --- 1. LOAD PREDICTIVE MODELS AND ENCODERS ---
def load_prediction_assets():
    """Loads the Keras model, encoders, and scaler."""
    try:
        model = load_model('Regressor_Training/models/best_model.keras')
        with open('Regressor_Training/pipelines/labelencoder.pkl', 'rb') as f:
            label_encoder_gender = pickle.load(f)
        with open('Regressor_Training/pipelines/onehotencoder.pkl', 'rb') as f:
            one_hot_encoder = pickle.load(f)
        with open('Regressor_Training/pipelines/scaler.pkl', 'rb') as f:
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
    'Geography': 'France', 'Gender': 'Male','Exited':0
}


# --- 3. PREDICTION FUNCTION ---
def predict_churn(model, label_encoder, one_hot_encoder, scaler, input_data):
    """Preprocesses user input and returns the churn prediction probability and result message."""
    try:
        input_df = pd.DataFrame([input_data])
        # One-hot encode Geography
        geo_vals = one_hot_encoder.transform(input_df[['Geography']])
        geo_cols = one_hot_encoder.get_feature_names_out(['Geography'])
        geo_df = pd.DataFrame(geo_vals, columns=geo_cols)
        # Merge encoded columns
        input_df = pd.concat([input_df, geo_df], axis=1)
        input_df = input_df.drop(['Geography'], axis=1)
        # Encode Gender
        input_df['Gender'] = label_encoder.transform(input_df['Gender'])
        # Reorder columns
        input_df = input_df[scaler.feature_names_in_]
        # Scale
        input_df_scaled = scaler.transform(input_df)
        # Predict
        prediction = model.predict(input_df_scaled)
        return prediction[0][0]
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
        You are an expert banking data extraction assistant. 
        When given a customer description in natural language, extract the following details and return them as a JSON object with these rules:

        - CreditScore: integer, typical range 300–850  
        - Age: integer, in years  
        - Tenure: integer, number of years with the bank  
        - Balance: float, account balance in the local currency  
        - NumOfProducts: integer, number of bank products owned  
        - HasCrCard: 0 or 1 (integer) indicating credit card ownership  
        - IsActiveMember: 0 or 1 (integer) indicating active membership  
        - Geography: string, name of the country  
        - Gender: string, "Male" or "Female"  
        - Exited: 0 or 1 (integer), whether the customer has exited

        Return only the JSON object. Do not include explanations, text, or formatting outside of the JSON. Translate vague terms like "poor", "good", or "early 30s" into numeric values.

        """),
        ("human", "{input}")
    ])
    extractor_chain = extraction_prompt | llm

    conversational_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert banking AI assistant and analyst. 
        Provide a concise and easy-to-read summary of the prediction. 

        - Begin with a bold heading showing the predicted value, e.g., **Salary Predicted: $100,000**.  
        - Then, give a brief summary for each assessment point (CreditScore, Age, Tenure, Balance, etc.).  
        - Finally, list the key factors that influenced this prediction.  
        Keep the language simple, professional, and focused on actionable insights.  
        Do not include unnecessary explanations or unrelated text.
        """),

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

        # Logging extracted data + timing
        data_logger.info(f"[EXTRACTION INPUT] {user_message}")
        data_logger.info(f"[EXTRACTION OUTPUT] {extracted_data}")
        timing_logger.info(f"Extraction API latency: {(t1_end - t1_start).total_seconds()}s")

        # Step 2: Merge with fallbacks
        final_data = DEFAULT_FALLBACKS.copy()
        for key, value in extracted_data.items():
            if value is not None:
                final_data[key] = value

        # Step 3: Prediction
        result_text = predict_churn(
            model=keras_model,
            label_encoder=label_encoder_gender,
            one_hot_encoder=one_hot_encoder,
            scaler=scaler,
            input_data=final_data
        )

        if result_text is None:
            return result_text

        analysis_summary_input = f"""
        **Prediction Result:** {result_text}
        **Final Customer Profile Used:**
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
st.set_page_config(page_title='Salary Prediction Chatbot',
                initial_sidebar_state='expanded',
                layout='wide')

st.title('💬 Salary Prediction Chatbot')

with st.sidebar:
    st.markdown('## Configuration')
    st.info("This chatbot uses Google's Gemini LLM with churn prediction model")
    api_key = st.text_input(label='API Key', type='password')

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Describe a customer... (e.g A female in her 30's with a poor credit score and balance)"):
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