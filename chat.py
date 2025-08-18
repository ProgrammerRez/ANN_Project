import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
import json

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

# --- 1. LOAD PREDICTIVE MODELS AND ENCODERS ---
@st.cache_resource
def load_prediction_assets():
    """Loads the Keras model, encoders, and scaler."""
    try:
        model = load_model('models/model.h5')
        with open('pipelines/label_encoder.pkl', 'rb') as f:
            label_encoder_gender = pickle.load(f)
        with open('pipelines/one_hot_encoding.pkl', 'rb') as f:
            one_hot_encoder = pickle.load(f)
        with open('pipelines/scaler.pkl', 'rb') as f:
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
        input_df = pd.DataFrame([input_data])
        input_df['Gender'] = label_encoder.transform(input_df['Gender'])
        geo_one_hot = one_hot_encoder.transform(input_df[['Geography']])
        geo_df = pd.DataFrame(geo_one_hot, columns=one_hot_encoder.get_feature_names_out(['Geography']))
        final_df = input_df.drop(['Geography'], axis=1)
        final_df_unordered = pd.concat([final_df, geo_df], axis=1)
        final_df = final_df_unordered[scaler.feature_names_in_]
        scaled_df = scaler.transform(final_df)
        prediction = model.predict(scaled_df)
        proba = prediction[0][0]
        
        if proba > 0.5:
            result = 'The customer is likely to **STAY** with the bank.'
        else:
            result = 'The customer is at high risk of **CHURNING** (leaving the bank).'
        return (proba, result)
    except Exception as e:
        return (None, f"An error occurred during prediction: {e}")

# --- 4. STREAMLIT APP SETUP ---
st.set_page_config(page_title="Churn Advisor Chatbot", layout="wide")
st.title("🤖 AI Customer Churn Advisor")
st.markdown("Describe a customer in plain English to predict their churn risk.")

with st.sidebar:
    st.markdown("## 💬 Chatbot Configuration")
    api_key = st.text_input("🔑 Enter your Gemini API Key:", type="password", key="api_key_input")

# --- 5. LLM AND CHATBOT LOGIC ---
history = StreamlitChatMessageHistory(key="langchain_messages")
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if api_key and all([keras_model, label_encoder_gender, one_hot_encoder, scaler]):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Using stable gemini-pro
        google_api_key=api_key,
        temperature=0.0
    )

    # --- CORRECTED and SAFER Feature Extractor Chain ---
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert data extraction and interpretation assistant. Your task is to extract customer details from the user's text and convert them into a structured format.
        The features to extract are: 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'.

        **INTERPRETATION GUIDELINES:**
        - **Prioritize Explicit Numbers:** If a user gives a specific number (e.g., "balance is 100k"), use that. Only use qualitative estimates if no specific number is given.
        - **Numerical Shorthand:** Interpret 'k' as thousands (e.g., '100k' -> 100000).
        - **Zero Values:** Interpret phrases like 'no balance', 'zero balance', or 'just joined' as the number 0.
        - **Combined Descriptors:** Unpack multiple features from single phrases (e.g., 'young German professional' implies Age, Geography, and possibly Salary).
        - **Age:** Convert qualitative descriptions to numbers (e.g., 'mid 30's' -> 35, 'late forties' -> 48, 'young professional' -> 28, 'senior' -> 65).
        - **Salary/Balance (Qualitative Estimates):** Use this scale only when no specific number is provided:
        - 'Meager', 'low', 'small', 'poor' -> ~30000
        - 'Average', 'modest', 'decent', 'normal' -> ~90000
        - 'High', 'healthy', 'large', 'wealthy' -> ~180000
        - **Tenure:** Convert text to numbers (e.g., 'a year's tenure' -> 1, 'couple of years' -> 2, 'a decade' -> 10).
        - **Booleans:** For 'HasCrCard' and 'IsActiveMember', infer 1 for yes/true ('has a credit card', 'is active') and 0 for no/false ('no card', 'inactive member').
        - **Geography:** Must be one of: 'France', 'Germany', 'Spain'.
        - **Gender:** Must be one of: 'Male', 'Female'.
        (Examples are omitted for brevity but should be included as before)

        Your output MUST be a single, valid JSON without markdown object and nothing else. **If a value for a feature is not mentioned in the user's input, you MUST use `null` for that key. The json object shold not have '''json in the start and ''' in the end**
        """),
        ("human", "{input}")
    ])
    extractor_chain = extraction_prompt | llm

# --- ENHANCED Conversational Chain for Follow-ups ---
    conversational_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert banking AI assistant and analyst.
        Your primary role is to interpret and explain customer churn predictions.
        When you are given a prediction result along with the customer's data, your first job is to present this information clearly to the user.
        Then, you must be ready to answer follow-up questions. Use the customer data you were given to explain *why* the prediction might be what it is.
        For example, if a user asks "Why is the risk high?", you should look at the data and point to factors like a high balance, low tenure, or being an inactive member.
        Be insightful, clear, and helpful.
        """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    conversational_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.7)
    conversational_chain = RunnableWithMessageHistory(
        conversational_prompt | conversational_llm,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="history",
    )

# Get user input
    user_message = st.chat_input("e.g., 'A man in his late 40s from Germany with a healthy balance and 2 products...'")

    if user_message:
        st.chat_message("human").write(user_message)
        with st.chat_message("ai"):
            with st.spinner("Analyzing customer details and predicting..."):
                try:
                    # Step 1: Extract features using the extractor chain
                    raw_json_response = extractor_chain.invoke({"input": user_message})
                    response_content = raw_json_response.content.strip()

                    # --- DEFENSIVE CHECK ---
                    # First, check if the response is empty or doesn't look like JSON
                    if not response_content or not response_content.startswith('{'):
                        # If it's not JSON, show the raw response for debugging
                        st.error("The AI returned an invalid response. It may be a refusal due to safety filters or an API issue.")
                        st.warning("Raw AI Output:")
                        st.code(response_content, language="text")
                        # Stop execution for this run
                        raise json.JSONDecodeError("Response is not a JSON object.", response_content, 0)

                    # If the check passes, try to parse it
                    extracted_data = json.loads(response_content)

                    # Step 2: Merge with fallbacks
                    final_data = DEFAULT_FALLBACKS.copy()
                    used_fallbacks, provided_details = [], {}
                    for key, value in extracted_data.items():
                        if value is not None:
                            final_data[key] = value
                            provided_details[key] = value
                        else:
                            used_fallbacks.append(key)
                    
                    # Step 3: Run prediction
                    proba, result_text = predict_churn(
                        model=keras_model, label_encoder=label_encoder_gender,
                        one_hot_encoder=one_hot_encoder, scaler=scaler, input_data=final_data
                    )
                    
                    # Step 4: Generate a conversational follow-up
                    # --- STEP 4: CREATE THE DETAILED ANALYSIS SUMMARY ---
                    if proba is not None:
                        # We create a detailed, multi-line string to pass to the conversational AI.
                        # This becomes the context for follow-up questions.
                        churn_risk_prob = 1 - proba if "STAY" in result_text else proba
                        
                        analysis_summary_input = f"""
                        I have analyzed the customer based on the provided details. Here is the churn prediction report:

                        **Prediction Result:** {result_text}
                        **Churn Risk Score:** {churn_risk_prob:.2f} (A score closer to 1.0 indicates a higher risk of churning).

                        **Final Customer Profile Used for Prediction:**
                        ```json
                        {json.dumps(final_data, indent=2)}
                        ```

                        Please present this report to the user and let them know I'm ready for any follow-up questions they might have about this analysis.
                        """

                        # --- STEP 5: INVOKE CONVERSATIONAL CHAIN WITH THE RICH CONTEXT ---
                        # The AI will now generate its response based on the detailed summary.
                        initial_response = conversational_chain.invoke(
                            {"input": analysis_summary_input},
                            config={"configurable": {"session_id": "any"}}
                        )
                        st.write(initial_response.content)
                    else:
                        # Handle prediction errors
                        st.error(result_text)

                except (json.JSONDecodeError, ValueError) as e:
                    st.error("I had trouble interpreting the AI's response. It may have been formatted incorrectly.")
                    st.warning("Please try rephrasing your query. If the problem persists, check the model's output.")
                    st.code(raw_json_response.content, language="text")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")