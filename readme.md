# ANN-Based Banking Analytics Project 🏦

## Overview
This project implements two machine learning applications for banking analytics:
1. Customer Churn Prediction
2. Salary Prediction

Both applications use Artificial Neural Networks (ANN) combined with Google's Gemini LLM for natural language interaction.

## Live Demos
- [Salary Prediction Chatbot](https://ann-project-chatbot-by-muhammad-umar.streamlit.app)
- [Salary Prediction App](https://ann-project-by-muhammad-umar-streamlit.app)

## Project Structure
```
ANN_Project/
├── Classifier_Training/       # Churn prediction model training
│   ├── models/               # Saved model files
│   ├── pipelines/           # Saved encoders and scalers
│   └── logs/                # Training and prediction logs
├── Regressor_Training/       # Salary prediction model training
│   ├── models/              # Saved model files
│   ├── pipelines/           # Saved encoders and scalers
│   └── logs/                # Training and prediction logs
├── Data/                     # Dataset files
├── chat_classifier.py        # Churn prediction chatbot
├── chat_regressor.py        # Salary prediction chatbot
└── requirements.txt         # Project dependencies
```

## Features
- **Natural Language Interface**: Interact with models using plain English
- **Real-time Predictions**: Get instant predictions for customer churn and salary
- **Comprehensive Analysis**: Detailed breakdown of prediction factors
- **Historical Context**: Maintains chat history for context-aware responses
- **Professional Insights**: AI-generated professional banking insights

## Model Performance
- Churn Prediction: Binary classification with >85% accuracy
- Salary Prediction: Regression with R² score of 0.98

## Technologies Used
- TensorFlow/Keras for Neural Networks
- Google's Gemini LLM for NLP
- Streamlit for Web Interface
- LangChain for LLM Integration
- Scikit-learn for Data Preprocessing
- Pandas for Data Manipulation

## Local Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ANN_Project.git
cd ANN_Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GOOGLE_API_KEY='your_api_key_here'
```

4. Run the applications:
```bash
streamlit run chat_regressor.py
streamlit run chat_classifier.py
```

## Usage
1. Visit either of the live demos
2. Enter your Google API key in the sidebar
3. Describe a customer in natural language
4. Get instant predictions and analysis

Example input:
```text
"A 35-year-old female from France with excellent credit score of 750, 
 $100,000 balance, and 5 years tenure with the bank"
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

##