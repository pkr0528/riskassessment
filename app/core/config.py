import os
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
from pydantic.v1 import BaseSettings


class Settings(BaseSettings):

    # Load Env file
    load_dotenv()

    # Gemini Integration
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL")
    is_gemini_available = False

    # Datasets
    TRANSACTIONAL_DATASET = os.getenv("TRANSACTIONAL_DATASET")

    # Models
    REGRESSION_MODEL_PATH = os.getenv("REGRESSION_MODEL_PATH")
    SIGNAL_CLASSIFICATION_MODEL_PATH = os.getenv("SIGNAL_CLASSIFICATION_MODEL_PATH")
    STANDARD_SCALER_MODEL_PATH = os.getenv("STANDARD_SCALER_MODEL_PATH")

    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            is_gemini_available = True
        except Exception as e:
            st.error(f"Failed to configure Gemini API: {e}")
            is_gemini_available = False
    else:
        is_gemini_available = False

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
