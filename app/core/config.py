import os
import logging
from typing import Optional
from dotenv import load_dotenv
from pydantic.v1 import BaseSettings

# Load environment variables first
load_dotenv()


class Settings(BaseSettings):
    """Application settings configuration"""

    # Gemini Integration
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Datasets
    TRANSACTIONAL_DATASET: str = os.getenv(
        "TRANSACTIONAL_DATASET", "data/partner_transactional_dataset.csv"
    )

    # Model Paths
    REGRESSION_MODEL_PATH: str = os.getenv(
        "REGRESSION_MODEL_PATH", "data/regression_model.pkl"
    )
    SIGNAL_CLASSIFICATION_MODEL_PATH: str = os.getenv(
        "SIGNAL_CLASSIFICATION_MODEL_PATH", "data/signal_classification_model.pkl"
    )
    STANDARD_SCALER_MODEL_PATH: str = os.getenv(
        "STANDARD_SCALER_MODEL_PATH", "data/standard_scaler_model.pkl"
    )

    # Application settings
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Private flag to track if Gemini has been configured
    _gemini_configured: bool = False
    _gemini_config_attempted: bool = False

    @property
    def is_gemini_available(self) -> bool:
        """Check if Gemini API is available and properly configured"""
        if not self.gemini_api_key:
            return False

        # Only attempt configuration once
        if not self._gemini_config_attempted:
            self._gemini_config_attempted = True
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.gemini_api_key)
                self._gemini_configured = True
                logging.info("Gemini API configured successfully")
            except Exception as e:
                self._gemini_configured = False
                logging.warning(f"Failed to configure Gemini API: {e}")

        return self._gemini_configured

    def configure_gemini_with_streamlit(self):
        """Configure Gemini API and show Streamlit error if needed - call after Streamlit init"""
        if not self.gemini_api_key:
            import streamlit as st

            st.warning("Gemini API key not found. AI features will be disabled.")
            return False

        if not self._gemini_config_attempted:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.gemini_api_key)
                self._gemini_configured = True
                self._gemini_config_attempted = True
                return True
            except Exception as e:
                import streamlit as st

                st.error(f"Failed to configure Gemini API: {e}")
                self._gemini_configured = False
                self._gemini_config_attempted = True
                return False

        return self._gemini_configured

    class Config:
        env_file = ".env"
        extra = "allow"
        case_sensitive = False


# Create settings instance
settings = Settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
