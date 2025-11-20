"""
Configuration management for the Text-to-SQL application.
Loads environment variables from .env file and provides fallback to Streamlit secrets.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in the project root (parent directory of src)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

import streamlit as st


def get_config(key: str, default: str = "") -> str:
    """
    Get configuration value with the following priority:
    1. Environment variables from .env file (for local development)
    2. Streamlit secrets (for cloud deployment)
    3. Default value
    
    Args:
        key: Configuration key to retrieve
        default: Default value if key is not found
        
    Returns:
        Configuration value as string
    """
    # First check environment variables (from .env file - preferred for local)
    env_value = os.getenv(key)
    if env_value:
        return env_value
        
    # Try Streamlit secrets (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except:
        # Secrets not available, continue to default
        pass
    
    # Return default
    return default


# Export configuration values
OPENAI_API_KEY = get_config("OPENAI_API_KEY")
LLM_MODEL_NAME = get_config("LLM_MODEL_NAME", "gpt-4.1-2025-04-14")
DATABASE = get_config("DATABASE", "ecommerce")
