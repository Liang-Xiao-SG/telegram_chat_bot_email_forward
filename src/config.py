import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_EMAIL = os.getenv("DEFAULT_EMAIL") # This can be None

# Load available Gemini models, splitting by comma and stripping spaces
GEMINI_MODELS_STR = os.getenv("GEMINI_MODELS", "gemini-pro") # Default to gemini-pro if not set
AVAILABLE_GEMINI_MODELS = [model.strip() for model in GEMINI_MODELS_STR.split(',')]

DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-pro") # Default model

# Basic validation
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set.")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

if DEFAULT_GEMINI_MODEL not in AVAILABLE_GEMINI_MODELS:
    # If default model isn't in the list, add it or pick the first one.
    # For simplicity, let's ensure it's in the list or default to the first available.
    if AVAILABLE_GEMINI_MODELS:
        DEFAULT_GEMINI_MODEL = AVAILABLE_GEMINI_MODELS[0]
    else:
        # This case should ideally not happen if GEMINI_MODELS_STR has a default
        raise ValueError("DEFAULT_GEMINI_MODEL is not in AVAILABLE_GEMINI_MODELS and no models are available.")

# Data persistence path
DATA_PATH = "/app/data/" # Path inside the Docker container
USER_DATA_FILE = os.path.join(DATA_PATH, "user_data.json")

# SMTP Configuration for sending emails
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT_STR = os.getenv("SMTP_PORT", "587")
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_PORT = int(SMTP_PORT_STR) if SMTP_PORT_STR.isdigit() else 587 # Simplified conversion
