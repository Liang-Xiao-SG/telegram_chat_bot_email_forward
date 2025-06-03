import logging
import google.generativeai as genai
from src import config # Assuming config.py is in src directory
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os # For path joining if needed for attachments
import json

logger = logging.getLogger(__name__)

# Configure the Gemini API client
if config.GEMINI_API_KEY:
    genai.configure(api_key=config.GEMINI_API_KEY)
else:
    logger.error("Gemini API Key not found. Please set GEMINI_API_KEY in your .env file.")
    # Optionally raise an error or handle this state appropriately
    # raise ValueError("Gemini API Key not configured")

def list_gemini_models():
    """Lists available Gemini models based on configuration."""
    # For now, we are using the models defined in the .env file
    # as the primary source of truth for which models the user *wants* to use.
    # The genai.list_models() can be used for discovery but might be too broad.
    return config.AVAILABLE_GEMINI_MODELS

async def generate_text(prompt: str, model_name: str = None) -> str:
    """
    Generates text using the specified Gemini model.

    Args:
        prompt: The text prompt to send to the model.
        model_name: The name of the Gemini model to use.
                    Defaults to DEFAULT_GEMINI_MODEL from config.

    Returns:
        The generated text response from the model, or an error message.
    """
    if not config.GEMINI_API_KEY:
        return "Error: Gemini API Key is not configured."

    selected_model_name = model_name if model_name and model_name in config.AVAILABLE_GEMINI_MODELS else config.DEFAULT_GEMINI_MODEL

    if not selected_model_name:
        return "Error: No Gemini model is selected or configured."

    logger.info(f"Generating text with model: {selected_model_name}")

    try:
        model = genai.GenerativeModel(selected_model_name)
        response = await model.generate_content_async(prompt) # Using async version

        # Ensure response.text is accessed correctly
        # Based on Gemini API, the response might be more complex.
        # For simple text, response.text should work.
        # Check response.parts if it's more structured.
        if response.parts:
             # Concatenate text from all parts, assuming they are text parts
            return "".join(part.text for part in response.parts if hasattr(part, 'text'))
        elif hasattr(response, 'text') and response.text:
            return response.text
        else:
            # Fallback for cases where the response structure is unexpected
            # or if the response was blocked.
            logger.warning(f"Gemini API response for model {selected_model_name} did not contain text. Response: {response}")
            # Check for prompt feedback which might indicate blocking
            if response.prompt_feedbacks:
                for feedback in response.prompt_feedbacks:
                    logger.warning(f"Prompt Feedback: {feedback}")
                return f"Error: The request was blocked by the AI for safety reasons. Reason: {response.prompt_feedbacks[0].block_reason.name if response.prompt_feedbacks else 'Unknown'}"
            return "Error: Received an empty or unexpected response from the AI model."

    except Exception as e:
        logger.error(f"Error generating text with Gemini model {selected_model_name}: {e}")
        return f"Error: Could not connect to or get a response from the AI model. Details: {str(e)}"

def send_email(receiver_email: str, subject: str, body: str, attachment_path: str = None, attachment_filename: str = None) -> bool:
    """
    Sends an email using SMTP.
    For this to work, you'll need to configure SMTP settings.
    These are placeholders and should be configured via environment variables for production.
    """
    # Use SMTP settings from src.config
    if not all([config.SMTP_SERVER, config.SMTP_PORT, config.SMTP_USERNAME, config.SMTP_PASSWORD]):
        logger.error("SMTP server settings are not fully configured in .env. Cannot send email.")
        return False

    sender_email = config.SMTP_USERNAME

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    if attachment_path and attachment_filename:
        try:
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {attachment_filename}",
            )
            msg.attach(part)
            logger.info(f"Attached file: {attachment_filename} from path: {attachment_path}")
        except Exception as e:
            logger.error(f"Error attaching file {attachment_filename} from {attachment_path}: {e}")
            # Optionally, inform the user that attachment failed but email might still be sent
            # For now, we'll let it try to send without it or fail if crucial

    try:
        with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
            server.starttls()  # Secure the connection
            server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        logger.info(f"Email sent successfully to {receiver_email}")
        return True
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP Authentication Error: {e}. Check your SMTP_USERNAME ({config.SMTP_USERNAME}) and SMTP_PASSWORD.")
        return False
    except Exception as e:
        logger.error(f"Error sending email to {receiver_email}: {e}")
        return False

# Functions for loading and saving user data
def load_user_data() -> dict:
    """Loads user data from the JSON file specified in config."""
    # Ensure the data directory exists
    if not os.path.exists(config.DATA_PATH):
        try:
            os.makedirs(config.DATA_PATH)
            logger.info(f"Created data directory: {config.DATA_PATH}")
        except OSError as e:
            logger.error(f"Could not create data directory {config.DATA_PATH}: {e}")
            return {} # Return empty if directory creation fails

    if os.path.exists(config.USER_DATA_FILE):
        try:
            with open(config.USER_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert keys back to int if they are chat_ids
                # JSON stores all keys as strings.
                return {int(k) if k.isdigit() else k: v for k, v in data.items()}
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading user data from {config.USER_DATA_FILE}: {e}")
            # Optionally, create a backup of the corrupted file here
            return {} # Return empty dict if file is corrupted or unreadable
    return {} # Return empty dict if file doesn't exist

def save_user_data(data: dict) -> None:
    """Saves the given data to the JSON file specified in config."""
    if not os.path.exists(config.DATA_PATH):
        try:
            os.makedirs(config.DATA_PATH)
            logger.info(f"Created data directory for saving: {config.DATA_PATH}")
        except OSError as e:
            logger.error(f"Could not create data directory {config.DATA_PATH} for saving: {e}")
            return

    try:
        with open(config.USER_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"User data saved to {config.USER_DATA_FILE}")
    except IOError as e:
        logger.error(f"Error saving user data to {config.USER_DATA_FILE}: {e}")

if __name__ == '__main__':
    # Example usage (for testing utils.py directly)
    # Note: This requires .env to be correctly set up in the root directory
    # or where this script is run from, if not using a main application entry point.

    # To run this test, you might need to adjust Python's import path
    # or run as a module: python -m src.utils

    # print("Available models:", list_gemini_models())
    # import asyncio
    # async def test_generation():
    #     test_prompt = "Hello, tell me a fun fact about the Python programming language."
    #     print(f"Sending prompt: '{test_prompt}'")
    #     response = await generate_text(test_prompt)
    #     print("\nResponse from AI:")
    #     print(response)
    # asyncio.run(test_generation())
    pass
