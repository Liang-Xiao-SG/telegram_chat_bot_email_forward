"""
Configuration module for Telegram AI Email Forwarder Bot.

This module loads and validates environment variables, defines constants,
and provides configuration validation for the application.
"""

import os
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class EmailProvider(Enum):
    """Supported email providers with their default configurations."""
    GMAIL = ("smtp.gmail.com", 587)
    OUTLOOK = ("smtp-mail.outlook.com", 587)
    YAHOO = ("smtp.mail.yahoo.com", 587)
    
    def __init__(self, server: str, port: int):
        self.server = server
        self.port = port


class FileType(Enum):
    """Supported file types for processing."""
    PHOTO = "photo"
    DOCUMENT = "document"
    IMAGE_DOCUMENT = "image_document"
    VIDEO = "video"
    AUDIO = "audio"


class ConversationState(Enum):
    """Conversation states for multi-step interactions."""
    WAITING_EMAIL = "waiting_email"
    NORMAL = "normal"


@dataclass(frozen=True)
class TelegramLimits:
    """Telegram Bot API limits and constraints."""
    MAX_FILE_SIZE: int = 20 * 1024 * 1024  # 20MB for bots
    MAX_MESSAGE_LENGTH: int = 4096  # Official limit
    MAX_MESSAGE_LENGTH_SAFE: int = 3800  # Conservative buffer for safety
    MAX_CAPTION_LENGTH: int = 1024  # Caption length limit


@dataclass(frozen=True)
class EmailLimits:
    """Email service limits and constraints."""
    GMAIL_MAX_ATTACHMENT_SIZE: int = 25 * 1024 * 1024  # 25MB per attachment
    GMAIL_MAX_TOTAL_SIZE: int = 25 * 1024 * 1024  # 25MB total email size
    MAX_ATTACHMENTS: int = 10  # Reasonable limit for multiple attachments


@dataclass(frozen=True)
class ApplicationLimits:
    """Application-specific limits and configurations."""
    MAX_ATTACHMENTS_PER_FORWARD: int = 5  # Prevent spam
    MAX_ATTACHMENT_QUEUE_SIZE: int = 10  # Per user queue limit
    CONVERSATION_TIMEOUT_SECONDS: int = 300  # 5 minutes
    AI_RESPONSE_WORD_THRESHOLD: int = 1000  # Words threshold for file attachment
    MAX_MODELS_IN_MENU: int = 4  # Number of models to show in inline menu


@dataclass(frozen=True)
class ImageProcessingConfig:
    """Image processing configuration."""
    CONVERSION_QUALITY: int = 100  # JPEG quality 1-100
    CONVERT_IOS_FORMATS: bool = True  # Enable/disable iOS format conversion
    SUPPORTED_OUTPUT_FORMATS: List[str] = None
    
    def __post_init__(self):
        if self.SUPPORTED_OUTPUT_FORMATS is None:
            object.__setattr__(self, 'SUPPORTED_OUTPUT_FORMATS', ['JPEG', 'PNG', 'WebP'])


class AppConfig:
    """Main application configuration class."""
    
    def __init__(self):
        self._validate_required_env_vars()
        self._load_configuration()
        self._validate_configuration()
    
    def _validate_required_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        required_vars = {
            "TELEGRAM_BOT_TOKEN": "Telegram bot token from BotFather",
            "GEMINI_API_KEY": "Google AI Studio API key for Gemini"
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            if not os.getenv(var):
                missing_vars.append(f"{var} ({description})")
        
        if missing_vars:
            raise ValueError(
                f"Required environment variables not set:\n" +
                "\n".join(f"  - {var}" for var in missing_vars)
            )
    
    def _load_configuration(self) -> None:
        """Load configuration from environment variables."""
        # Core API credentials
        self.TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
        self.GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
        
        # Optional default email
        self.DEFAULT_EMAIL: Optional[str] = os.getenv("DEFAULT_EMAIL")
        
        # Gemini models configuration
        gemini_models_str = os.getenv("GEMINI_MODELS", "gemini-pro")
        self.AVAILABLE_GEMINI_MODELS: List[str] = [
            model.strip() for model in gemini_models_str.split(',')
        ]
        
        self.DEFAULT_GEMINI_MODEL: str = os.getenv(
            "DEFAULT_GEMINI_MODEL", 
            self.AVAILABLE_GEMINI_MODELS[0] if self.AVAILABLE_GEMINI_MODELS else "gemini-pro"
        )
        
        # SMTP Configuration
        self.SMTP_SERVER: Optional[str] = os.getenv("SMTP_SERVER")
        self.SMTP_USERNAME: Optional[str] = os.getenv("SMTP_USERNAME")
        self.SMTP_PASSWORD: Optional[str] = os.getenv("SMTP_PASSWORD")
        
        smtp_port_str = os.getenv("SMTP_PORT", "587")
        self.SMTP_PORT: int = int(smtp_port_str) if smtp_port_str.isdigit() else 587
        
        # Data persistence paths
        self.DATA_PATH: str = os.getenv("DATA_PATH", "/app/data/")
        self.USER_DATA_FILE: str = os.path.join(self.DATA_PATH, "user_data.json")
        
        # Image processing settings
        quality_str = os.getenv("IMAGE_CONVERSION_QUALITY", "100")
        self.IMAGE_CONVERSION_QUALITY: int = (
            int(quality_str) if quality_str.isdigit() and 1 <= int(quality_str) <= 100 else 100
        )
        
        convert_ios = os.getenv("CONVERT_IOS_FORMATS", "true").lower()
        self.CONVERT_IOS_FORMATS: bool = convert_ios == "true"
        
        # Limits and constraints
        self.telegram_limits = TelegramLimits()
        self.email_limits = EmailLimits()
        self.app_limits = ApplicationLimits()
        self.image_config = ImageProcessingConfig()
    
    def _validate_configuration(self) -> None:
        """Validate the loaded configuration."""
        # Validate default model is in available models
        if self.DEFAULT_GEMINI_MODEL not in self.AVAILABLE_GEMINI_MODELS:
            if self.AVAILABLE_GEMINI_MODELS:
                self.DEFAULT_GEMINI_MODEL = self.AVAILABLE_GEMINI_MODELS[0]
            else:
                raise ValueError(
                    "No valid Gemini models configured. Please set GEMINI_MODELS environment variable."
                )
        
        # Validate SMTP configuration if email functionality is needed
        smtp_fields = [self.SMTP_SERVER, self.SMTP_USERNAME, self.SMTP_PASSWORD]
        if any(smtp_fields) and not all(smtp_fields):
            raise ValueError(
                "Incomplete SMTP configuration. All of SMTP_SERVER, SMTP_USERNAME, "
                "and SMTP_PASSWORD must be set for email functionality."
            )
    
    @property
    def is_email_configured(self) -> bool:
        """Check if email functionality is properly configured."""
        return all([self.SMTP_SERVER, self.SMTP_USERNAME, self.SMTP_PASSWORD])
    
    @property
    def telegram_max_file_size(self) -> int:
        """Get Telegram maximum file size limit."""
        return self.telegram_limits.MAX_FILE_SIZE
    
    @property
    def gmail_max_attachment_size(self) -> int:
        """Get Gmail maximum attachment size limit."""
        return self.email_limits.GMAIL_MAX_ATTACHMENT_SIZE
    
    @property
    def max_attachments_per_forward(self) -> int:
        """Get maximum attachments per forward operation."""
        return self.app_limits.MAX_ATTACHMENTS_PER_FORWARD


# Create global configuration instance
config = AppConfig()

# Export commonly used constants for backward compatibility
TELEGRAM_BOT_TOKEN = config.TELEGRAM_BOT_TOKEN
GEMINI_API_KEY = config.GEMINI_API_KEY
DEFAULT_EMAIL = config.DEFAULT_EMAIL
AVAILABLE_GEMINI_MODELS = config.AVAILABLE_GEMINI_MODELS
DEFAULT_GEMINI_MODEL = config.DEFAULT_GEMINI_MODEL
SMTP_SERVER = config.SMTP_SERVER
SMTP_PORT = config.SMTP_PORT
SMTP_USERNAME = config.SMTP_USERNAME
SMTP_PASSWORD = config.SMTP_PASSWORD
DATA_PATH = config.DATA_PATH
USER_DATA_FILE = config.USER_DATA_FILE
IMAGE_CONVERSION_QUALITY = config.IMAGE_CONVERSION_QUALITY
CONVERT_IOS_FORMATS = config.CONVERT_IOS_FORMATS

# Legacy compatibility - these will be gradually phased out
TELEGRAM_MAX_FILE_SIZE = config.telegram_limits.MAX_FILE_SIZE
TELEGRAM_MAX_MESSAGE_LENGTH = config.telegram_limits.MAX_MESSAGE_LENGTH
TELEGRAM_MAX_MESSAGE_LENGTH_SAFE = config.telegram_limits.MAX_MESSAGE_LENGTH_SAFE
GMAIL_MAX_ATTACHMENT_SIZE = config.email_limits.GMAIL_MAX_ATTACHMENT_SIZE
GMAIL_MAX_TOTAL_SIZE = config.email_limits.GMAIL_MAX_TOTAL_SIZE
GMAIL_MAX_ATTACHMENTS = config.email_limits.MAX_ATTACHMENTS
MAX_ATTACHMENTS_PER_FORWARD = config.app_limits.MAX_ATTACHMENTS_PER_FORWARD
MAX_ATTACHMENT_QUEUE_SIZE = config.app_limits.MAX_ATTACHMENT_QUEUE_SIZE