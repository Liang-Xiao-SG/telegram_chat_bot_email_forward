"""
Utility module for Telegram AI Email Forwarder Bot.

This module provides core services for AI integration, email functionality,
image processing, and data persistence with proper error handling and
type safety.
"""

import logging
import os
import json
import tempfile
import mimetypes
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import contextmanager

import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from PIL import Image
import pillow_heif

from src.config import config, FileType

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class EmailServiceError(Exception):
    """Base exception for email service errors."""
    pass


class SMTPConfigurationError(EmailServiceError):
    """Raised when SMTP configuration is invalid."""
    pass


class ImageProcessingError(Exception):
    """Base exception for image processing errors."""
    pass


class GeminiAPIError(Exception):
    """Base exception for Gemini API errors."""
    pass


class DataPersistenceError(Exception):
    """Base exception for data persistence errors."""
    pass


# ============================================================================
# Data Classes and Types
# ============================================================================

@dataclass
class ProcessedAttachment:
    """Represents a processed attachment ready for email sending."""
    path: str
    filename: str
    mime_type: str
    size: int
    original_filename: str


@dataclass
class EmailContent:
    """Represents email content with subject and body."""
    subject: str
    body: str
    recipient: str


@dataclass
class UserData:
    """Represents user data structure with type safety."""
    email: Optional[str] = None
    last_ai_response: Optional[str] = None
    selected_model: str = config.DEFAULT_GEMINI_MODEL
    attachments_queue: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.attachments_queue is None:
            self.attachments_queue = []


# ============================================================================
# Abstract Base Classes for Services
# ============================================================================

class ImageProcessor(ABC):
    """Abstract base class for image processing operations."""
    
    @abstractmethod
    def detect_format(self, file_path: str) -> Tuple[str, str]:
        """Detect image format and return format name and MIME type."""
        pass
    
    @abstractmethod
    def convert_to_jpeg(self, input_path: str, output_path: str, quality: int) -> bool:
        """Convert image to JPEG format."""
        pass


class ContentAnalyzer(ABC):
    """Abstract base class for content analysis operations."""
    
    @abstractmethod
    def generate_subject(self, content: str) -> str:
        """Generate meaningful email subject from content."""
        pass
    
    @abstractmethod
    def generate_filename(self, content: str) -> str:
        """Generate meaningful filename from content."""
        pass


class EmailService(ABC):
    """Abstract base class for email services."""
    
    @abstractmethod
    def send_email(self, email_content: EmailContent, attachments: List[ProcessedAttachment] = None) -> bool:
        """Send email with optional attachments."""
        pass


class DataManager(ABC):
    """Abstract base class for data persistence operations."""
    
    @abstractmethod
    def load_user_data(self) -> Dict[int, UserData]:
        """Load user data from storage."""
        pass
    
    @abstractmethod
    def save_user_data(self, data: Dict[int, UserData]) -> None:
        """Save user data to storage."""
        pass


# ============================================================================
# Image Processing Implementation
# ============================================================================

class PillowImageProcessor(ImageProcessor):
    """Image processor implementation using Pillow library."""
    
    # iOS and other image format mappings
    IOS_IMAGE_FORMATS = {
        'image/heic': '.heic',
        'image/heif': '.heif',
        'image/avif': '.avif',
        'image/webp': '.webp',
    }
    
    def detect_format(self, file_path: str) -> Tuple[str, str]:
        """
        Detect image format and return format name and MIME type.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Tuple of (format_name, mime_type)
            
        Raises:
            ImageProcessingError: If format detection fails
        """
        try:
            with Image.open(file_path) as img:
                format_name = img.format or "UNKNOWN"
                mime_type = f"image/{format_name.lower()}" if format_name != "UNKNOWN" else "image/jpeg"
                
                # Handle special cases
                if format_name == 'JPEG':
                    mime_type = "image/jpeg"
                elif format_name == 'HEIF':
                    mime_type = "image/heic"
                    
                return format_name, mime_type
                
        except Exception as e:
            logger.warning(f"Could not detect image format for {file_path}: {e}")
            # Fallback to mimetypes
            mime_type, _ = mimetypes.guess_type(file_path)
            return "UNKNOWN", mime_type or "application/octet-stream"
    
    def convert_to_jpeg(self, input_path: str, output_path: str, quality: int = None) -> bool:
        """
        Convert any supported image format to JPEG.
        
        Args:
            input_path: Path to input image
            output_path: Path for output JPEG
            quality: JPEG quality (1-100), uses config default if None
            
        Returns:
            True if conversion successful, False otherwise
            
        Raises:
            ImageProcessingError: If conversion fails critically
        """
        if quality is None:
            quality = config.IMAGE_CONVERSION_QUALITY
            
        try:
            with Image.open(input_path) as img:
                # Convert to RGB mode for JPEG (removes alpha channel if present)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparent images
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as JPEG
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
                logger.info(f"Successfully converted image to JPEG: {output_path} (quality: {quality})")
                return True
                
        except Exception as e:
            logger.error(f"Error converting image to JPEG: {e}")
            raise ImageProcessingError(f"Failed to convert image: {str(e)}")
    
    def process_attachment(self, file_path: str, original_filename: str) -> ProcessedAttachment:
        """
        Process an image attachment, converting iOS formats if needed.
        
        Args:
            file_path: Path to the downloaded image file
            original_filename: Original filename from Telegram
            
        Returns:
            ProcessedAttachment object with processed file information
            
        Raises:
            ImageProcessingError: If processing fails
        """
        try:
            # Detect the actual format
            format_name, detected_mime = self.detect_format(file_path)
            logger.info(f"Detected image format: {format_name}, MIME: {detected_mime}")
            
            file_size = os.path.getsize(file_path)
            
            # Check if it's an iOS-specific format that needs conversion
            if (config.CONVERT_IOS_FORMATS and 
                (detected_mime in self.IOS_IMAGE_FORMATS or format_name in ['HEIF', 'HEIC', 'AVIF'])):
                logger.info(f"Converting iOS/unsupported format {format_name} to JPEG")
                
                # Create output path for converted image
                base_name = Path(original_filename).stem
                converted_filename = f"{base_name}_converted.jpg"
                converted_path = f"{file_path}_converted.jpg"
                
                # Convert to JPEG
                if self.convert_to_jpeg(file_path, converted_path):
                    converted_size = os.path.getsize(converted_path)
                    return ProcessedAttachment(
                        path=converted_path,
                        filename=converted_filename,
                        mime_type="image/jpeg",
                        size=converted_size,
                        original_filename=original_filename
                    )
                else:
                    logger.warning("Conversion failed, using original file")
            
            # No conversion needed or conversion failed
            return ProcessedAttachment(
                path=file_path,
                filename=original_filename,
                mime_type=detected_mime,
                size=file_size,
                original_filename=original_filename
            )
            
        except Exception as e:
            logger.error(f"Error processing image attachment: {e}")
            # Return original file if processing fails
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            return ProcessedAttachment(
                path=file_path,
                filename=original_filename,
                mime_type="image/jpeg",
                size=file_size,
                original_filename=original_filename
            )


# ============================================================================
# Content Analysis Implementation
# ============================================================================

class AIContentAnalyzer(ContentAnalyzer):
    """Content analyzer for AI responses and general text content."""
    
    # Common patterns for content analysis
    CONTENT_PATTERNS = [
        (r'^(.*?)\?', "Question: {}"),  # Questions
        (r'how to (.*?)[\.\!\?]', "How to {}"),  # How-to topics
        (r'(step\s+\d+|first|second|third|finally)', "Tutorial/Guide"),  # Tutorials
        (r'(error|problem|issue|bug)', "Troubleshooting Help"),  # Problem solving
        (r'(code|function|class|variable)', "Code Discussion"),  # Programming
        (r'(recipe|cook|ingredient)', "Recipe/Cooking"),  # Cooking
        (r'(analysis|data|report|summary)', "Analysis Report"),  # Analysis
    ]
    
    FILENAME_PATTERNS = [
        (r'how to ([\w\s]{3,25})', "how_to_{}"),
        (r'(tutorial|guide|step)', "tutorial_guide"),
        (r'(code|programming|function)', "code_help"),
        (r'(recipe|cooking|ingredient)', "recipe"),
        (r'(analysis|report|summary)', "analysis_report"),
        (r'(error|problem|troubleshoot)', "troubleshooting"),
        (r'(explain|explanation)', "explanation"),
    ]
    
    def _clean_content(self, content: str, max_chars: int = 500) -> str:
        """Clean content for analysis by removing special characters and normalizing whitespace."""
        if not content:
            return ""
        
        # Take first max_chars characters and clean
        clean_content = re.sub(r'[^\w\s]', ' ', content[:max_chars])
        return re.sub(r'\s+', ' ', clean_content).strip()
    
    def generate_subject(self, content: str) -> str:
        """
        Generate a meaningful email subject based on content.
        
        Args:
            content: The AI response or content to analyze
            
        Returns:
            A meaningful subject line
        """
        if not content or len(content.strip()) < 10:
            return "AI Response - Brief Note"
        
        clean_content = self._clean_content(content)
        
        # Try to match common patterns
        for pattern, template in self.CONTENT_PATTERNS:
            match = re.search(pattern, clean_content, re.IGNORECASE)
            if match:
                if "{}" in template:
                    topic = match.group(1).strip()[:30]  # Limit topic length
                    return template.format(topic.title())
                else:
                    return template
        
        # If no patterns match, try to get first meaningful sentence
        sentences = re.split(r'[.!?]', clean_content)
        if sentences and len(sentences[0].strip()) > 5:
            first_sentence = sentences[0].strip()[:50]  # First 50 chars
            if len(first_sentence) > 10:
                return f"AI Response: {first_sentence}"
        
        # Fallback to word analysis
        words = clean_content.split()[:8]  # First 8 words
        if len(words) >= 3:
            key_phrase = ' '.join(words[:6])
            return f"AI Response: {key_phrase.title()}"
        
        # Final fallback with timestamp
        timestamp = datetime.now().strftime("%m/%d %H:%M")
        return f"AI Response - {timestamp}"
    
    def generate_filename(self, content: str) -> str:
        """
        Generate a meaningful filename based on content.
        
        Args:
            content: The AI response content to analyze
            
        Returns:
            A meaningful filename with .md extension
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not content or len(content.strip()) < 10:
            return f"ai_brief_note_{timestamp}.md"
        
        clean_content = self._clean_content(content, max_chars=300)
        
        # Try to extract a meaningful topic
        for pattern, template in self.FILENAME_PATTERNS:
            match = re.search(pattern, clean_content, re.IGNORECASE)
            if match:
                if "{}" in template:
                    topic = match.group(1).strip()
                    # Clean topic for filename
                    topic = re.sub(r'[^\w\s]', '', topic)
                    topic = re.sub(r'\s+', '_', topic)[:20]  # Max 20 chars
                    filename = template.format(topic.lower())
                else:
                    filename = template
                
                return f"{filename}_{timestamp}.md"
        
        # If no patterns match, use first few meaningful words
        words = [w for w in clean_content.split() if len(w) > 2][:4]  # First 4 meaningful words
        if len(words) >= 2:
            topic = '_'.join(words).lower()
            topic = re.sub(r'[^\w_]', '', topic)[:25]  # Clean and limit length
            return f"ai_response_{topic}_{timestamp}.md"
        
        # Final fallback
        return f"ai_response_{timestamp}.md"


# ============================================================================
# Email Service Implementation
# ============================================================================

class SMTPEmailService(EmailService):
    """SMTP-based email service implementation."""
    
    def __init__(self):
        self._validate_smtp_config()
    
    def _validate_smtp_config(self) -> None:
        """Validate SMTP configuration."""
        if not config.is_email_configured:
            raise SMTPConfigurationError(
                "SMTP configuration incomplete. Please set SMTP_SERVER, "
                "SMTP_USERNAME, and SMTP_PASSWORD environment variables."
            )
    
    def send_email(self, email_content: EmailContent, attachments: List[ProcessedAttachment] = None) -> bool:
        """
        Send email using SMTP with optional attachments.
        
        Args:
            email_content: EmailContent object with subject, body, and recipient
            attachments: Optional list of ProcessedAttachment objects
            
        Returns:
            True if email sent successfully, False otherwise
            
        Raises:
            EmailServiceError: If email sending fails critically
        """
        try:
            msg = self._create_message(email_content)
            
            if attachments:
                self._attach_files(msg, attachments)
            
            return self._send_message(msg, email_content.recipient)
            
        except Exception as e:
            logger.error(f"Error sending email to {email_content.recipient}: {e}")
            raise EmailServiceError(f"Failed to send email: {str(e)}")
    
    def _create_message(self, email_content: EmailContent) -> MIMEMultipart:
        """Create basic email message."""
        msg = MIMEMultipart()
        msg["From"] = config.SMTP_USERNAME
        msg["To"] = email_content.recipient
        msg["Subject"] = email_content.subject
        msg.attach(MIMEText(email_content.body, "plain"))
        return msg
    
    def _attach_files(self, msg: MIMEMultipart, attachments: List[ProcessedAttachment]) -> None:
        """Attach files to email message."""
        for attachment in attachments:
            try:
                with open(attachment.path, "rb") as file:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(file.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {attachment.filename}",
                )
                msg.attach(part)
                logger.info(f"Attached file: {attachment.filename} from path: {attachment.path}")
                
            except Exception as e:
                logger.error(f"Error attaching file {attachment.filename}: {e}")
                # Continue with other attachments even if one fails
                continue
    
    def _send_message(self, msg: MIMEMultipart, recipient: str) -> bool:
        """Send the prepared message via SMTP."""
        try:
            with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
                server.starttls()  # Secure the connection
                server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
                server.sendmail(config.SMTP_USERNAME, recipient, msg.as_string())
            
            logger.info(f"Email sent successfully to {recipient}")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP Authentication Error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False


# ============================================================================
# Gemini AI Service
# ============================================================================

class GeminiAIService:
    """Service for interacting with Google's Gemini AI models."""
    
    def __init__(self):
        self._configure_api()
    
    def _configure_api(self) -> None:
        """Configure the Gemini API client."""
        if not config.GEMINI_API_KEY:
            raise GeminiAPIError("Gemini API Key not configured")
        
        genai.configure(api_key=config.GEMINI_API_KEY)
    
    def list_available_models(self) -> List[str]:
        """
        List available Gemini models based on configuration.
        
        Returns:
            List of available model names
        """
        return config.AVAILABLE_GEMINI_MODELS.copy()
    
    async def generate_text(self, prompt: str, model_name: Optional[str] = None) -> str:
        """
        Generate text using the specified Gemini model.
        
        Args:
            prompt: The text prompt to send to the model
            model_name: The name of the Gemini model to use (optional)
            
        Returns:
            The generated text response from the model
            
        Raises:
            GeminiAPIError: If API call fails or returns invalid response
        """
        if not config.GEMINI_API_KEY:
            raise GeminiAPIError("Gemini API Key is not configured")
        
        selected_model_name = self._get_valid_model_name(model_name)
        logger.info(f"Generating text with model: {selected_model_name}")
        
        try:
            model = genai.GenerativeModel(selected_model_name)
            response = await model.generate_content_async(prompt)
            
            return self._extract_text_from_response(response)
            
        except Exception as e:
            logger.error(f"Error generating text with Gemini model {selected_model_name}: {e}")
            raise GeminiAPIError(f"Failed to generate text: {str(e)}")
    
    def _get_valid_model_name(self, model_name: Optional[str]) -> str:
        """Get a valid model name, falling back to default if needed."""
        if model_name and model_name in config.AVAILABLE_GEMINI_MODELS:
            return model_name
        
        if not config.DEFAULT_GEMINI_MODEL:
            raise GeminiAPIError("No valid Gemini model configured")
        
        return config.DEFAULT_GEMINI_MODEL
    
    def _extract_text_from_response(self, response) -> str:
        """Extract text content from Gemini API response."""
        try:
            # Try to get text from response parts
            if hasattr(response, 'parts') and response.parts:
                return "".join(part.text for part in response.parts if hasattr(part, 'text'))
            
            # Try direct text access
            if hasattr(response, 'text') and response.text:
                return response.text
            
            # Check for blocked content
            if hasattr(response, 'prompt_feedbacks') and response.prompt_feedbacks:
                feedback = response.prompt_feedbacks[0]
                block_reason = getattr(feedback, 'block_reason', 'Unknown')
                logger.warning(f"Prompt blocked: {block_reason}")
                return f"Error: Content was blocked for safety reasons. Reason: {block_reason}"
            
            # Fallback for unexpected response structure
            logger.warning(f"Unexpected response structure: {response}")
            return "Error: Received an empty or unexpected response from the AI model."
            
        except Exception as e:
            logger.error(f"Error extracting text from response: {e}")
            raise GeminiAPIError(f"Failed to extract response text: {str(e)}")


# ============================================================================
# Data Persistence Implementation
# ============================================================================

class JSONDataManager(DataManager):
    """JSON-based data persistence manager."""
    
    def __init__(self, data_path: str = None, user_data_file: str = None):
        self.data_path = Path(data_path or config.DATA_PATH)
        self.user_data_file = Path(user_data_file or config.USER_DATA_FILE)
        self._ensure_data_directory()
    
    def _ensure_data_directory(self) -> None:
        """Ensure the data directory exists."""
        try:
            self.data_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Data directory ready: {self.data_path}")
        except OSError as e:
            logger.error(f"Could not create data directory {self.data_path}: {e}")
            raise DataPersistenceError(f"Failed to create data directory: {str(e)}")
    
    def load_user_data(self) -> Dict[int, UserData]:
        """
        Load user data from the JSON file.
        
        Returns:
            Dictionary mapping chat IDs to UserData objects
            
        Raises:
            DataPersistenceError: If loading fails critically
        """
        if not self.user_data_file.exists():
            logger.info("User data file does not exist, returning empty data")
            return {}
        
        try:
            with open(self.user_data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Convert raw JSON data to UserData objects
            user_data = {}
            for chat_id_str, user_dict in raw_data.items():
                try:
                    chat_id = int(chat_id_str) if chat_id_str.isdigit() else chat_id_str
                    user_data[chat_id] = UserData(**user_dict)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid user data for {chat_id_str}: {e}")
                    continue
            
            logger.info(f"Loaded user data for {len(user_data)} users")
            return user_data
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading user data from {self.user_data_file}: {e}")
            # Create backup of corrupted file
            self._backup_corrupted_file()
            return {}
    
    def save_user_data(self, data: Dict[int, UserData]) -> None:
        """
        Save user data to the JSON file.
        
        Args:
            data: Dictionary mapping chat IDs to UserData objects
            
        Raises:
            DataPersistenceError: If saving fails
        """
        try:
            # Convert UserData objects to dictionaries for JSON serialization
            serializable_data = {}
            for chat_id, user_data in data.items():
                if isinstance(user_data, UserData):
                    serializable_data[str(chat_id)] = {
                        'email': user_data.email,
                        'last_ai_response': user_data.last_ai_response,
                        'selected_model': user_data.selected_model,
                        'attachments_queue': user_data.attachments_queue
                    }
                else:
                    # Handle legacy data format
                    serializable_data[str(chat_id)] = user_data
            
            with open(self.user_data_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"User data saved to {self.user_data_file}")
            
        except (IOError, TypeError) as e:
            logger.error(f"Error saving user data to {self.user_data_file}: {e}")
            raise DataPersistenceError(f"Failed to save user data: {str(e)}")
    
    def _backup_corrupted_file(self) -> None:
        """Create a backup of corrupted user data file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.user_data_file.with_suffix(f'.backup_{timestamp}.json')
            self.user_data_file.rename(backup_path)
            logger.info(f"Corrupted user data backed up to: {backup_path}")
        except Exception as e:
            logger.error(f"Failed to backup corrupted file: {e}")


# ============================================================================
# Service Factory and Utility Functions
# ============================================================================

class ServiceFactory:
    """Factory for creating service instances with proper dependency injection."""
    
    @staticmethod
    def create_image_processor() -> ImageProcessor:
        """Create an image processor instance."""
        return PillowImageProcessor()
    
    @staticmethod
    def create_content_analyzer() -> ContentAnalyzer:
        """Create a content analyzer instance."""
        return AIContentAnalyzer()
    
    @staticmethod
    def create_email_service() -> EmailService:
        """Create an email service instance."""
        return SMTPEmailService()
    
    @staticmethod
    def create_data_manager() -> DataManager:
        """Create a data manager instance."""
        return JSONDataManager()
    
    @staticmethod
    def create_gemini_service() -> GeminiAIService:
        """Create a Gemini AI service instance."""
        return GeminiAIService()


# ============================================================================
# Context Managers and Utility Functions
# ============================================================================

@contextmanager
def temporary_file_cleanup(*file_paths: str):
    """Context manager for automatic cleanup of temporary files."""
    try:
        yield
    finally:
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")


def validate_email_format(email: str) -> bool:
    """
    Validate email format using basic regex.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email format is valid, False otherwise
    """
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


# ============================================================================
# Legacy Compatibility Functions
# ============================================================================

# Create global service instances for backward compatibility
_image_processor = ServiceFactory.create_image_processor()
_content_analyzer = ServiceFactory.create_content_analyzer()
_email_service = ServiceFactory.create_email_service()
_data_manager = ServiceFactory.create_data_manager()
_gemini_service = ServiceFactory.create_gemini_service()


# Legacy function wrappers for backward compatibility
def detect_image_format(file_path: str) -> Tuple[str, str]:
    """Legacy wrapper for image format detection."""
    return _image_processor.detect_format(file_path)


def convert_image_to_jpeg(input_path: str, output_path: str, quality: int = None) -> bool:
    """Legacy wrapper for image conversion."""
    return _image_processor.convert_to_jpeg(input_path, output_path, quality or config.IMAGE_CONVERSION_QUALITY)


def process_image_attachment(file_path: str, original_filename: str) -> Tuple[str, str, str]:
    """Legacy wrapper for image attachment processing."""
    processed = _image_processor.process_attachment(file_path, original_filename)
    return processed.path, processed.filename, processed.mime_type


def generate_meaningful_subject(content: str, content_type: str = "ai_response") -> str:
    """Legacy wrapper for subject generation."""
    if content_type == "attachments":
        return content
    return _content_analyzer.generate_subject(content)


def generate_meaningful_filename(content: str, original_name: str = "ai_response.md") -> str:
    """Legacy wrapper for filename generation."""
    return _content_analyzer.generate_filename(content)


async def generate_text(prompt: str, model_name: str = None) -> str:
    """Legacy wrapper for text generation."""
    try:
        return await _gemini_service.generate_text(prompt, model_name)
    except GeminiAPIError as e:
        return f"Error: {str(e)}"


def list_gemini_models() -> List[str]:
    """Legacy wrapper for listing Gemini models."""
    return _gemini_service.list_available_models()


def send_email(receiver_email: str, subject: str, body: str, 
               attachment_path: str = None, attachment_filename: str = None) -> bool:
    """Legacy wrapper for sending single email."""
    try:
        email_content = EmailContent(subject=subject, body=body, recipient=receiver_email)
        attachments = []
        
        if attachment_path and attachment_filename:
            size = os.path.getsize(attachment_path) if os.path.exists(attachment_path) else 0
            attachments.append(ProcessedAttachment(
                path=attachment_path,
                filename=attachment_filename,
                mime_type="application/octet-stream",
                size=size,
                original_filename=attachment_filename
            ))
        
        return _email_service.send_email(email_content, attachments if attachments else None)
    except EmailServiceError:
        return False


def send_multiple_attachments_email(receiver_email: str, subject: str, body: str, attachments: List[Dict]) -> bool:
    """Legacy wrapper for sending email with multiple attachments."""
    try:
        email_content = EmailContent(subject=subject, body=body, recipient=receiver_email)
        processed_attachments = []
        
        for attachment in attachments:
            size = os.path.getsize(attachment["path"]) if os.path.exists(attachment["path"]) else 0
            processed_attachments.append(ProcessedAttachment(
                path=attachment["path"],
                filename=attachment["filename"],
                mime_type="application/octet-stream",
                size=size,
                original_filename=attachment["filename"]
            ))
        
        return _email_service.send_email(email_content, processed_attachments)
    except EmailServiceError:
        return False


def load_user_data() -> Dict:
    """Legacy wrapper for loading user data."""
    try:
        data = _data_manager.load_user_data()
        # Convert UserData objects back to dictionaries for legacy compatibility
        legacy_data = {}
        for chat_id, user_data in data.items():
            if isinstance(user_data, UserData):
                legacy_data[chat_id] = {
                    'email': user_data.email,
                    'last_ai_response': user_data.last_ai_response,
                    'selected_model': user_data.selected_model,
                    'attachments_queue': user_data.attachments_queue
                }
            else:
                legacy_data[chat_id] = user_data
        return legacy_data
    except DataPersistenceError:
        return {}


def save_user_data(data: Dict) -> None:
    """Legacy wrapper for saving user data."""
    try:
        # Convert legacy data format to UserData objects
        typed_data = {}
        for chat_id, user_dict in data.items():
            if isinstance(user_dict, dict):
                typed_data[chat_id] = UserData(
                    email=user_dict.get('email'),
                    last_ai_response=user_dict.get('last_ai_response'),
                    selected_model=user_dict.get('selected_model', config.DEFAULT_GEMINI_MODEL),
                    attachments_queue=user_dict.get('attachments_queue', [])
                )
            else:
                typed_data[chat_id] = user_dict
        
        _data_manager.save_user_data(typed_data)
    except DataPersistenceError:
        logger.error("Failed to save user data")


if __name__ == '__main__':
    # Example usage and testing
    pass