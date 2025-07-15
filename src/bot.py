"""
Telegram AI Email Forwarder Bot - Main Application Module.

This module provides the main bot implementation with proper separation of concerns,
comprehensive error handling, and modular architecture following SOLID principles.
"""

import logging
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from contextlib import asynccontextmanager

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler as TelegramCommandHandler, MessageHandler, CallbackQueryHandler,
    ConversationHandler, ContextTypes, filters
)
from telegram.constants import ParseMode
from telegram.error import BadRequest

from src.config import config, ConversationState, FileType
from src.utils import (
    ServiceFactory, UserData, ProcessedAttachment, EmailContent,
    temporary_file_cleanup, validate_email_format, format_file_size,
    GeminiAPIError, EmailServiceError, ImageProcessingError
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants and Configuration
# ============================================================================

class BotConstants:
    """Constants used throughout the bot application."""
    
    # Emojis for user interface
    EMOJI_ROBOT = "🤖"
    EMOJI_EMAIL = "📧" 
    EMOJI_ATTACHMENT = "📎"
    EMOJI_PHOTO = "📸"
    EMOJI_DOCUMENT = "📄"
    EMOJI_SUCCESS = "✅"
    EMOJI_ERROR = "❌"
    EMOJI_WARNING = "⚠️"
    EMOJI_INFO = "ℹ️"
    EMOJI_FORWARD = "📤"
    EMOJI_FOLDER = "📁"
    
    # Message templates
    WELCOME_MESSAGE = f"""{EMOJI_ROBOT} Welcome to the AI Forwarder Bot! 

I can help you:
• Chat with AI models (Gemini)
• Forward responses to your email
• Handle document and photo attachments

Commands:
/start - Start using the bot
/help - Show detailed help
/email - Set your email address
/models - Choose AI model
/forward - Forward last response
/status - Check your settings
/clear_attachments - Clear attachment queue

Send me a message to chat with AI, or send documents/photos to queue them for forwarding."""

    ERROR_EMAIL_NOT_SET = f"{EMOJI_EMAIL} Please set your email address first using /email command."
    ERROR_NO_CONTENT = f"{EMOJI_WARNING} Nothing to forward. Chat with AI or send attachments first."
    ERROR_FILE_TOO_LARGE = f"{EMOJI_ERROR} File is too large. Maximum size: {{max_size}}."
    ERROR_QUEUE_FULL = f"{EMOJI_WARNING} Attachment queue is full. Use /clear_attachments to make space."
    ERROR_SMTP_NOT_CONFIGURED = f"{EMOJI_ERROR} Email service is not configured. Contact administrator."


# ============================================================================
# Data Classes and Types
# ============================================================================

@dataclass
class AttachmentInfo:
    """Information about a queued attachment."""
    file_id: str
    file_name: str
    file_size: int
    file_type: FileType
    needs_conversion: bool = False
    original_format: str = "Unknown"


@dataclass
class BotState:
    """Represents the current state of the bot for a user."""
    user_data: UserData
    conversation_state: ConversationState = ConversationState.NORMAL
    
    def has_email(self) -> bool:
        """Check if user has email configured."""
        return self.user_data.email is not None and validate_email_format(self.user_data.email)
    
    def has_ai_response(self) -> bool:
        """Check if user has a recent AI response."""
        return bool(self.user_data.last_ai_response)
    
    def has_attachments(self) -> bool:
        """Check if user has queued attachments."""
        return len(self.user_data.attachments_queue) > 0


# ============================================================================
# Abstract Base Classes
# ============================================================================

class BotCommandHandler(ABC):
    """Abstract base class for command handlers."""
    
    @abstractmethod
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Any:
        """Handle the command."""
        pass


class AttachmentProcessor(ABC):
    """Abstract base class for attachment processing."""
    
    @abstractmethod
    async def process(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                     attachment_info: AttachmentInfo) -> bool:
        """Process an attachment and add to queue."""
        pass


class MessageProcessor(ABC):
    """Abstract base class for message processing."""
    
    @abstractmethod
    async def process_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                            bot_state: BotState) -> bool:
        """Process a text message."""
        pass


# ============================================================================
# User Data Management
# ============================================================================

class UserDataManager:
    """Manages user data with proper encapsulation and validation."""
    
    def __init__(self):
        self._data_manager = ServiceFactory.create_data_manager()
        self._user_data_cache: Dict[int, UserData] = {}
        self._load_initial_data()
    
    def _load_initial_data(self) -> None:
        """Load initial user data from storage."""
        try:
            # Use legacy wrapper for now to maintain compatibility
            from src.utils import load_user_data
            raw_data = load_user_data()
            
            for chat_id, user_dict in raw_data.items():
                self._user_data_cache[chat_id] = UserData(
                    email=user_dict.get('email'),
                    last_ai_response=user_dict.get('last_ai_response'),
                    selected_model=user_dict.get('selected_model', config.DEFAULT_GEMINI_MODEL),
                    attachments_queue=user_dict.get('attachments_queue', [])
                )
            
            logger.info(f"Loaded user data for {len(self._user_data_cache)} users")
        except Exception as e:
            logger.error(f"Failed to load initial user data: {e}")
            self._user_data_cache = {}
    
    def get_bot_state(self, chat_id: int) -> BotState:
        """Get or create bot state for a user."""
        if chat_id not in self._user_data_cache:
            self._user_data_cache[chat_id] = UserData(
                email=config.DEFAULT_EMAIL,
                selected_model=config.DEFAULT_GEMINI_MODEL
            )
        
        return BotState(user_data=self._user_data_cache[chat_id])
    
    def update_user_data(self, chat_id: int, **kwargs) -> None:
        """Update user data fields."""
        if chat_id not in self._user_data_cache:
            self._user_data_cache[chat_id] = UserData()
        
        user_data = self._user_data_cache[chat_id]
        for key, value in kwargs.items():
            if hasattr(user_data, key):
                setattr(user_data, key, value)
        
        self._save_data()
    
    def add_attachment(self, chat_id: int, attachment_info: Dict[str, Any]) -> bool:
        """Add attachment to user's queue."""
        bot_state = self.get_bot_state(chat_id)
        
        if len(bot_state.user_data.attachments_queue) >= config.app_limits.MAX_ATTACHMENT_QUEUE_SIZE:
            return False
        
        bot_state.user_data.attachments_queue.append(attachment_info)
        self._save_data()
        return True
    
    def clear_attachments(self, chat_id: int) -> int:
        """Clear all attachments for a user and return count cleared."""
        bot_state = self.get_bot_state(chat_id)
        count = len(bot_state.user_data.attachments_queue)
        bot_state.user_data.attachments_queue.clear()
        self._save_data()
        return count
    
    def _save_data(self) -> None:
        """Save user data to storage."""
        try:
            # Convert to legacy format for compatibility
            legacy_data = {}
            for chat_id, user_data in self._user_data_cache.items():
                legacy_data[chat_id] = {
                    'email': user_data.email,
                    'last_ai_response': user_data.last_ai_response,
                    'selected_model': user_data.selected_model,
                    'attachments_queue': user_data.attachments_queue
                }
            
            from src.utils import save_user_data
            save_user_data(legacy_data)
        except Exception as e:
            logger.error(f"Failed to save user data: {e}")


# ============================================================================
# Validation Services
# ============================================================================

class FileValidator:
    """Validates files and attachments."""
    
    @staticmethod
    def validate_file_size(file_size: int, max_size: int) -> Tuple[bool, str]:
        """Validate file size against limit."""
        if file_size > max_size:
            return False, format_file_size(max_size)
        return True, ""
    
    @staticmethod
    def validate_telegram_file(file_size: int) -> Tuple[bool, str]:
        """Validate file against Telegram limits."""
        return FileValidator.validate_file_size(
            file_size, config.telegram_limits.MAX_FILE_SIZE
        )
    
    @staticmethod
    def validate_email_attachment(file_size: int) -> Tuple[bool, str]:
        """Validate file against email attachment limits."""
        return FileValidator.validate_file_size(
            file_size, config.email_limits.GMAIL_MAX_ATTACHMENT_SIZE
        )


class UserValidator:
    """Validates user input and state."""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """Validate email address format."""
        if not email or not email.strip():
            return False, "Email address cannot be empty."
        
        if not validate_email_format(email.strip()):
            return False, "Please enter a valid email address."
        
        return True, ""
    
    @staticmethod
    def validate_user_ready_for_forwarding(bot_state: BotState) -> Tuple[bool, str]:
        """Validate if user is ready for forwarding operations."""
        if not bot_state.has_email():
            return False, BotConstants.ERROR_EMAIL_NOT_SET
        
        if not config.is_email_configured:
            return False, BotConstants.ERROR_SMTP_NOT_CONFIGURED
        
        return True, ""


# ============================================================================
# Response Formatting Services
# ============================================================================

class ResponseFormatter:
    """Formats bot responses and messages."""
    
    @staticmethod
    def format_status_message(bot_state: BotState) -> str:
        """Format user status information."""
        status_parts = [
            f"{BotConstants.EMOJI_ROBOT} **Your Bot Status**\n",
            f"{BotConstants.EMOJI_EMAIL} **Email:** {bot_state.user_data.email or 'Not set'}",
            f"🧠 **AI Model:** {bot_state.user_data.selected_model}",
            f"{BotConstants.EMOJI_ATTACHMENT} **Queued Attachments:** {len(bot_state.user_data.attachments_queue)}",
        ]
        
        if bot_state.has_ai_response():
            response_preview = bot_state.user_data.last_ai_response[:50] + "..." if len(bot_state.user_data.last_ai_response) > 50 else bot_state.user_data.last_ai_response
            status_parts.append(f"💬 **Last AI Response:** {response_preview}")
        
        if bot_state.has_attachments():
            status_parts.append("\n**Attachments Queue:**")
            for i, att in enumerate(bot_state.user_data.attachments_queue[:3], 1):
                size_str = format_file_size(att.get('file_size', 0))
                status_parts.append(f"  {i}. {att.get('file_name', 'Unknown')} ({size_str})")
            
            if len(bot_state.user_data.attachments_queue) > 3:
                remaining = len(bot_state.user_data.attachments_queue) - 3
                status_parts.append(f"  ... and {remaining} more")
        
        return "\n".join(status_parts)
    
    @staticmethod
    def format_help_message() -> str:
        """Format comprehensive help message."""
        return f"""
{BotConstants.EMOJI_ROBOT} **AI Email Forwarder Bot Help**

**Basic Commands:**
• `/start` - Initialize the bot and see welcome message
• `/help` - Show this help message
• `/status` - Check your current settings and queued content

**Email Configuration:**
• `/email` - Set or update your email address
• Must be set before forwarding any content

**AI Interaction:**
• Just send a text message to chat with AI
• Responses are automatically saved for forwarding
• Use `/models` to switch between available AI models

**File Handling:**
• Send documents or photos to queue them
• Maximum file size: {format_file_size(config.telegram_limits.MAX_FILE_SIZE)}
• Maximum email attachment: {format_file_size(config.email_limits.GMAIL_MAX_ATTACHMENT_SIZE)}
• Maximum queue size: {config.app_limits.MAX_ATTACHMENT_QUEUE_SIZE} files

**Forwarding:**
• `/forward` - Forward last AI response OR all queued attachments
• Long AI responses (>{config.app_limits.AI_RESPONSE_WORD_THRESHOLD} words) sent as file attachments
• Multiple attachments sent in a single email

**Queue Management:**
• `/clear_attachments` - Remove all queued attachments
• Attachments are automatically cleared after successful forwarding

**Model Selection:**
• `/models` - Choose from available AI models
• Current options: {', '.join(config.AVAILABLE_GEMINI_MODELS[:3])}{'...' if len(config.AVAILABLE_GEMINI_MODELS) > 3 else ''}

**Limits:**
• Max attachments per forward: {config.app_limits.MAX_ATTACHMENTS_PER_FORWARD}
• Total email size limit: {format_file_size(config.email_limits.GMAIL_MAX_TOTAL_SIZE)}

**Tips:**
• Set your email first with `/email`
• Queue multiple files before forwarding
• Use `/status` to check what's ready to forward
• Long AI responses automatically become email attachments
"""
    
    @staticmethod
    def format_attachment_summary(attachments: List[Dict[str, Any]]) -> str:
        """Format attachment summary for email body."""
        if not attachments:
            return ""
        
        lines = ["Files included:"]
        total_size = 0
        
        for i, att in enumerate(attachments, 1):
            size = att.get('file_size', 0)
            total_size += size
            size_str = format_file_size(size)
            file_type = att.get('type', 'unknown')
            lines.append(f"{i}. {att.get('file_name', 'Unknown')} ({size_str}, {file_type})")
        
        lines.append(f"\nTotal size: {format_file_size(total_size)}")
        return "\n".join(lines)
    
    @staticmethod
    def generate_attachment_email_subject(attachments: List[Dict[str, Any]]) -> str:
        """Generate meaningful email subject for attachments."""
        if not attachments:
            return "Telegram Bot - Empty Attachment"
        
        count = len(attachments)
        file_types = set(att.get('type', 'file') for att in attachments)
        
        if len(file_types) == 1:
            file_type = next(iter(file_types))
            if file_type in ['photo', 'image_document']:
                return f"{BotConstants.EMOJI_PHOTO} {count} Image{'s' if count > 1 else ''} from Telegram"
            elif file_type == 'document':
                return f"{BotConstants.EMOJI_DOCUMENT} {count} Document{'s' if count > 1 else ''} from Telegram"
            else:
                return f"{BotConstants.EMOJI_ATTACHMENT} {count} File{'s' if count > 1 else ''} from Telegram"
        else:
            return f"{BotConstants.EMOJI_FOLDER} {count} Mixed Files from Telegram"


# ============================================================================
# Command Handlers Implementation
# ============================================================================

class StartCommandHandler(BotCommandHandler):
    """Handler for /start command."""
    
    def __init__(self, user_manager: UserDataManager):
        self.user_manager = user_manager
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /start command."""
        chat_id = update.effective_chat.id
        
        try:
            # Ensure user is initialized
            bot_state = self.user_manager.get_bot_state(chat_id)
            
            await update.message.reply_text(
                BotConstants.WELCOME_MESSAGE,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True
            )
            
            logger.info(f"User {chat_id} started the bot")
            
        except Exception as e:
            logger.error(f"Error in start command for user {chat_id}: {e}")
            await update.message.reply_text(
                f"{BotConstants.EMOJI_ERROR} Sorry, there was an error starting the bot."
            )


class HelpCommandHandler(BotCommandHandler):
    """Handler for /help command."""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /help command."""
        try:
            help_message = ResponseFormatter.format_help_message()
            await update.message.reply_text(
                help_message,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True
            )
        except BadRequest as e:
            # Fallback to plain text if Markdown fails
            logger.warning(f"Markdown parsing failed for help message: {e}")
            await update.message.reply_text(ResponseFormatter.format_help_message())


class StatusCommandHandler(BotCommandHandler):
    """Handler for /status command."""
    
    def __init__(self, user_manager: UserDataManager):
        self.user_manager = user_manager
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /status command."""
        chat_id = update.effective_chat.id
        
        try:
            bot_state = self.user_manager.get_bot_state(chat_id)
            status_message = ResponseFormatter.format_status_message(bot_state)
            
            await update.message.reply_text(
                status_message,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True
            )
            
        except BadRequest:
            # Fallback to plain text if Markdown fails
            await update.message.reply_text(status_message)
        except Exception as e:
            logger.error(f"Error in status command for user {chat_id}: {e}")
            await update.message.reply_text(
                f"{BotConstants.EMOJI_ERROR} Sorry, there was an error retrieving your status."
            )


class EmailCommandHandler(BotCommandHandler):
    """Handler for /email command."""
    
    def __init__(self, user_manager: UserDataManager):
        self.user_manager = user_manager
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /email command."""
        chat_id = update.effective_chat.id
        
        try:
            bot_state = self.user_manager.get_bot_state(chat_id)
            current_email = bot_state.user_data.email
            
            if current_email:
                message = (
                    f"{BotConstants.EMOJI_EMAIL} Your current email: {current_email}\n\n"
                    "Send a new email address to update it, or /cancel to keep current."
                )
            else:
                message = (
                    f"{BotConstants.EMOJI_EMAIL} Please send your email address.\n"
                    "This is required for forwarding AI responses and attachments."
                )
            
            await update.message.reply_text(message)
            return ConversationState.WAITING_EMAIL.value
            
        except Exception as e:
            logger.error(f"Error in email command for user {chat_id}: {e}")
            await update.message.reply_text(
                f"{BotConstants.EMOJI_ERROR} Sorry, there was an error with the email setup."
            )
            return ConversationHandler.END


class ModelsCommandHandler(BotCommandHandler):
    """Handler for /models command."""
    
    def __init__(self, user_manager: UserDataManager):
        self.user_manager = user_manager
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /models command."""
        chat_id = update.effective_chat.id
        
        try:
            bot_state = self.user_manager.get_bot_state(chat_id)
            current_model = bot_state.user_data.selected_model
            
            # Create inline keyboard for model selection
            keyboard = []
            models_for_menu = config.AVAILABLE_GEMINI_MODELS[:config.app_limits.MAX_MODELS_IN_MENU]
            
            for model in models_for_menu:
                is_current = "✓ " if model == current_model else ""
                keyboard.append([InlineKeyboardButton(
                    f"{is_current}{model}", 
                    callback_data=f"model_{model}"
                )])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            message = (
                f"🧠 **Current AI Model:** {current_model}\n\n"
                "Select a different model:"
            )
            
            await update.message.reply_text(
                message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
            
        except Exception as e:
            logger.error(f"Error in models command for user {chat_id}: {e}")
            await update.message.reply_text(
                f"{BotConstants.EMOJI_ERROR} Sorry, there was an error loading model options."
            )


# ============================================================================
# Message Processing Implementation
# ============================================================================

class AIMessageProcessor(MessageProcessor):
    """Processes messages for AI interaction."""
    
    def __init__(self, user_manager: UserDataManager):
        self.user_manager = user_manager
        self.gemini_service = ServiceFactory.create_gemini_service()
        self.content_analyzer = ServiceFactory.create_content_analyzer()
    
    async def process_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                            bot_state: BotState) -> bool:
        """Process a text message through AI and handle response."""
        chat_id = update.effective_chat.id
        user_message = update.message.text
        
        try:
            # Send "typing" indicator
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            
            # Generate AI response
            ai_response = await self.gemini_service.generate_text(
                user_message, 
                bot_state.user_data.selected_model
            )
            
            if not ai_response or ai_response.startswith("Error:"):
                await update.message.reply_text(
                    ai_response or f"{BotConstants.EMOJI_ERROR} No response from AI model."
                )
                return False
            
            # Store response for potential forwarding
            self.user_manager.update_user_data(chat_id, last_ai_response=ai_response)
            
            # Check if response is too long for Telegram
            return await self._handle_ai_response(update, context, ai_response)
            
        except GeminiAPIError as e:
            logger.error(f"Gemini API error for user {chat_id}: {e}")
            await update.message.reply_text(
                f"{BotConstants.EMOJI_ERROR} AI service is currently unavailable. Please try again later."
            )
            return False
        except Exception as e:
            logger.error(f"Error processing message for user {chat_id}: {e}")
            await update.message.reply_text(
                f"{BotConstants.EMOJI_ERROR} Sorry, there was an error processing your message."
            )
            return False
    
    async def _handle_ai_response(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                                ai_response: str) -> bool:
        """Handle AI response, sending as message or file based on length."""
        chat_id = update.effective_chat.id
        
        if len(ai_response) <= config.telegram_limits.MAX_MESSAGE_LENGTH_SAFE:
            # Send as regular message
            try:
                await update.message.reply_text(
                    ai_response,
                    parse_mode=ParseMode.MARKDOWN,
                    disable_notification=True
                )
                return True
            except BadRequest:
                # Fallback to plain text if Markdown fails
                await update.message.reply_text(ai_response, disable_notification=True)
                return True
        else:
            # Send as file attachment
            return await self._send_response_as_file(update, context, ai_response)
    
    async def _send_response_as_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                                   ai_response: str) -> bool:
        """Send AI response as a markdown file."""
        chat_id = update.effective_chat.id
        
        try:
            filename = self.content_analyzer.generate_filename(ai_response)
            
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.md', delete=False, encoding='utf-8') as tmp_file:
                tmp_file.write(ai_response)
                tmp_file_path = tmp_file.name
            
            with temporary_file_cleanup(tmp_file_path):
                with open(tmp_file_path, 'rb') as file:
                    await context.bot.send_document(
                        chat_id=chat_id,
                        document=file,
                        filename=filename,
                        caption=f"{BotConstants.EMOJI_DOCUMENT} AI response (too long for message)",
                        disable_notification=True
                    )
            
            await update.message.reply_text(
                f"{BotConstants.EMOJI_SUCCESS} Response sent as file. Use /forward to email it."
            )
            return True
            
        except Exception as e:
            logger.error(f"Error sending response as file for user {chat_id}: {e}")
            await update.message.reply_text(
                f"{BotConstants.EMOJI_ERROR} Error sending file. You can try /forward to get it via email."
            )
            return False


class EmailSetupProcessor:
    """Processes email setup conversation."""
    
    def __init__(self, user_manager: UserDataManager):
        self.user_manager = user_manager
    
    async def process_email(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Process email address input from user."""
        chat_id = update.effective_chat.id
        email_input = update.message.text.strip()
        
        try:
            # Validate email format
            is_valid, error_message = UserValidator.validate_email(email_input)
            
            if not is_valid:
                await update.message.reply_text(
                    f"{BotConstants.EMOJI_ERROR} {error_message}\nPlease try again:"
                )
                return ConversationState.WAITING_EMAIL.value
            
            # Save email
            self.user_manager.update_user_data(chat_id, email=email_input)
            
            await update.message.reply_text(
                f"{BotConstants.EMOJI_SUCCESS} Email address saved: {email_input}\n"
                f"You can now use /forward to send AI responses and attachments!"
            )
            
            logger.info(f"User {chat_id} set email: {email_input}")
            return ConversationHandler.END
            
        except Exception as e:
            logger.error(f"Error processing email for user {chat_id}: {e}")
            await update.message.reply_text(
                f"{BotConstants.EMOJI_ERROR} Sorry, there was an error saving your email."
            )
            return ConversationHandler.END


# ============================================================================
# Attachment Processing Implementation
# ============================================================================

class DocumentAttachmentProcessor(AttachmentProcessor):
    """Processes document attachments."""
    
    def __init__(self, user_manager: UserDataManager):
        self.user_manager = user_manager
    
    async def process(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                     attachment_info: AttachmentInfo) -> bool:
        """Process a document attachment."""
        chat_id = update.effective_chat.id
        
        try:
            bot_state = self.user_manager.get_bot_state(chat_id)
            
            # Validate file size
            if not await self._validate_attachment(update, update.message.document, chat_id):
                return False
            
            # Check queue capacity
            if len(bot_state.user_data.attachments_queue) >= config.app_limits.MAX_ATTACHMENT_QUEUE_SIZE:
                await update.message.reply_text(BotConstants.ERROR_QUEUE_FULL)
                return False
            
            # Create attachment info
            attachment_dict = {
                "file_id": update.message.document.file_id,
                "file_name": update.message.document.file_name or "document",
                "file_size": update.message.document.file_size,
                "type": FileType.DOCUMENT.value,
                "needs_conversion": False
            }
            
            # Add to queue
            success = self.user_manager.add_attachment(chat_id, attachment_dict)
            
            if success:
                queue_size = len(bot_state.user_data.attachments_queue) + 1
                await update.message.reply_text(
                    f"{BotConstants.EMOJI_SUCCESS} Document queued for forwarding.\n"
                    f"{BotConstants.EMOJI_ATTACHMENT} Queue: {queue_size}/{config.app_limits.MAX_ATTACHMENT_QUEUE_SIZE}\n"
                    f"{BotConstants.EMOJI_FORWARD} Use /forward to send all attachments to your email."
                )
                return True
            else:
                await update.message.reply_text(BotConstants.ERROR_QUEUE_FULL)
                return False
                
        except Exception as e:
            logger.error(f"Error processing document for user {chat_id}: {e}")
            await update.message.reply_text(
                f"{BotConstants.EMOJI_ERROR} Sorry, there was an error processing the document."
            )
            return False
    
    async def _validate_attachment(self, update: Update, document, chat_id: int) -> bool:
        """Validate document attachment."""
        # Validate Telegram file size
        is_valid, max_size_str = FileValidator.validate_telegram_file(document.file_size)
        if not is_valid:
            await update.message.reply_text(
                BotConstants.ERROR_FILE_TOO_LARGE.format(max_size=max_size_str)
            )
            return False
        
        # Validate email attachment size
        is_valid, max_size_str = FileValidator.validate_email_attachment(document.file_size)
        if not is_valid:
            await update.message.reply_text(
                f"{BotConstants.EMOJI_WARNING} File exceeds email attachment limit ({max_size_str}). "
                "It may not be deliverable via email."
            )
            # Continue processing but warn user
        
        return True


class PhotoAttachmentProcessor(AttachmentProcessor):
    """Processes photo attachments."""
    
    def __init__(self, user_manager: UserDataManager):
        self.user_manager = user_manager
    
    async def process(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                     attachment_info: AttachmentInfo) -> bool:
        """Process a photo attachment."""
        chat_id = update.effective_chat.id
        
        try:
            bot_state = self.user_manager.get_bot_state(chat_id)
            
            # Get largest photo size
            photo = update.message.photo[-1]
            
            # Validate file size
            if not await self._validate_photo(update, photo):
                return False
            
            # Check queue capacity
            if len(bot_state.user_data.attachments_queue) >= config.app_limits.MAX_ATTACHMENT_QUEUE_SIZE:
                await update.message.reply_text(BotConstants.ERROR_QUEUE_FULL)
                return False
            
            # Create attachment info
            attachment_dict = {
                "file_id": photo.file_id,
                "file_name": f"photo_{photo.file_unique_id}.jpg",
                "file_size": photo.file_size,
                "type": FileType.PHOTO.value,
                "needs_conversion": False,
                "original_format": "JPEG"
            }
            
            # Add to queue
            success = self.user_manager.add_attachment(chat_id, attachment_dict)
            
            if success:
                queue_size = len(bot_state.user_data.attachments_queue) + 1
                await update.message.reply_text(
                    f"{BotConstants.EMOJI_SUCCESS} Photo queued for forwarding.\n"
                    f"{BotConstants.EMOJI_ATTACHMENT} Queue: {queue_size}/{config.app_limits.MAX_ATTACHMENT_QUEUE_SIZE}\n"
                    f"{BotConstants.EMOJI_FORWARD} Use /forward to send all attachments to your email."
                )
                return True
            else:
                await update.message.reply_text(BotConstants.ERROR_QUEUE_FULL)
                return False
                
        except Exception as e:
            logger.error(f"Error processing photo for user {chat_id}: {e}")
            await update.message.reply_text(
                f"{BotConstants.EMOJI_ERROR} Sorry, there was an error processing the photo."
            )
            return False
    
    async def _validate_photo(self, update: Update, photo) -> bool:
        """Validate photo attachment."""
        if not photo.file_size:
            # Photo size unknown, allow but warn
            await update.message.reply_text(
                f"{BotConstants.EMOJI_WARNING} Photo size unknown. Proceeding with upload..."
            )
            return True
        
        # Validate Telegram file size
        is_valid, max_size_str = FileValidator.validate_telegram_file(photo.file_size)
        if not is_valid:
            await update.message.reply_text(
                BotConstants.ERROR_FILE_TOO_LARGE.format(max_size=max_size_str)
            )
            return False
        
        return True


# ============================================================================
# Forwarding Services
# ============================================================================

class ForwardingService:
    """Handles forwarding of AI responses and attachments via email."""
    
    def __init__(self, user_manager: UserDataManager):
        self.user_manager = user_manager
        self.email_service = ServiceFactory.create_email_service()
        self.image_processor = ServiceFactory.create_image_processor()
        self.content_analyzer = ServiceFactory.create_content_analyzer()
    
    async def forward_content(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Forward either AI response or attachments based on what's available."""
        chat_id = update.effective_chat.id
        
        try:
            bot_state = self.user_manager.get_bot_state(chat_id)
            
            # Validate user is ready for forwarding
            is_valid, error_message = UserValidator.validate_user_ready_for_forwarding(bot_state)
            if not is_valid:
                await update.message.reply_text(error_message)
                return False
            
            # Decide what to forward
            if bot_state.has_attachments():
                return await self._forward_attachments(update, context, bot_state)
            elif bot_state.has_ai_response():
                return await self._forward_ai_response(update, context, bot_state)
            else:
                await update.message.reply_text(BotConstants.ERROR_NO_CONTENT)
                return False
                
        except Exception as e:
            logger.error(f"Error in forward command for user {chat_id}: {e}")
            await update.message.reply_text(
                f"{BotConstants.EMOJI_ERROR} Sorry, there was an error with the forwarding process."
            )
            return False
    
    async def _forward_ai_response(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                                 bot_state: BotState) -> bool:
        """Forward AI response via email."""
        chat_id = update.effective_chat.id
        ai_response = bot_state.user_data.last_ai_response
        
        try:
            # Generate meaningful subject and email content
            subject = self.content_analyzer.generate_subject(ai_response)
            body_intro = "Here's the latest response from the AI model:\n\n"
            
            # Check if response should be sent as attachment
            word_count = len(ai_response.split())
            
            if word_count > config.app_limits.AI_RESPONSE_WORD_THRESHOLD:
                await update.message.reply_text(
                    f"📄 Response is quite long ({word_count} words). Preparing as attachment..."
                )
                
                # Create temporary file
                filename = self.content_analyzer.generate_filename(ai_response)
                
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.md', delete=False, encoding='utf-8') as tmp_file:
                    tmp_file.write(ai_response)
                    tmp_file_path = tmp_file.name
                
                with temporary_file_cleanup(tmp_file_path):
                    attachment = ProcessedAttachment(
                        path=tmp_file_path,
                        filename=filename,
                        mime_type="text/markdown",
                        size=os.path.getsize(tmp_file_path),
                        original_filename=filename
                    )
                    
                    email_content = EmailContent(
                        subject=subject,
                        body=body_intro + "The full response is attached as a Markdown file due to its length.",
                        recipient=bot_state.user_data.email
                    )
                    
                    success = self.email_service.send_email(email_content, [attachment])
            else:
                # Send as email body
                email_content = EmailContent(
                    subject=subject,
                    body=body_intro + ai_response,
                    recipient=bot_state.user_data.email
                )
                
                success = self.email_service.send_email(email_content)
            
            if success:
                await update.message.reply_text(
                    f"{BotConstants.EMOJI_SUCCESS} AI response forwarded to {bot_state.user_data.email}"
                )
                return True
            else:
                await update.message.reply_text(
                    f"{BotConstants.EMOJI_ERROR} Failed to send email. Please check configuration."
                )
                return False
                
        except EmailServiceError as e:
            logger.error(f"Email service error for user {chat_id}: {e}")
            await update.message.reply_text(
                f"{BotConstants.EMOJI_ERROR} Email service error. Please try again later."
            )
            return False
    
    async def _forward_attachments(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                                 bot_state: BotState) -> bool:
        """Forward queued attachments via email."""
        chat_id = update.effective_chat.id
        attachments_queue = bot_state.user_data.attachments_queue
        
        try:
            # Validate attachment count
            if len(attachments_queue) > config.app_limits.MAX_ATTACHMENTS_PER_FORWARD:
                await update.message.reply_text(
                    f"{BotConstants.EMOJI_ERROR} Too many attachments! "
                    f"Maximum {config.app_limits.MAX_ATTACHMENTS_PER_FORWARD} per forward."
                )
                return False
            
            # Validate total size
            total_size = sum(att.get('file_size', 0) for att in attachments_queue)
            if total_size > config.email_limits.GMAIL_MAX_TOTAL_SIZE:
                total_mb = total_size // (1024 * 1024)
                limit_mb = config.email_limits.GMAIL_MAX_TOTAL_SIZE // (1024 * 1024)
                await update.message.reply_text(
                    f"{BotConstants.EMOJI_ERROR} Total size ({total_mb}MB) exceeds email limit ({limit_mb}MB)."
                )
                return False
            
            await update.message.reply_text(
                f"{BotConstants.EMOJI_FORWARD} Preparing to forward {len(attachments_queue)} attachment(s)..."
            )
            
            # Download and process all attachments
            processed_attachments = []
            temp_files = []
            
            try:
                for attachment_info in attachments_queue:
                    processed = await self._download_and_process_attachment(context, attachment_info)
                    if processed:
                        processed_attachments.append(processed)
                        temp_files.append(processed.path)
                
                if not processed_attachments:
                    await update.message.reply_text(
                        f"{BotConstants.EMOJI_ERROR} Failed to process any attachments."
                    )
                    return False
                
                # Create email content
                subject = ResponseFormatter.generate_attachment_email_subject(attachments_queue)
                body = f"You received {len(processed_attachments)} attachment(s) from your Telegram bot.\n\n"
                body += ResponseFormatter.format_attachment_summary(attachments_queue)
                
                email_content = EmailContent(
                    subject=subject,
                    body=body,
                    recipient=bot_state.user_data.email
                )
                
                # Send email
                with temporary_file_cleanup(*temp_files):
                    success = self.email_service.send_email(email_content, processed_attachments)
                
                if success:
                    # Clear queue after successful send
                    count_cleared = self.user_manager.clear_attachments(chat_id)
                    
                    await update.message.reply_text(
                        f"{BotConstants.EMOJI_SUCCESS} {len(processed_attachments)} attachment(s) "
                        f"forwarded to {bot_state.user_data.email}\n"
                        f"📭 Attachment queue cleared."
                    )
                    return True
                else:
                    await update.message.reply_text(
                        f"{BotConstants.EMOJI_ERROR} Failed to send email with attachments."
                    )
                    return False
                    
            except Exception as e:
                # Cleanup temp files in case of error
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except:
                        pass
                raise e
                
        except EmailServiceError as e:
            logger.error(f"Email service error forwarding attachments for user {chat_id}: {e}")
            await update.message.reply_text(
                f"{BotConstants.EMOJI_ERROR} Email service error. Please try again later."
            )
            return False
    
    async def _download_and_process_attachment(self, context: ContextTypes.DEFAULT_TYPE,
                                             attachment_info: Dict[str, Any]) -> Optional[ProcessedAttachment]:
        """Download and process a single attachment."""
        try:
            # Download file
            file = await context.bot.get_file(attachment_info["file_id"])
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                await file.download_to_drive(tmp_file.name)
                original_path = tmp_file.name
            
            # Process based on type
            if attachment_info.get("type") in [FileType.PHOTO.value, "image_document"]:
                try:
                    return self.image_processor.process_attachment(
                        original_path, 
                        attachment_info["file_name"]
                    )
                except ImageProcessingError as e:
                    logger.warning(f"Image processing failed, using original: {e}")
                    # Fall through to use original file
            
            # Use file as-is
            return ProcessedAttachment(
                path=original_path,
                filename=attachment_info["file_name"],
                mime_type="application/octet-stream",
                size=attachment_info.get("file_size", 0),
                original_filename=attachment_info["file_name"]
            )
            
        except Exception as e:
            logger.error(f"Error processing attachment {attachment_info.get('file_name', 'unknown')}: {e}")
            return None


# ============================================================================
# Main Bot Application
# ============================================================================

class TelegramAIBot:
    """Main bot application with proper architecture and error handling."""
    
    def __init__(self):
        self.user_manager = UserDataManager()
        self.ai_processor = AIMessageProcessor(self.user_manager)
        self.email_processor = EmailSetupProcessor(self.user_manager)
        self.forwarding_service = ForwardingService(self.user_manager)
        self.document_processor = DocumentAttachmentProcessor(self.user_manager)
        self.photo_processor = PhotoAttachmentProcessor(self.user_manager)
        
        # Command handlers
        self.start_handler = StartCommandHandler(self.user_manager)
        self.help_handler = HelpCommandHandler()
        self.status_handler = StatusCommandHandler(self.user_manager)
        self.email_handler = EmailCommandHandler(self.user_manager)
        self.models_handler = ModelsCommandHandler(self.user_manager)
    
    def create_application(self) -> Application:
        """Create and configure the Telegram application."""
        application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        
        # Add conversation handler for email setup
        email_conv_handler = ConversationHandler(
            entry_points=[TelegramCommandHandler("email", self.email_handler.handle)],
            states={
                ConversationState.WAITING_EMAIL.value: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.email_processor.process_email)
                ],
            },
            fallbacks=[TelegramCommandHandler("cancel", self._cancel_conversation)],
            conversation_timeout=config.app_limits.CONVERSATION_TIMEOUT_SECONDS,
        )
        
        # Add all handlers
        application.add_handler(TelegramCommandHandler("start", self.start_handler.handle))
        application.add_handler(TelegramCommandHandler("help", self.help_handler.handle))
        application.add_handler(TelegramCommandHandler("status", self.status_handler.handle))
        application.add_handler(email_conv_handler)
        application.add_handler(TelegramCommandHandler("models", self.models_handler.handle))
        application.add_handler(TelegramCommandHandler("forward", self.forwarding_service.forward_content))
        application.add_handler(TelegramCommandHandler("clear_attachments", self._clear_attachments))
        
        # Model selection callback
        application.add_handler(CallbackQueryHandler(self._model_selection_callback, pattern="^model_"))
        
        # Message handlers
        application.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        application.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text_message))
        
        # Error handler
        application.add_error_handler(self._error_handler)
        
        return application
    
    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle text messages for AI interaction."""
        chat_id = update.effective_chat.id
        bot_state = self.user_manager.get_bot_state(chat_id)
        await self.ai_processor.process_message(update, context, bot_state)
    
    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle document uploads."""
        chat_id = update.effective_chat.id
        bot_state = self.user_manager.get_bot_state(chat_id)
        
        attachment_info = AttachmentInfo(
            file_id=update.message.document.file_id,
            file_name=update.message.document.file_name or "document",
            file_size=update.message.document.file_size or 0,
            file_type=FileType.DOCUMENT
        )
        
        await self.document_processor.process(update, context, attachment_info)
    
    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle photo uploads."""
        chat_id = update.effective_chat.id
        photo = update.message.photo[-1]  # Get largest size
        
        attachment_info = AttachmentInfo(
            file_id=photo.file_id,
            file_name=f"photo_{photo.file_unique_id}.jpg",
            file_size=photo.file_size or 0,
            file_type=FileType.PHOTO
        )
        
        await self.photo_processor.process(update, context, attachment_info)
    
    async def _model_selection_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle model selection from inline keyboard."""
        query = update.callback_query
        await query.answer()
        
        chat_id = update.effective_chat.id
        selected_model = query.data.replace("model_", "")
        
        try:
            if selected_model in config.AVAILABLE_GEMINI_MODELS:
                self.user_manager.update_user_data(chat_id, selected_model=selected_model)
                
                await query.edit_message_text(
                    f"{BotConstants.EMOJI_SUCCESS} AI model updated to: **{selected_model}**",
                    parse_mode=ParseMode.MARKDOWN
                )
                
                logger.info(f"User {chat_id} selected model: {selected_model}")
            else:
                await query.edit_message_text(
                    f"{BotConstants.EMOJI_ERROR} Invalid model selection."
                )
        except Exception as e:
            logger.error(f"Error in model selection for user {chat_id}: {e}")
            await query.edit_message_text(
                f"{BotConstants.EMOJI_ERROR} Error updating model selection."
            )
    
    async def _clear_attachments(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle clear attachments command."""
        chat_id = update.effective_chat.id
        
        try:
            count_cleared = self.user_manager.clear_attachments(chat_id)
            
            if count_cleared > 0:
                await update.message.reply_text(
                    f"{BotConstants.EMOJI_SUCCESS} Cleared {count_cleared} attachment(s) from queue."
                )
            else:
                await update.message.reply_text(
                    f"{BotConstants.EMOJI_INFO} No attachments to clear."
                )
        except Exception as e:
            logger.error(f"Error clearing attachments for user {chat_id}: {e}")
            await update.message.reply_text(
                f"{BotConstants.EMOJI_ERROR} Error clearing attachments."
            )
    
    async def _cancel_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Cancel current conversation."""
        await update.message.reply_text(
            f"{BotConstants.EMOJI_INFO} Operation cancelled."
        )
        return ConversationHandler.END
    
    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors in the bot."""
        logger.error(f"Exception while handling update {update}: {context.error}")
        
        if update and update.effective_chat:
            try:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"{BotConstants.EMOJI_ERROR} An unexpected error occurred. Please try again."
                )
            except Exception:
                # If we can't even send an error message, just log it
                logger.error("Failed to send error message to user")


# ============================================================================
# Application Entry Point
# ============================================================================

def main() -> None:
    """Main entry point for the bot application."""
    try:
        logger.info("Starting Telegram AI Email Forwarder Bot...")
        
        # Create bot instance
        bot = TelegramAIBot()
        
        # Create application
        application = bot.create_application()
        
        # Start bot
        logger.info("Bot is starting...")
        application.run_polling(drop_pending_updates=True)
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Critical error starting bot: {e}")
        raise


if __name__ == '__main__':
    main()