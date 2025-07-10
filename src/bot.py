import logging
import os
import tempfile  # For handling large attachments
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler,
    ConversationHandler,
)
from telegram.constants import ParseMode

# Assuming your config and utils are structured in src
from src import config  # Loads .env variables
from src import utils  # For generate_text and send_email

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states for /set_email
AWAITING_EMAIL_REPLY = 1
CONVERSATION_TIMEOUT_SECONDS = 300  # 5 minutes

# Load user data at startup
user_data = utils.load_user_data()
logger.info(f"Loaded user data: {len(user_data)} users.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    user = update.effective_user
    chat_id = update.effective_chat.id

    # Initialize user data if not present
    if chat_id not in user_data:
        user_data[chat_id] = {
            "email": config.DEFAULT_EMAIL,  # Load default from config
            "last_ai_response": None,
            "selected_model": config.DEFAULT_GEMINI_MODEL,
            "attachments_queue": [],  # List to store multiple attachments
        }
        utils.save_user_data(user_data)  # Save after initializing new user
        logger.info(f"Initialized data for new user: {chat_id}. Data saved.")

    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
    )
    await update.message.reply_text(
        f"""Welcome to the AI Forwarder Bot! I can help you chat with an AI model and forward the responses to your email address.

Your current default email is: {user_data[chat_id].get('email') or 'Not set'}
Your current AI model is: {user_data[chat_id].get('selected_model')}

Use /help to see all commands."""
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a help message when the /help command is issued."""
    chat_id = update.effective_chat.id
    current_email = user_data.get(chat_id, {}).get("email") or "Not set"
    current_model = user_data.get(chat_id, {}).get(
        "selected_model", config.DEFAULT_GEMINI_MODEL
    )

    available_models_text = ", ".join(config.AVAILABLE_GEMINI_MODELS)
    if not available_models_text:
        available_models_text = "No models configured (check .env)."

    help_text = f"""Here's how I can help you:

Your current default email: **{current_email}**
Your current AI model: **{current_model}**

**Chatting with AI:**
Simply send me any message, and I'll pass it to the AI. I'll try to render the AI's response using Markdown formatting directly in the chat.
*(If an AI response is too long for a direct message, I'll send it as an `.md` (Markdown) file instead.)*

**File Attachments:**
Send me documents or photos and I'll queue them for forwarding. You can send multiple files and forward them all at once.

**Commands:**
/start - Welcome message & current settings.
/help - Show this help message.
/set_email [email_address] - Set/update your default email.
  If no email is given, I will ask for it.
  Example: `/set_email user@example.com`
/cancel_set_email - Cancel the ongoing email setting process.
/forward - Forward the last AI response OR all queued attachments to your email.
/list_attachments - Show current attachments in queue.
/clear_attachments - Clear all attachments from queue.
/switch_model - Show a menu to quickly switch between common AI models.
/set_model <model_name> - Set a specific AI model by name. Available: {available_models_text}
  Example: `/set_model {config.AVAILABLE_GEMINI_MODELS[0] if config.AVAILABLE_GEMINI_MODELS else 'gemini-pro'}`

**File Limits:**
• Max file size: {config.TELEGRAM_MAX_FILE_SIZE // (1024*1024)}MB per file
• Max total email size: {config.GMAIL_MAX_TOTAL_SIZE // (1024*1024)}MB
• Max attachments per forward: {config.MAX_ATTACHMENTS_PER_FORWARD}
• Max queue size: {config.MAX_ATTACHMENT_QUEUE_SIZE} files

**Note:** Your email and model preferences are saved automatically."""
    await update.message.reply_text(help_text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles regular messages by sending them to the AI and returning the response."""
    chat_id = update.effective_chat.id
    user_message = update.message.text

    if chat_id not in user_data:  # Should be initialized by /start, but as a fallback
        user_data[chat_id] = {
            "email": config.DEFAULT_EMAIL,
            "last_ai_response": None,
            "selected_model": config.DEFAULT_GEMINI_MODEL,
            "attachments_queue": [],
        }
        # Save immediately if a new user entry is created this way
        utils.save_user_data(user_data)
        logger.info(
            f"Initialized and saved data for new user via handle_message: {chat_id}"
        )

    selected_model = user_data[chat_id].get(
        "selected_model", config.DEFAULT_GEMINI_MODEL
    )

    # Let user know the bot is working on it
    processing_message = await update.message.reply_text(
        f"🤖 Thinking with {selected_model}...", disable_notification=True
    )

    ai_response = await utils.generate_text(
        prompt=user_message, model_name=selected_model
    )

    # Delete the "Thinking..." message
    if processing_message:
        try:
            await context.bot.delete_message(
                chat_id=chat_id, message_id=processing_message.message_id
            )
        except Exception as e:
            logger.warning(f"Could not delete 'Thinking...' message: {e}")

    if ai_response and not ai_response.startswith("Error:"):
        user_data[chat_id]["last_ai_response"] = ai_response
        user_data[chat_id]["attachments_queue"] = []  # Clear attachments when new AI response is received
        utils.save_user_data(user_data)  # Save data after getting a valid AI response
        logger.info(f"Stored AI response for chat_id {chat_id} and saved data.")
    elif not ai_response:  # Empty response but not an explicit "Error:"
        logger.error(f"AI response was empty for chat_id {chat_id}. Not storing.")
        ai_response = "Sorry, I received an empty response from the AI."
        # Do not store this as last_ai_response
    # If ai_response starts with "Error:", it will be sent as is.

    # Handle sending the response (potentially long)
    if len(ai_response) > config.TELEGRAM_MAX_MESSAGE_LENGTH_SAFE:
        logger.info(
            f"AI response for chat_id {chat_id} is too long ({len(ai_response)} chars). Sending as a file."
        )
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+", delete=False, suffix=".md", encoding="utf-8"
            ) as tmp_file:
                tmp_file.write(ai_response)
                attachment_path = tmp_file.name

            caption_text = "The AI's response was too long to display directly in the chat, so it's attached as a text file."
            await context.bot.send_document(
                chat_id=chat_id,
                document=open(attachment_path, "rb"),
                filename="ai_response.md",
                caption=caption_text,
            )
            os.remove(attachment_path)  # Clean up the temp file
            logger.info(f"Sent long AI response as a file to chat_id {chat_id}.")
        except Exception as e:
            logger.error(f"Error sending long AI response as file to {chat_id}: {e}")
            await update.message.reply_text(
                "There was an error sending the AI's long response as a file. You can try using /forward to get it via email."
            )
    else:
        is_error_or_generic_response = (
            ai_response.startswith("Error:")
            or ai_response == "Sorry, I received an empty response from the AI."
        )
        if is_error_or_generic_response:
            await update.message.reply_text(
                ai_response
            )  # Send errors/generic as plain text
        else:
            try:
                await update.message.reply_text(
                    ai_response, parse_mode=ParseMode.MARKDOWN_V2
                )
            except Exception as e:  # Catch potential parsing errors from MarkdownV2
                logger.warning(
                    f"Failed to send message with MarkdownV2 for chat_id {chat_id}: {e}. Sending as plain text."
                )
                await update.message.reply_text(ai_response)  # Fallback to plain text


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles document attachments by storing them for potential forwarding."""
    chat_id = update.effective_chat.id
    document = update.message.document
    
    # File size validation
    if document.file_size and document.file_size > config.TELEGRAM_MAX_FILE_SIZE:
        await update.message.reply_text(
            f"❌ Document too large! Telegram limit is {config.TELEGRAM_MAX_FILE_SIZE // (1024*1024)}MB. "
            f"Your file is {document.file_size // (1024*1024)}MB."
        )
        return
    
    if document.file_size and document.file_size > config.GMAIL_MAX_ATTACHMENT_SIZE:
        await update.message.reply_text(
            f"❌ Document too large for email! Gmail limit is {config.GMAIL_MAX_ATTACHMENT_SIZE // (1024*1024)}MB. "
            f"Your file is {document.file_size // (1024*1024)}MB."
        )
        return
    
    if chat_id not in user_data:
        user_data[chat_id] = {
            "email": config.DEFAULT_EMAIL,
            "last_ai_response": None,
            "selected_model": config.DEFAULT_GEMINI_MODEL,
            "attachments_queue": [],
        }
    
    # Check if queue is full
    if len(user_data[chat_id]["attachments_queue"]) >= config.MAX_ATTACHMENT_QUEUE_SIZE:
        await update.message.reply_text(
            f"❌ Attachment queue is full! Maximum {config.MAX_ATTACHMENT_QUEUE_SIZE} attachments allowed. "
            f"Use /forward to send current attachments or /clear_attachments to clear the queue."
        )
        return
    
    # Store document information
    attachment_info = {
        "type": "document",
        "file_id": document.file_id,
        "file_name": document.file_name or "document",
        "mime_type": document.mime_type or "application/octet-stream",
        "file_size": document.file_size or 0,
        "timestamp": update.message.date.isoformat(),
    }
    
    user_data[chat_id]["attachments_queue"].append(attachment_info)
    user_data[chat_id]["last_ai_response"] = None  # Clear AI response when attachment is received
    
    utils.save_user_data(user_data)
    logger.info(f"Stored document attachment for chat_id {chat_id}: {document.file_name}")
    
    queue_count = len(user_data[chat_id]["attachments_queue"])
    await update.message.reply_text(
        f"📎 Document received: {document.file_name}\n"
        f"📊 Attachments in queue: {queue_count}/{config.MAX_ATTACHMENT_QUEUE_SIZE}\n"
        f"Use /forward to send all attachments to your email address."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles photo attachments by storing them for potential forwarding."""
    chat_id = update.effective_chat.id
    photo = update.message.photo[-1]  # Get the highest resolution photo
    
    # File size validation
    if photo.file_size and photo.file_size > config.TELEGRAM_MAX_FILE_SIZE:
        await update.message.reply_text(
            f"❌ Photo too large! Telegram limit is {config.TELEGRAM_MAX_FILE_SIZE // (1024*1024)}MB."
        )
        return
    
    if photo.file_size and photo.file_size > config.GMAIL_MAX_ATTACHMENT_SIZE:
        await update.message.reply_text(
            f"❌ Photo too large for email! Gmail limit is {config.GMAIL_MAX_ATTACHMENT_SIZE // (1024*1024)}MB."
        )
        return
    
    if chat_id not in user_data:
        user_data[chat_id] = {
            "email": config.DEFAULT_EMAIL,
            "last_ai_response": None,
            "selected_model": config.DEFAULT_GEMINI_MODEL,
            "attachments_queue": [],
        }
    
    # Check if queue is full
    if len(user_data[chat_id]["attachments_queue"]) >= config.MAX_ATTACHMENT_QUEUE_SIZE:
        await update.message.reply_text(
            f"❌ Attachment queue is full! Maximum {config.MAX_ATTACHMENT_QUEUE_SIZE} attachments allowed. "
            f"Use /forward to send current attachments or /clear_attachments to clear the queue."
        )
        return
    
    # Store photo information
    attachment_info = {
        "type": "photo",
        "file_id": photo.file_id,
        "file_name": f"photo_{photo.file_unique_id}.jpg",
        "mime_type": "image/jpeg",
        "file_size": photo.file_size or 0,
        "timestamp": update.message.date.isoformat(),
    }
    
    user_data[chat_id]["attachments_queue"].append(attachment_info)
    user_data[chat_id]["last_ai_response"] = None  # Clear AI response when attachment is received
    
    utils.save_user_data(user_data)
    logger.info(f"Stored photo attachment for chat_id {chat_id}")
    
    queue_count = len(user_data[chat_id]["attachments_queue"])
    await update.message.reply_text(
        f"📷 Photo received!\n"
        f"📊 Attachments in queue: {queue_count}/{config.MAX_ATTACHMENT_QUEUE_SIZE}\n"
        f"Use /forward to send all attachments to your email address."
    )


async def forward_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Forwards the last AI response or attachments to the user's configured email address."""
    chat_id = update.effective_chat.id

    if chat_id not in user_data or not user_data[chat_id].get("email"):
        await update.message.reply_text(
            "Your email address is not set. Please set it using /set_email <your_email@example.com>"
        )
        return

    email_address = user_data[chat_id]["email"]
    last_response = user_data[chat_id].get("last_ai_response")
    attachments_queue = user_data[chat_id].get("attachments_queue", [])

    # Check if we have attachments to forward
    if attachments_queue:
        await _forward_multiple_attachments(update, context, email_address, attachments_queue)
    elif last_response:
        await _forward_ai_response(update, context, email_address, last_response)
    else:
        await update.message.reply_text(
            "There's nothing to forward yet. Either talk to the AI or send documents/photos first!"
        )


async def _forward_multiple_attachments(update: Update, context: ContextTypes.DEFAULT_TYPE, email_address: str, attachments_queue: list) -> None:
    """Forwards multiple attachments to the user's email address."""
    chat_id = update.effective_chat.id
    
    if not attachments_queue:
        await update.message.reply_text("❌ No attachments to forward.")
        return
    
    # Validate total size doesn't exceed Gmail limits
    total_size = sum(att.get("file_size", 0) for att in attachments_queue)
    if total_size > config.GMAIL_MAX_TOTAL_SIZE:
        await update.message.reply_text(
            f"❌ Total attachment size ({total_size // (1024*1024)}MB) exceeds Gmail limit "
            f"({config.GMAIL_MAX_TOTAL_SIZE // (1024*1024)}MB). Please reduce the number of attachments."
        )
        return
    
    # Limit number of attachments per forward
    if len(attachments_queue) > config.MAX_ATTACHMENTS_PER_FORWARD:
        await update.message.reply_text(
            f"❌ Too many attachments! Maximum {config.MAX_ATTACHMENTS_PER_FORWARD} attachments per forward. "
            f"You have {len(attachments_queue)} attachments. Use /clear_attachments to clear some."
        )
        return
    
    await update.message.reply_text(
        f"📤 Preparing to forward {len(attachments_queue)} attachment(s)..."
    )
    
    temp_files = []
    try:
        # Download all files
        for i, attachment_info in enumerate(attachments_queue):
            file = await context.bot.get_file(attachment_info["file_id"])
            
            # Create a temporary file to store the attachment
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                await file.download_to_drive(tmp_file.name)
                temp_files.append({
                    "path": tmp_file.name,
                    "filename": attachment_info["file_name"],
                    "info": attachment_info
                })
        
        # Prepare email with multiple attachments
        attachment_count = len(attachments_queue)
        subject = f"Forwarded Attachments ({attachment_count} files)"
        
        body = f"You have received {attachment_count} attachment(s) from your Telegram bot.\n\n"
        body += "Files included:\n"
        for i, attachment_info in enumerate(attachments_queue, 1):
            body += f"{i}. {attachment_info['file_name']} "
            body += f"({attachment_info.get('file_size', 0) // 1024}KB, {attachment_info.get('type', 'unknown')})\n"
        
        body += f"\nTotal size: {total_size // 1024}KB\n"
        
        # Use the enhanced send_email function for multiple attachments
        success = utils.send_multiple_attachments_email(
            receiver_email=email_address,
            subject=subject,
            body=body,
            attachments=temp_files
        )
        
        if success:
            # Clear the attachments queue after successful send
            user_data[chat_id]["attachments_queue"] = []
            utils.save_user_data(user_data)
            
            await update.message.reply_text(
                f"✅ {attachment_count} attachment(s) have been forwarded to {email_address}.\n"
                f"📭 Attachment queue cleared."
            )
        else:
            await update.message.reply_text(
                "❌ Sorry, there was an error sending the email. Please check the bot logs or SMTP configuration."
            )
            
    except Exception as e:
        logger.error(f"Error forwarding multiple attachments for chat_id {chat_id}: {e}")
        await update.message.reply_text(
            f"❌ Sorry, there was an error processing the attachments. Please try again."
        )
    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file["path"])
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_file['path']}: {e}")


async def clear_attachments_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clears the current attachment queue."""
    chat_id = update.effective_chat.id
    
    if chat_id not in user_data:
        await update.message.reply_text("No attachments to clear.")
        return
    
    queue_size = len(user_data[chat_id].get("attachments_queue", []))
    if queue_size == 0:
        await update.message.reply_text("📭 No attachments in queue.")
        return
    
    user_data[chat_id]["attachments_queue"] = []
    utils.save_user_data(user_data)
    
    await update.message.reply_text(
        f"🗑️ Cleared {queue_size} attachment(s) from queue."
    )


async def list_attachments_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Lists current attachments in queue."""
    chat_id = update.effective_chat.id
    
    if chat_id not in user_data:
        await update.message.reply_text("No attachments in queue.")
        return
    
    attachments_queue = user_data[chat_id].get("attachments_queue", [])
    if not attachments_queue:
        await update.message.reply_text("📭 No attachments in queue.")
        return
    
    message = f"📋 Current attachments ({len(attachments_queue)}/{config.MAX_ATTACHMENT_QUEUE_SIZE}):\n\n"
    total_size = 0
    
    for i, att in enumerate(attachments_queue, 1):
        size_kb = att.get("file_size", 0) // 1024
        total_size += att.get("file_size", 0)
        message += f"{i}. {att['file_name']} ({size_kb}KB, {att['type']})\n"
    
    total_size_mb = total_size // (1024 * 1024)
    message += f"\n📊 Total size: {total_size_mb}MB/{config.GMAIL_MAX_TOTAL_SIZE // (1024*1024)}MB"
    message += f"\n🚀 Use /forward to send all attachments"
    message += f"\n🗑️ Use /clear_attachments to clear queue"
    
    await update.message.reply_text(message)


async def _forward_ai_response(update: Update, context: ContextTypes.DEFAULT_TYPE, email_address: str, ai_response: str) -> None:
    """Forwards an AI response to the user's email address."""
    chat_id = update.effective_chat.id
    
    subject = "AI Model Response Digest"
    body_intro = "Here's the latest response from the AI model:\n\n"

    # Word count for attachment decision (approximate)
    word_count = len(ai_response.split())
    max_words_inline = 1000  # As per requirement

    if word_count > max_words_inline:
        await update.message.reply_text(
            f"The response is quite long ({word_count} words). Preparing it as an attachment..."
        )
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+", delete=False, suffix=".md", encoding="utf-8"
            ) as tmp_file:
                tmp_file.write(ai_response)
                attachment_path = tmp_file.name

            attachment_filename = "ai_response.md"
            email_body = (
                body_intro
                + "The full response is attached as a Markdown (.md) file due to its length."
            )

            success = utils.send_email(
                receiver_email=email_address,
                subject=subject,
                body=email_body,
                attachment_path=attachment_path,
                attachment_filename=attachment_filename,
            )
            os.remove(attachment_path)  # Clean up the temp file
        except Exception as e:
            logger.error(f"Error creating or sending attachment: {e}")
            await update.message.reply_text(
                "Sorry, there was an error preparing the email attachment."
            )
            return
    else:
        email_body = body_intro + ai_response
        success = utils.send_email(
            receiver_email=email_address, subject=subject, body=email_body
        )

    if success:
        await update.message.reply_text(
            f"✅ The AI response has been forwarded to {email_address}."
        )
    else:
        await update.message.reply_text(
            "❌ Sorry, there was an error sending the email. Please check the bot logs or SMTP configuration."
        )


async def _process_email_update(
    update: Update, context: ContextTypes.DEFAULT_TYPE, email_address: str
) -> None:
    """Validates and saves the provided email address."""
    chat_id = update.effective_chat.id

    # Basic email validation (can be improved with regex if needed)
    if "@" not in email_address or "." not in email_address.split("@")[-1]:
        await update.message.reply_text(
            "That doesn't look like a valid email address. Please try again, or use /cancel_set_email to abort."
        )
        # If in a conversation, we might want to return the state to allow retry
        # For now, this function is called directly or after email is received.
        # If called from conversation handler and validation fails, the conversation handler
        # will decide whether to end or ask for retry.
        return  # Or raise an error to be caught by caller

    if chat_id not in user_data:
        user_data[chat_id] = {  # Initialize if somehow not present
            "email": email_address,
            "last_ai_response": None,
            "selected_model": config.DEFAULT_GEMINI_MODEL,
            "attachments_queue": [],
        }
    else:
        user_data[chat_id]["email"] = email_address

    utils.save_user_data(user_data)  # Save after setting email
    logger.info(f"User {chat_id} set email to {email_address}. Data saved.")
    await update.message.reply_text(
        f"Your email address has been set to: {email_address}"
    )


async def set_email_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Handles the /set_email command.
    If an email is provided as an argument, it's processed directly.
    Otherwise, it asks the user to reply with their email and enters the AWAITING_EMAIL_REPLY state.
    """
    chat_id = update.effective_chat.id  # Used for logging perhaps

    if context.args:  # Email provided directly in the command
        email_to_set = context.args[0]
        logger.info(f"Chat {chat_id}: /set_email called with argument: {email_to_set}")
        await _process_email_update(update, context, email_to_set)
        return (
            ConversationHandler.END
        )  # End conversation if any was active (though unlikely here) or just operate statelessly
    else:  # No email provided, start conversation
        logger.info(
            f"Chat {chat_id}: /set_email called without arguments. Asking for email reply."
        )
        await update.message.reply_text(
            "Please reply with the email address you'd like to set. Type /cancel_set_email to abort."
        )
        return AWAITING_EMAIL_REPLY  # Return the next state


async def received_email_reply(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """
    Handles the user's reply when the bot is awaiting an email address.
    Processes the received text as an email.
    """
    chat_id = update.effective_chat.id
    email_candidate = update.message.text
    logger.info(f"Chat {chat_id}: Received reply for email: {email_candidate}")

    # We need to check if _process_email_update was successful.
    # _process_email_update sends its own messages for success/failure.
    # To check success here, we can modify _process_email_update to return a boolean
    # or re-validate here and then call it.

    # Let's re-validate quickly here to decide conversation flow, then let _process_email_update do its full job.
    if "@" not in email_candidate or "." not in email_candidate.split("@")[-1]:
        await update.message.reply_text(
            "That doesn't look like a valid email address. "
            "Please try again with a valid email, or type /cancel_set_email to abort."
        )
        return AWAITING_EMAIL_REPLY  # Stay in the same state to allow retry

    # If basic validation passes, let _process_email_update handle the detailed logic and user messaging
    await _process_email_update(update, context, email_candidate)
    # _process_email_update already sends confirmation or its own detailed error message.
    # If it was successful, the user got "Your email address has been set to: ..."

    # Assume if _process_email_update was called, and basic validation passed, we end.
    # The user will see the confirmation from _process_email_update.
    # If _process_email_update had a more subtle failure not caught by basic check,
    # the user would see its specific error, and conversation ends. This is acceptable.
    return ConversationHandler.END  # End the conversation


async def cancel_set_email_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Cancels the ongoing action (e.g., setting email)."""
    chat_id = update.effective_chat.id
    logger.info(f"Chat {chat_id}: User cancelled the set_email conversation.")

    # Check if the user was actually in a conversation state we want to cancel
    # This basic version cancels any conversation it's a fallback for.
    # For more specific cancellation (e.g. if user is not in AWAITING_EMAIL_REPLY state),
    # context.user_data could be used to check a flag, but ConversationHandler handles this.

    await update.message.reply_text("The email setting process has been cancelled.")
    return ConversationHandler.END  # End the conversation


async def conversation_timeout(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    """Handles conversation timeout."""
    chat_id = update.effective_chat.id  # Or from query if it's a callback query timeout

    # Check if context.user_data has any info about which conversation timed out, if needed.
    # For this specific email conversation, we know what timed out.
    logger.info(f"Chat {chat_id}: Conversation for setting email timed out.")

    # update.callback_query will be None if it's a message based timeout
    # update.message will be the message that triggered the timeout (if any, usually none for timeout itself)
    # So, we need to send a new message.

    # It's possible that 'update.effective_message' is what we need to reply to,
    # but for a timeout, the original message that started the conversation might be old.
    # Sending a new message is safer.
    await context.bot.send_message(
        chat_id=chat_id,
        text="You took too long to reply. The email setting process has been cancelled.",
    )
    return ConversationHandler.END  # End the conversation


# update_email will be an alias, handled by adding another CommandHandler pointing to set_email_command


async def set_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sets the user's preferred Gemini model for chat."""
    chat_id = update.effective_chat.id

    if not context.args:
        available_models_text = ", ".join(config.AVAILABLE_GEMINI_MODELS)
        await update.message.reply_text(
            "Please provide a model name.\n"
            f"Usage: /set_model <model_name>\n"
            f"Available models: {available_models_text}"
        )
        return

    chosen_model = context.args[0]

    if chosen_model not in config.AVAILABLE_GEMINI_MODELS:
        available_models_text = ", ".join(config.AVAILABLE_GEMINI_MODELS)
        await update.message.reply_text(
            f"Sorry, '{chosen_model}' is not a recognized model.\n"
            f"Available models: {available_models_text}"
        )
        return

    if chat_id not in user_data:  # Initialize if somehow not present
        user_data[chat_id] = {
            "email": config.DEFAULT_EMAIL,
            "last_ai_response": None,
            "selected_model": chosen_model,  # Set the new model
            "attachments_queue": [],
        }
    else:
        user_data[chat_id]["selected_model"] = chosen_model

    utils.save_user_data(user_data)  # Save after setting model
    logger.info(f"User {chat_id} set model to {chosen_model}. Data saved.")
    await update.message.reply_text(f"Your AI model has been set to: {chosen_model}")


async def switch_model_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Displays an inline keyboard for quick model switching."""
    # Get up to the first 4 models from the configured available models
    # config.AVAILABLE_GEMINI_MODELS is already loaded in src.config
    models_for_menu = config.AVAILABLE_GEMINI_MODELS[:4]

    if not models_for_menu:
        await update.message.reply_text(
            "No Gemini models are configured for the menu. Please check the .env file "
            "and ensure GEMINI_MODELS is set."
        )
        return

    keyboard = []
    for model_name in models_for_menu:
        keyboard.append(
            [
                InlineKeyboardButton(
                    model_name, callback_data=f"model_select:{model_name}"
                )
            ]
        )

    # Also add a button to list all available models via /set_model command, if there are more
    if len(config.AVAILABLE_GEMINI_MODELS) > len(models_for_menu):
        keyboard.append(
            [
                InlineKeyboardButton(
                    "More models (use /set_model)",
                    callback_data="model_select:show_all_info",
                )
            ]
        )
    elif (
        not config.AVAILABLE_GEMINI_MODELS
    ):  # Should be caught by models_for_menu check, but as a safeguard
        await update.message.reply_text(
            "No models configured. Use /set_model <model_name> after configuring GEMINI_MODELS in .env."
        )
        return

    reply_markup = InlineKeyboardMarkup(keyboard)

    chat_id = update.effective_chat.id
    current_model = "Not set"
    if chat_id in user_data and user_data[chat_id].get("selected_model"):
        current_model = user_data[chat_id]["selected_model"]
    elif config.DEFAULT_GEMINI_MODEL:
        current_model = config.DEFAULT_GEMINI_MODEL

    await update.message.reply_text(
        f"Your current AI model is: {current_model}\n"
        "Choose a new Gemini model from the menu below:",
        reply_markup=reply_markup,
    )


async def model_button_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles callbacks from the model selection inline keyboard."""
    query = update.callback_query
    await query.answer()  # Answer the callback query immediately

    callback_data = query.data
    chat_id = query.message.chat_id

    if callback_data == "model_select:show_all_info":
        available_models_text = ", ".join(config.AVAILABLE_GEMINI_MODELS)
        if not available_models_text:
            available_models_text = "None configured."
        await query.edit_message_text(
            text=f"To select from all available models, please use the command:\n"
            f"`/set_model <model_name>`\n\n"
            f"Available models: {available_models_text}\n\n"
            f"Example: `/set_model {config.AVAILABLE_GEMINI_MODELS[0] if config.AVAILABLE_GEMINI_MODELS else 'gemini-pro'}`",
        )
        return

    # Expected format: "model_select:MODEL_NAME"
    try:
        action, model_name = callback_data.split(":", 1)
        if action == "model_select" and model_name in config.AVAILABLE_GEMINI_MODELS:
            if (
                chat_id not in user_data
            ):  # Should be initialized by /start or other commands
                user_data[chat_id] = {
                    "email": config.DEFAULT_EMAIL,
                    "last_ai_response": None,
                    "selected_model": model_name,
                    "attachments_queue": [],
                }
            else:
                user_data[chat_id]["selected_model"] = model_name

            utils.save_user_data(user_data)
            logger.info(
                f"User {chat_id} selected model '{model_name}' via menu. Data saved."
            )

            # Edit the original message to confirm selection and remove keyboard
            await query.edit_message_text(text=f"AI model set to: {model_name}")
        else:
            logger.warning(
                f"Invalid or unknown model in callback_data for chat_id {chat_id}: {callback_data}"
            )
            await query.edit_message_text(
                text="Invalid selection. Please try again or use /set_model."
            )
    except ValueError:
        logger.error(
            f"Error parsing callback_data for chat_id {chat_id}: {callback_data}"
        )
        await query.edit_message_text(
            text="Error processing selection. Please try again."
        )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in model_button_callback for chat_id {chat_id}: {e}"
        )
        # Try to send a new message if editing fails or is inappropriate
        await context.bot.send_message(
            chat_id=chat_id, text="An error occurred. Please try selecting again."
        )


def main() -> None:
    """Start the bot."""
    if not config.TELEGRAM_BOT_TOKEN:
        logger.critical(
            "TELEGRAM_BOT_TOKEN not found in configuration. Bot cannot start."
        )
        return

    application = ApplicationBuilder().token(config.TELEGRAM_BOT_TOKEN).build()

    # Command Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("forward", forward_command))
    application.add_handler(CommandHandler("clear_attachments", clear_attachments_command))
    application.add_handler(CommandHandler("list_attachments", list_attachments_command))
    # application.add_handler(CommandHandler("set_email", set_email_command)) # Old one
    # application.add_handler(CommandHandler("update_email", set_email_command)) # Old alias
    application.add_handler(CommandHandler("set_model", set_model_command))
    application.add_handler(CommandHandler("switch_model", switch_model_command))

    # Add ConversationHandler for /set_email and /update_email
    set_email_conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("set_email", set_email_command),
            CommandHandler("update_email", set_email_command),  # Alias entry point
        ],
        states={
            AWAITING_EMAIL_REPLY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, received_email_reply)
            ],
            ConversationHandler.TIMEOUT: [  # Handler for the timeout event
                conversation_timeout  # Our previously defined timeout function
            ],
        },
        fallbacks=[CommandHandler("cancel_set_email", cancel_set_email_command)],
        conversation_timeout=CONVERSATION_TIMEOUT_SECONDS,  # Set the timeout duration
    )
    application.add_handler(set_email_conv_handler)

    # CallbackQueryHandler for model selection
    application.add_handler(
        CallbackQueryHandler(model_button_callback, pattern="^model_select:")
    )

    # Attachment handlers (must be added before the general message handler)
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Message Handler for AI interaction (ensure it's added after ConversationHandler and attachment handlers)
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("Starting bot polling...")
    application.run_polling()


if __name__ == "__main__":
    # This allows running the bot directly for development/testing
    # Ensure .env file is in the project root or accessible
    main()
