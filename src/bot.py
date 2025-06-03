import logging
import os
import tempfile # For handling large attachments
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# Assuming your config and utils are structured in src
from src import config  # Loads .env variables
from src import utils   # For generate_text and send_email

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load user data at startup
user_data = utils.load_user_data()
logger.info(f"Loaded user data: {len(user_data)} users.")

# Define a constant for Telegram message length limit
TELEGRAM_MAX_MESSAGE_LENGTH = 3800 # Telegram's official limit is 4096, using a more conservative buffer

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    user = update.effective_user
    chat_id = update.effective_chat.id

    # Initialize user data if not present
    if chat_id not in user_data:
        user_data[chat_id] = {
            "email": config.DEFAULT_EMAIL, # Load default from config
            "last_ai_response": None,
            "selected_model": config.DEFAULT_GEMINI_MODEL
        }
        utils.save_user_data(user_data) # Save after initializing new user
        logger.info(f"Initialized data for new user: {chat_id}. Data saved.")

    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
    )
    await update.message.reply_text(
        "Welcome to the AI Forwarder Bot! I can help you chat with an AI model "
        "and forward the responses to your email address.

"
        f"Your current default email is: {user_data[chat_id].get('email') or 'Not set'}
"
        f"Your current AI model is: {user_data[chat_id].get('selected_model')}

"
        "Use /help to see all commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a help message when the /help command is issued."""
    chat_id = update.effective_chat.id
    current_email = user_data.get(chat_id, {}).get('email') or 'Not set'
    current_model = user_data.get(chat_id, {}).get('selected_model', config.DEFAULT_GEMINI_MODEL)

    available_models_text = ", ".join(config.AVAILABLE_GEMINI_MODELS)
    if not available_models_text:
        available_models_text = "No models configured (check .env)."

    help_text = (
        "Here's how I can help you:

"
        f"Your current default email: **{current_email}**
"
        f"Your current AI model: **{current_model}**

"
        "**Chatting with AI:**
"
        "Simply send me any message, and I'll pass it to the AI. "
        "I'll try to render the AI's response using Markdown formatting directly in the chat.
"
        "*(If an AI response is too long for a direct message, I'll send it as an `.md` (Markdown) file instead.)*

"
        "**Commands:**
"
        "/start - Welcome message & current settings.
"
        "/help - Show this help message.
"
        "/set_email <email_address> - Set/update your default email for forwarding.
"
        "  Example: `/set_email user@example.com`
"
        "/forward - Forward the last AI response to your email. "
        "If long, it's sent as an attachment.
"
        f"/set_model <model_name> - Choose an AI model. Available: {available_models_text}
"
        f"  Example: `/set_model {config.AVAILABLE_GEMINI_MODELS[0] if config.AVAILABLE_GEMINI_MODELS else 'gemini-pro'}`

"
        "**Note:** Your email and model preferences are saved automatically."
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles regular messages by sending them to the AI and returning the response."""
    chat_id = update.effective_chat.id
    user_message = update.message.text

    if chat_id not in user_data: # Should be initialized by /start, but as a fallback
        user_data[chat_id] = {
            "email": config.DEFAULT_EMAIL,
            "last_ai_response": None,
            "selected_model": config.DEFAULT_GEMINI_MODEL
        }
        # Save immediately if a new user entry is created this way
        utils.save_user_data(user_data)
        logger.info(f"Initialized and saved data for new user via handle_message: {chat_id}")

    selected_model = user_data[chat_id].get("selected_model", config.DEFAULT_GEMINI_MODEL)

    # Let user know the bot is working on it
    processing_message = await update.message.reply_text(f"🤖 Thinking with {selected_model}...", disable_notification=True)

    ai_response = await utils.generate_text(prompt=user_message, model_name=selected_model)

    # Delete the "Thinking..." message
    if processing_message:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=processing_message.message_id)
        except Exception as e:
            logger.warning(f"Could not delete 'Thinking...' message: {e}")

    if ai_response and not ai_response.startswith("Error:"):
        user_data[chat_id]["last_ai_response"] = ai_response
        utils.save_user_data(user_data) # Save data after getting a valid AI response
        logger.info(f"Stored AI response for chat_id {chat_id} and saved data.")
    elif not ai_response: # Empty response but not an explicit "Error:"
        logger.error(f"AI response was empty for chat_id {chat_id}. Not storing.")
        ai_response = "Sorry, I received an empty response from the AI."
        # Do not store this as last_ai_response
    # If ai_response starts with "Error:", it will be sent as is.

    # Handle sending the response (potentially long)
    if len(ai_response) > TELEGRAM_MAX_MESSAGE_LENGTH:
        logger.info(f"AI response for chat_id {chat_id} is too long ({len(ai_response)} chars). Sending as a file.")
        try:
            with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".md", encoding='utf-8') as tmp_file:
                tmp_file.write(ai_response)
                attachment_path = tmp_file.name

            caption_text = "The AI's response was too long to display directly in the chat, so it's attached as a text file."
            await context.bot.send_document(
                chat_id=chat_id,
                document=open(attachment_path, 'rb'),
                filename="ai_response.md",
                caption=caption_text
            )
            os.remove(attachment_path) # Clean up the temp file
            logger.info(f"Sent long AI response as a file to chat_id {chat_id}.")
        except Exception as e:
            logger.error(f"Error sending long AI response as file to {chat_id}: {e}")
            await update.message.reply_text("There was an error sending the AI's long response as a file. You can try using /forward to get it via email.")
    else:
        is_error_or_generic_response = ai_response.startswith("Error:") or ai_response == "Sorry, I received an empty response from the AI."
        if is_error_or_generic_response:
            await update.message.reply_text(ai_response) # Send errors/generic as plain text
        else:
            try:
                await update.message.reply_text(ai_response, parse_mode=ParseMode.MARKDOWN_V2)
            except Exception as e: # Catch potential parsing errors from MarkdownV2
                logger.warning(f"Failed to send message with MarkdownV2 for chat_id {chat_id}: {e}. Sending as plain text.")
                await update.message.reply_text(ai_response) # Fallback to plain text

async def forward_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Forwards the last AI response to the user's configured email address."""
    chat_id = update.effective_chat.id

    if chat_id not in user_data or not user_data[chat_id].get("email"):
        await update.message.reply_text("Your email address is not set. Please set it using /set_email <your_email@example.com>")
        return

    email_address = user_data[chat_id]["email"]
    last_response = user_data[chat_id].get("last_ai_response")

    if not last_response:
        await update.message.reply_text("There's no AI response to forward yet. Talk to the AI first!")
        return

    subject = "AI Model Response Digest"
    body_intro = "Here's the latest response from the AI model:

"

    # Word count for attachment decision (approximate)
    word_count = len(last_response.split())
    max_words_inline = 1000 # As per requirement

    if word_count > max_words_inline:
        await update.message.reply_text(f"The response is quite long ({word_count} words). Preparing it as an attachment...")
        try:
            with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".md", encoding='utf-8') as tmp_file:
                tmp_file.write(last_response)
                attachment_path = tmp_file.name

            attachment_filename = "ai_response.md"
            email_body = body_intro + "The full response is attached as a Markdown (.md) file due to its length."

            success = utils.send_email(
                receiver_email=email_address,
                subject=subject,
                body=email_body,
                attachment_path=attachment_path,
                attachment_filename=attachment_filename
            )
            os.remove(attachment_path) # Clean up the temp file
        except Exception as e:
            logger.error(f"Error creating or sending attachment: {e}")
            await update.message.reply_text("Sorry, there was an error preparing the email attachment.")
            return
    else:
        email_body = body_intro + last_response
        success = utils.send_email(
            receiver_email=email_address,
            subject=subject,
            body=email_body
        )

    if success:
        await update.message.reply_text(f"The AI response has been forwarded to {email_address}.")
    else:
        await update.message.reply_text("Sorry, there was an error sending the email. Please check the bot logs or SMTP configuration.")

async def set_email_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sets or updates the user's email address."""
    chat_id = update.effective_chat.id

    if not context.args:
        await update.message.reply_text("Please provide an email address.\nUsage: /set_email your_email@example.com")
        return

    new_email = context.args[0]
    # Basic email validation (can be improved with regex if needed)
    if "@" not in new_email or "." not in new_email.split('@')[-1]:
        await update.message.reply_text("That doesn't look like a valid email address. Please try again.")
        return

    if chat_id not in user_data:
        user_data[chat_id] = { # Initialize if somehow not present
            "email": new_email,
            "last_ai_response": None,
            "selected_model": config.DEFAULT_GEMINI_MODEL
        }
    else:
        user_data[chat_id]["email"] = new_email

    utils.save_user_data(user_data) # Save after setting email
    logger.info(f"User {chat_id} set email to {new_email}. Data saved.")
    await update.message.reply_text(f"Your email address has been set to: {new_email}")

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

    if chat_id not in user_data: # Initialize if somehow not present
        user_data[chat_id] = {
            "email": config.DEFAULT_EMAIL,
            "last_ai_response": None,
            "selected_model": chosen_model # Set the new model
        }
    else:
        user_data[chat_id]["selected_model"] = chosen_model

    utils.save_user_data(user_data) # Save after setting model
    logger.info(f"User {chat_id} set model to {chosen_model}. Data saved.")
    await update.message.reply_text(f"Your AI model has been set to: {chosen_model}")

def main() -> None:
    """Start the bot."""
    if not config.TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN not found in configuration. Bot cannot start.")
        return

    application = ApplicationBuilder().token(config.TELEGRAM_BOT_TOKEN).build()

    # Command Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("forward", forward_command))
    application.add_handler(CommandHandler("set_email", set_email_command))
    application.add_handler(CommandHandler("update_email", set_email_command)) # Alias
    application.add_handler(CommandHandler("set_model", set_model_command))

    # Message Handler for AI interaction
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Starting bot polling...")
    application.run_polling()

if __name__ == "__main__":
    # This allows running the bot directly for development/testing
    # Ensure .env file is in the project root or accessible
    main()
