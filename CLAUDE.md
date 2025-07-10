# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Docker Commands (Primary Development Method)
```bash
# Build and run the bot (recommended approach)
docker-compose build
docker-compose up -d

# For ARM64 platforms (e.g., Raspberry Pi, certain VMs)
DOCKER_DEFAULT_PLATFORM=linux/arm64 docker-compose build
DOCKER_DEFAULT_PLATFORM=linux/arm64 docker-compose up -d

# Alternative with docker buildx for ARM64
docker buildx build --platform linux/arm64 -t <Docker Hub account>/ai_email_bot --push .

# View logs
docker-compose logs -f

# Stop the bot
docker-compose down
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the bot directly (ensure .env is configured)
python -m src.bot
```

## Architecture Overview

This is a Telegram bot that integrates with Google's Gemini AI models and provides email forwarding functionality. The bot is designed to run in Docker containers with persistent data storage.

### Core Components

- **src/bot.py**: Main Telegram bot logic with command handlers, conversation states, and user interaction
- **src/config.py**: Configuration management loading environment variables from .env file
- **src/utils.py**: Utility functions for Gemini API integration, email sending, and data persistence

### Key Architecture Patterns

1. **Conversation State Management**: Uses python-telegram-bot's ConversationHandler for multi-step interactions (e.g., setting email addresses)

2. **Asynchronous AI Integration**: Gemini API calls are handled asynchronously using `generate_content_async()` 

3. **Data Persistence**: User data (email addresses, model preferences, last AI responses) is stored in JSON format with Docker volume mounting for persistence across container restarts

4. **Email Attachment Handling**: Large AI responses (>1000 words) are automatically sent as Markdown file attachments via email

5. **Model Selection**: Dynamic model switching via inline keyboards and command-based selection from configured model list

### Data Flow

1. User sends message → Bot processes via handle_message()
2. Message sent to selected Gemini model via utils.generate_text()
3. Response stored in user_data and saved to JSON
4. Response sent to user (either directly or as file if too long)
5. User can forward response to email via /forward command

### Configuration Management

- Environment variables loaded via python-dotenv
- Configuration validation ensures required API keys are present
- SMTP settings for email functionality
- Configurable Gemini model list from environment variables

### Error Handling

- Comprehensive error handling for API failures, email sending, and file operations
- Graceful fallbacks for Markdown parsing failures
- User-friendly error messages for common issues

## Important Implementation Details

### User Data Structure
```python
user_data[chat_id] = {
    "email": str,                 # User's email address
    "last_ai_response": str,      # Last AI response for forwarding
    "selected_model": str         # User's preferred Gemini model
}
```

### Message Length Handling
- Telegram messages limited to ~3800 characters (conservative buffer)
- Email attachments for responses >1000 words
- Automatic file attachment for oversized Telegram messages

### Model Management
- Available models configured via GEMINI_MODELS environment variable
- Model validation ensures only configured models are accessible
- Default model fallback if user selection is invalid

### Email Integration
- SMTP authentication with configurable server settings
- Markdown file attachments for long responses
- HTML email support for rich formatting

## Environment Configuration

Required environment variables (see .env.example):
- TELEGRAM_BOT_TOKEN: Bot token from BotFather
- GEMINI_API_KEY: Google AI Studio API key
- SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD: Email configuration
- GEMINI_MODELS: Comma-separated list of available models
- DEFAULT_GEMINI_MODEL: Default model selection

## Data Persistence

- User data stored in `/app/data/user_data.json` inside container
- Host directory `./data/` mounted as Docker volume
- Automatic directory creation and JSON file management
- Chat IDs used as integer keys for user data indexing