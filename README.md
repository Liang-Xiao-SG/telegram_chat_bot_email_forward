# Telegram AI Email Forwarder Bot

This Telegram bot allows you to chat with a Gemini AI model and forward the conversation (or parts of it) to your email address. It's designed to be deployed using Docker.

## Features

- **AI Chat**: Interact with Google's Gemini models directly through Telegram.
- **Email Forwarding**: Forward AI responses to your configured email address using the `/forward` command.
- **Long Message Handling**: AI responses exceeding 1000 words are automatically sent as `.txt` file attachments in emails.
- **Customizable Email**: Set and update your default email address using `/set_email` or `/update_email`.
- **Model Selection**: Choose your preferred Gemini model from a configurable list using `/set_model`.
- **Persistent Storage**: Your email address, preferred model, and the last AI response are saved and persist across bot restarts (using Docker volumes).
- **Dockerized**: Easy to build and deploy using Docker and Docker Compose.
- **Configurable**: All sensitive keys and settings are managed through an `.env` file.

## Prerequisites

- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Install Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop).

## Setup and Configuration

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Configure Environment Variables:**
    Create a `.env` file in the project root directory by copying the example:
    ```bash
    cp .env.example .env
    ```
    Now, open the `.env` file and fill in your details:

    *   `TELEGRAM_BOT_TOKEN`: Your Telegram Bot Token obtained from [BotFather](https://core.telegram.org/bots#6-botfather).
    *   `GEMINI_API_KEY`: Your API key for the Gemini API from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   `DEFAULT_EMAIL` (Optional): A default email address where forwards will be sent if the user hasn't set one.
    *   `GEMINI_MODELS`: A comma-separated list of Gemini model names you want to make available (e.g., `gemini-pro,gemini-1.5-flash-latest`). Do not use spaces between model names. The first model in this list will be the default if `DEFAULT_GEMINI_MODEL` is not set or invalid.
    *   `DEFAULT_GEMINI_MODEL`: The default Gemini model to use for chats (must be one of the models listed in `GEMINI_MODELS`).
    *   `SMTP_SERVER`: The hostname or IP address of your SMTP server (e.g., `smtp.gmail.com`).
    *   `SMTP_PORT`: The port number for your SMTP server (e.g., `587` for TLS, `465` for SSL).
    *   `SMTP_USERNAME`: Your email address (used to log in to the SMTP server).
    *   `SMTP_PASSWORD`: Your email password or an app-specific password if using services like Gmail with 2FA.

    **Example `.env` structure:**
    ```env
    TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
    GEMINI_API_KEY=AIzaSy***********************************
    DEFAULT_EMAIL=mydefault@example.com
    GEMINI_MODELS=gemini-pro,gemini-1.0-pro,gemini-1.5-flash-latest
    DEFAULT_GEMINI_MODEL=gemini-1.5-flash-latest
    SMTP_SERVER=smtp.example.com
    SMTP_PORT=587
    SMTP_USERNAME=user@example.com
    SMTP_PASSWORD=supersecretpassword
    ```

## How to Build and Run

The application is designed to be run with Docker Compose.

1.  **Build the Docker Image and Start the Container:**
    ```bash
    docker-compose build
    docker-compose up -d
    ```
    The bot should now be running in the background.

2.  **Building for a Specific Platform (e.g., `linux/arm64`):**
    If you need to build the image for a specific platform like `linux/arm64` (e.g., for Raspberry Pi or certain cloud VMs), you can use `docker buildx`.

    **Option 1: Set default platform for Docker Compose (Recommended for simplicity)**
    You can set the `DOCKER_DEFAULT_PLATFORM` environment variable before running Docker Compose commands:
    ```bash
    export DOCKER_DEFAULT_PLATFORM=linux/arm64
    docker-compose build
    docker-compose up -d
    ```
    Or, set it inline:
    ```bash
    DOCKER_DEFAULT_PLATFORM=linux/arm64 docker-compose build
    DOCKER_DEFAULT_PLATFORM=linux/arm64 docker-compose up -d
    ```

    **Option 2: Build with `docker buildx` and then run with Docker Compose**
    First, build the image using `docker buildx` and tag it appropriately so Docker Compose can find it. The image name should match what's implicitly used by Docker Compose (usually `projectname_servicename`, e.g., `telegrambot_ai_email_bot` if your project directory is `telegrambot`) or explicitly defined in `docker-compose.yml` with `image: your-custom-image-name`.
    Assuming your service name in `docker-compose.yml` is `ai_email_bot` and your project directory is named, for example, `my_bot_project`, Docker Compose might look for an image named `my_bot_project_ai_email_bot`.

    To build and tag for `linux/arm64`:
    ```bash
    docker buildx build --platform linux/arm64 -t my_bot_project_ai_email_bot:latest --load .
    ```
    Replace `my_bot_project_ai_email_bot:latest` with the correct image name if you've specified one in `docker-compose.yml` or determine the name Docker Compose uses. The `--load` flag builds the image and loads it into the local Docker daemon, making it available for `docker-compose up`.

    Then, you can run:
    ```bash
    docker-compose up -d
    ```
    Docker Compose should pick up the pre-built image. If you specified `platform: linux/arm64` within the service definition in your `docker-compose.yml` (and your Docker Compose version supports it), that can also help ensure the correct platform is used at runtime if the image is multi-arch.

3.  **Viewing Logs:**
    To view the bot's logs:
    ```bash
    docker-compose logs -f
    ```

4.  **Stopping the Bot:**
    ```bash
    docker-compose down
    ```

## Bot Commands

-   `/start` - Welcome message, shows current email and AI model settings.
-   `/help` - Detailed help message with all commands and current settings.
-   `/set_email <email_address>` - Set or update your default email address for forwarding.
    Example: `/set_email myname@example.com`
-   `/update_email <email_address>` - Alias for `/set_email`.
-   `/forward` - Forward the last AI response to your default email address. If the response is very long, it will be sent as a text file attachment.
-   `/set_model <model_name>` - Choose a specific Gemini model to interact with from the `GEMINI_MODELS` list in your `.env`.
    Example: `/set_model gemini-pro`

## Data Persistence

User data (email address, selected AI model, and the last AI response) is stored in a JSON file (`user_data.json`) inside the `data/` directory on your host machine. This directory is mounted into the Docker container, ensuring that your settings and the last response persist even if you stop or restart the bot container.

## Project Structure

```
.
├── data/                     # Persisted user data (mounted as Docker volume)
│   └── .gitkeep              # Ensures 'data' directory is tracked by Git
├── src/                      # Python source code
│   ├── bot.py                # Main Telegram bot logic, command handlers
│   ├── config.py             # Configuration loader (environment variables)
│   └── utils.py              # Utility functions (Gemini API, email sending, data persistence)
├── .env.example              # Example environment variables file
├── Dockerfile                # Instructions to build the Docker image
├── docker-compose.yml        # Docker Compose configuration for services
├── README.md                 # This file
└── requirements.txt          # Python package dependencies
```

---

Feel free to contribute or report issues!
