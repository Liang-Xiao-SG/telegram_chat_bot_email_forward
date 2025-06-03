# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
# Using --default-timeout to prevent timeouts during pip install
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the src directory into the container at /app/src
COPY src/ src/
# Copy .env.example for reference, though actual .env will be mounted or variables injected
COPY .env.example .

# Ensure the data directory exists and has appropriate permissions if needed
# The volume mount will create it if it doesn't exist on the host,
# but good to ensure the path is expected by the app.
# The user data load/save functions in utils.py already create /app/data if not present.

# Command to run the application
# Using python -m src.bot to run the bot module
CMD ["python", "-m", "src.bot"]
