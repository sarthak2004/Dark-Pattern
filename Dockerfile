# Use an appropriate base image
FROM python:3.10.8-slim

# Copy your application and script
COPY . /app
WORKDIR /app

# Copy and run the shell script
COPY setup-chrome.sh /usr/local/bin/setup-chrome.sh
RUN chmod +x /usr/local/bin/setup-chrome.sh && /usr/local/bin/setup-chrome.sh

# Install dependencies
RUN pip install -r requirements.txt

# Specify the command to run your Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
