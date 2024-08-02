FROM python:3.10-slim

# Install wget and other dependencies
RUN apt-get update && apt-get install -y \
  wget \
  dpkg \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application files and script from Dark-Pattern directory
COPY Dark-Pattern/ /app

# Copy and run the shell script
COPY setup-chrome.sh /usr/local/bin/setup-chrome.sh
RUN chmod +x /usr/local/bin/setup-chrome.sh && /usr/local/bin/setup-chrome.sh

# Install Python dependencies
RUN pip install -r requirements.txt

# Specify the command to run your Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
