# Use the official Python 3.10.8 image from the Docker Hub
FROM python:3.10.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
  && apt-get install -y \
  build-essential \
  curl \
  gnupg \
  wget \
  unzip \
  libnss3 \
  libatk-bridge2.0-0 \
  libgtk-3-0 \
  libvulkan1 \
  xdg-utils \
  libu2f-udev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install Chrome from Google's official repository
RUN curl -sSL https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
  && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | tee /etc/apt/sources.list.d/google-chrome.list \
  && apt-get update \
  && apt-get install -y google-chrome-stable \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip \
  && pip install -r requirements.txt

# Copy the application code
COPY . /app/

# Expose the port the app runs on
EXPOSE 5000

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
