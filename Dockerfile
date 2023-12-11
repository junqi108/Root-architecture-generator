# Use a smaller base image
FROM python:3.9-slim

# Set working directory
WORKDIR /usr/src/app

# Install necessary packages (combine into one RUN command to reduce layers)
RUN apt-get update && \
    apt-get install -y gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/* # Clean up to reduce image size

# Copy only necessary files
COPY requirements.txt ./

# Install Python dependencies
# First, install pyperclip without enforcing binary distribution
RUN pip install pyperclip==1.8.2

# Then, install the rest of the requirements, potentially using wheels
RUN pip install --no-cache-dir -r requirements.txt

# Further steps for your application...
