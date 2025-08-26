FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /workspaces

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
