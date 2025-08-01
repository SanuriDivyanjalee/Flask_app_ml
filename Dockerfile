# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model
COPY app.py .
COPY model.pkl .

# Expose Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]