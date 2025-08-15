# Use a lightweight official Python image as a base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies first
# This improves build speed by leveraging Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port that the application will run on
EXPOSE 8000

# The command to start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]