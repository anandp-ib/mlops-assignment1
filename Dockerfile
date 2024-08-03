# Use the official Python image as the base image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Expose the port that the Flask app will run on
EXPOSE 5000

# Define the command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
