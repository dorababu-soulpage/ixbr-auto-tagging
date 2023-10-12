# Use a base image with Python preinstalled
FROM python:3.8.10

# Set the working directory
WORKDIR /app

# Copy your application files into the container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# upgrade pip
RUN pip3 install --upgrade pip

# Expose the port your Flask app will run on
EXPOSE 5000

# Specify the command to run your Flask app
CMD ["python", "app.py"]
