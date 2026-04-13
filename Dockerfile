FROM python:3.9-slim

# Install necessary system libraries for OpenCV and Database operations
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user called "user" with user ID 1000
# (Hugging Face strongly recommends running containers as a non-root user)
RUN useradd -m -u 1000 user

# Set the working directory to /app
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements into container
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run Django setup commands (collectstatic and database setup)
RUN python manage.py collectstatic --noinput
RUN python manage.py makemigrations --noinput
RUN python manage.py migrate --noinput

# Give ownership of the /app directory to the new "user"
RUN chown -R user:user /app

# Switch to the new "user"
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose port 7860 (This is the specific port Hugging Face Spaces looks for)
EXPOSE 7860

# Start the Django application using Gunicorn on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "ADHD.wsgi:application"]
