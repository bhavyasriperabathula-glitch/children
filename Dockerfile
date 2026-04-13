FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# (Optional but safe) Django setup
RUN python manage.py collectstatic --noinput || true
RUN python manage.py migrate --noinput || true

# Expose Hugging Face port
EXPOSE 7860

# Start Django app
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "ADHD.wsgi:application"]