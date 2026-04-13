FROM python:3.9-slim

# Install system dependencies (FIXED)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# 🔥 IMPORTANT: Django setup (safe handling)
RUN python manage.py collectstatic --noinput || true
RUN python manage.py makemigrations --noinput || true
RUN python manage.py migrate --noinput || true

# Set permissions
RUN chown -R user:user /app

# Switch user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose Hugging Face port
EXPOSE 7860

# Start server
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "ADHD.wsgi:application"]