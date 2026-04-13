FROM python:3.9-slim

WORKDIR /app

# 👇 PLACE IT HERE
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "adhd_django_app.wsgi:application"]