from django.apps import AppConfig
from django.db import connection
from django.core.management import call_command
import sys

class UsersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'users'

    def ready(self):
        """
        Auto-initialize the entire database if it doesn't exist.
        This provides a fail-safe for core tables like 'django_session'.
        """
        # Only run migrations in the main process (not during collectstatic or other commands)
        if 'runserver' in sys.argv or 'gunicorn' in sys.argv or 'ADHD.wsgi' in sys.argv:
            try:
                print("🚀 Auto-initializing database...")
                call_command('migrate', interactive=False)
                
                # Double-check custom table
                with connection.cursor() as cursor:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS UserRegistrations (
                            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                            name VARCHAR(100) NOT NULL,
                            loginid VARCHAR(100) NOT NULL UNIQUE,
                            password VARCHAR(100) NOT NULL,
                            mobile VARCHAR(100) NOT NULL UNIQUE,
                            email VARCHAR(100) NOT NULL UNIQUE,
                            locality VARCHAR(100) NOT NULL,
                            address VARCHAR(1000) NOT NULL,
                            city VARCHAR(100) NOT NULL,
                            state VARCHAR(100) NOT NULL,
                            status VARCHAR(100) NOT NULL
                        );
                    """)
                print("✅ Database initialization complete.")
            except Exception as e:
                print(f"⚠️ Database init warning: {e}")
