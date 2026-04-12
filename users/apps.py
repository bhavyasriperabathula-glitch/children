from django.apps import AppConfig
from django.db import connection

class UsersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'users'

    def ready(self):
        """
        Auto-initialize the UserRegistrations table if it doesn't exist.
        This provides a fail-safe for deployment environments like Render.
        """
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
