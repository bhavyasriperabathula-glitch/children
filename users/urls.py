```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.AdminLoginCheck, name='home'),   # homepage
    path('admin-home/', views.AdminHome, name='admin_home'),
    path('view-users/', views.ViewRegisteredUsers, name='view_users'),
    path('activate-user/', views.AdminActivaUsers, name='activate_user'),
    path('upload/', views.upload_file, name='upload_file'),
]
```
