from django.urls import path
from .views import upload_file, UserHome, predict_api

# urlpatterns = [
#     path('upload/', upload_file),
# ]
urlpatterns = [
    path('', UserHome),          
    path('upload/', upload_file),
    path('predict/', predict_api),
]