from django.urls import path
from . import views

urlpatterns = [
    path('UserHome/', views.UserHome, name='UserHome'),
    path('training/', views.training, name='training'),
    path('prediction/', views.prediction, name='prediction'),
]
