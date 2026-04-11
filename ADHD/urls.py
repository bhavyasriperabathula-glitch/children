"""ADHD URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic.base import RedirectView
from django.conf import settings
from admins import views as admins
from users import views as usr
from . import views as mainView
from .views import predict

from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    # ✅ Favicon
    path('favicon.ico', RedirectView.as_view(url=settings.STATIC_URL + 'favicon.ico')),

    # ✅ Admin
    path('admin/', admin.site.urls),

    # ✅ Main Pages
    path('', mainView.index, name='index'),
    path('index/', mainView.index, name='index'),

    # ✅ Authentication
    path('UserRegister/', mainView.UserRegister, name='UserRegister'),
    path('UserLogin/', mainView.UserLogin, name='UserLogin'),
    path('AdminLogin/', mainView.AdminLogin, name='AdminLogin'),

    # ✅ User Actions
    path('UserRegisterActions/', usr.UserRegisterActions, name='UserRegisterActions'),
    path('UserLoginCheck/', usr.UserLoginCheck, name='UserLoginCheck'),
    path('UserHome/', usr.UserHome, name='UserHome'),
    path('training/', usr.training, name='training'),
    path('prediction/', usr.prediction, name='prediction'),

    # ✅ Admin Actions
    path('AdminLoginCheck/', admins.AdminLoginCheck, name='AdminLoginCheck'),
    path('AdminHome/', admins.AdminHome, name='AdminHome'),
    path('ViewRegisteredUsers/', admins.ViewRegisteredUsers, name='ViewRegisteredUsers'),
    path('AdminActivaUsers/', admins.AdminActivaUsers, name='AdminActivaUsers'),

    # ✅ ML Prediction
    path('predict/', predict, name='predict'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)