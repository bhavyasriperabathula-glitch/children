from django.shortcuts import render
from django.http import JsonResponse
from users.forms import UserRegistrationForm
import random

# Create your views here.
def index(request):
    return render(request, 'index.html', {})

def logout(request):
    return render(request, 'index.html', {})

def UserLogin(request):
    return render(request, 'UserLogin.html', {})

def UserRegister(request):
    form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})

def home(request):
    return JsonResponse({"message": "Server is running"})
    
def predict(request):
    result = random.choice(["ADHD", "Normal"])
    return JsonResponse({"prediction": result})  