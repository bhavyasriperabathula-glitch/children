import os
import numpy as np
from django.conf import settings
from django.shortcuts import render
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel

IMG_SIZE = 48

# ---------------- SAFE IMPORT FUNCTION ----------------
def load_model_safe():
    from tensorflow.keras.models import load_model
    model_path = os.path.join(settings.MEDIA_ROOT, "adhd_model.h5")
    if os.path.exists(model_path):
        return load_model(model_path)
    return None


# ---------------- USER REGISTER ----------------
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Registered successfully')
        else:
            messages.error(request, 'User already exists')
    else:
        form = UserRegistrationForm()

    return render(request, 'UserRegistrations.html', {'form': form})


# ---------------- LOGIN ----------------
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')

        try:
            user = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)

            if user.status == "activated":
                request.session['loggeduser'] = user.name
                return render(request, 'users/UserHome.html')
            else:
                messages.error(request, 'Not activated')
        except:
            messages.error(request, 'Invalid login')

    return render(request, 'UserLogin.html')


def UserHome(request):
    return render(request, 'users/UserHome.html')


# ---------------- PREDICTION ----------------
def prediction(request):

    model = load_model_safe()

    if not model:
        return render(request, 'users/prediction.html', {
            'predicted_class': 'Model not trained'
        })

    if request.method == 'POST' and request.FILES.get('image'):

        from tensorflow.keras.preprocessing.image import load_img, img_to_array

        image = request.FILES['image']
        path = os.path.join(settings.MEDIA_ROOT, image.name)

        with open(path, 'wb+') as f:
            for chunk in image.chunks():
                f.write(chunk)

        img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0][0]

        result = "ADHD-Hyperactive" if pred < 0.5 else "Normal"

        return render(request, 'users/prediction.html', {
            'predicted_class': result,
            'image_url': f"/media/{image.name}"
        })

    return render(request, 'users/prediction.html')