import os
import numpy as np
from django.conf import settings
from django.shortcuts import render
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

IMG_SIZE = 48


# ---------------- IMAGE THRESHOLDING ----------------
def apply_thresholding(image):
    import cv2
    image = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholded_rgb = np.stack([thresholded] * 3, axis=-1)
    return thresholded_rgb


# ---------------- USER REGISTRATION ----------------
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Registered successfully')
        else:
            messages.success(request, 'Email or Mobile Already Exists')
    else:
        form = UserRegistrationForm()

    return render(request, 'UserRegistrations.html', {'form': form})


# ---------------- USER LOGIN ----------------
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')

        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)

            if check.status == "activated":
                request.session['loggeduser'] = check.name
                return render(request, 'users/UserHome.html')
            else:
                messages.success(request, 'Account not activated')

        except:
            messages.success(request, 'Invalid Login')

    return render(request, 'UserLogin.html')


def UserHome(request):
    return render(request, 'users/UserHome.html')


# ---------------- TRAINING ----------------
def training(request):
    try:
        import cv2
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from tqdm import tqdm
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        from tensorflow.keras.applications import VGG19
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.metrics import confusion_matrix

        def load_images(directory):
            images, labels = [], []
            label_map = {'ADHD-Hyperactive': 0, 'Typically Developing Children': 1}

            for class_name in os.listdir(directory):
                path = os.path.join(directory, class_name)
                if os.path.isdir(path):
                    for file in os.listdir(path):
                        if file.endswith((".jpg", ".png")):
                            img = load_img(os.path.join(path, file), target_size=(IMG_SIZE, IMG_SIZE))
                            img = img_to_array(img) / 255.0
                            img = apply_thresholding(img)
                            images.append(img)
                            labels.append(label_map[class_name])

            return np.array(images), np.array(labels)

        xtrain, ytrain = load_images("media/train")
        xval, yval = load_images("media/val")
        xtest, ytest = load_images("media/test")

        base_model = VGG19(weights='imagenet', include_top=False,
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(xtrain, ytrain, epochs=3, validation_data=(xval, yval))

        model.save(os.path.join(settings.MEDIA_ROOT, "adhd_model.h5"))

        y_pred = (model.predict(xtest) > 0.5).astype(int)
        cm = confusion_matrix(ytest, y_pred)

        plt.figure()
        sns.heatmap(cm, annot=True)
        plt.savefig(os.path.join(settings.MEDIA_ROOT, 'cm.png'))

        return render(request, 'users/training.html', {'accuracy': history.history['accuracy'][-1]})

    except Exception as e:
        return render(request, 'users/training.html', {'error': str(e)})


# ---------------- PREDICTION ----------------
def prediction(request):
    context = {}
    model_path = os.path.join(settings.MEDIA_ROOT, "adhd_model.h5")

    if not os.path.exists(model_path):
        context['predicted_class'] = "Model not trained"
        return render(request, 'users/prediction.html', context)

    if request.method == 'POST' and request.FILES.get('image'):
        try:
            import cv2
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.image import load_img, img_to_array

            model = load_model(model_path)

            image = request.FILES['image']
            path = os.path.join(settings.MEDIA_ROOT, image.name)

            with open(path, 'wb+') as f:
                for chunk in image.chunks():
                    f.write(chunk)

            img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
            img = img_to_array(img) / 255.0
            img = apply_thresholding(img)
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)[0][0]

            context['predicted_class'] = "ADHD" if pred < 0.5 else "Normal"
            context['image_url'] = f"/media/{image.name}"

        except Exception as e:
            context['predicted_class'] = f"Error: {e}"

    return render(request, 'users/prediction.html', context)


# ---------------- API ----------------
@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({"error": "No file"})

        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        path = os.path.join(settings.MEDIA_ROOT, file.name)

        with open(path, 'wb+') as f:
            for chunk in file.chunks():
                f.write(chunk)

        return JsonResponse({"message": "Uploaded"})

    return JsonResponse({"error": "Invalid"})


@csrf_exempt
def predict_api(request):
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.image import load_img, img_to_array

        model_path = os.path.join(settings.MEDIA_ROOT, "adhd_model.h5")
        if not os.path.exists(model_path):
            return JsonResponse({"error": "Model missing"})

        files = os.listdir(settings.MEDIA_ROOT)
        if not files:
            return JsonResponse({"error": "No image"})

        image_path = os.path.join(settings.MEDIA_ROOT, files[-1])

        model = load_model(model_path)
        img = load_img(image_path, target_size=(48, 48))
        img = img_to_array(img) / 255.0
        img = apply_thresholding(img)
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0][0]
        result = "ADHD" if pred < 0.5 else "Normal"

        return JsonResponse({"prediction": result})

    except Exception as e:
        return JsonResponse({"error": str(e)})