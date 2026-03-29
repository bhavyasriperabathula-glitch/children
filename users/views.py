import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from django.conf import settings
from django.shortcuts import render
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from scipy.stats import entropy
import math

IMG_SIZE = 48


# ---------------- IMAGE THRESHOLDING ----------------

def apply_thresholding(image):
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
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


# ---------------- USER LOGIN ----------------

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')

        print("Login ID = ", loginid, 'Password = ', pswd)

        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status

            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email

                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not Activated')
                return render(request, 'UserLogin.html')

        except Exception as e:
            print('Exception is ', str(e))

        messages.success(request, 'Invalid Login id and password')

    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


# ---------------- MODEL TRAINING ----------------

def training(request):

    def load_images_from_directory(directory, max_per_class=None):

        images, labels = [], []
        label_map = {'ADHD-Hyperactive': 0, 'Typically Developing Children': 1}

        for class_name in os.listdir(directory):

            class_folder = os.path.join(directory, class_name)

            if os.path.isdir(class_folder):

                print(f"\nProcessing class: {class_name}")

                count = 0

                for filename in tqdm(os.listdir(class_folder), desc=class_name):

                    if filename.endswith((".jpg", ".png")):

                        try:

                            img_path = os.path.join(class_folder, filename)

                            img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                            img = img_to_array(img) / 255.0

                            img = apply_thresholding(img)

                            images.append(img)
                            labels.append(label_map[class_name])

                            count += 1

                            if max_per_class and count >= max_per_class:
                                break

                        except Exception as e:
                            print(f"Error processing {filename}: {e}")

        return np.array(images), np.array(labels)


    # LOAD DATASET

    xtrain, ytrain = load_images_from_directory(r"media\train", max_per_class=100)
    xval, yval = load_images_from_directory(r"media\val", max_per_class=100)
    xtest, ytest = load_images_from_directory(r"media\test", max_per_class=100)


    # BUILD MODEL

    base_model = VGG19(weights='imagenet', include_top=False,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3))

    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=3,
                                   restore_best_weights=True)


    # TRAIN MODEL

    history = model.fit(
        xtrain,
        ytrain,
        batch_size=32,
        epochs=10,
        validation_data=(xval, yval),
        callbacks=[early_stopping]
    )


    # SAVE TRAINED MODEL
    model.save(os.path.join(settings.MEDIA_ROOT, "adhd_model.h5"))


    # EVALUATE MODEL

    test_loss, test_accuracy = model.evaluate(xtest, ytest)


    # PREDICTIONS

    y_pred = model.predict(xtest)
    y_pred_labels = (y_pred > 0.5).astype(int).flatten()


    # CONFUSION MATRIX

    cm = confusion_matrix(ytest, y_pred_labels)

    plt.figure(figsize=(6, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ADHD-Hyperactive', 'Typically Developing Children'],
                yticklabels=['ADHD-Hyperactive', 'Typically Developing Children'],
                cbar=False)

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    plot_path = os.path.join(settings.MEDIA_ROOT, 'confusion_matrix.png')

    plt.savefig(plot_path)
    plt.close()


    final_train_acc = round(history.history['accuracy'][-1] * 100, 2)
    final_val_acc = round(history.history['val_accuracy'][-1] * 100, 2)


    return render(request, 'users/training.html', {
        'accuracy': final_train_acc,
        'val_accuracy': final_val_acc,
        'conf_matrix_path': 'confusion_matrix.png'
    })


# ---------------- FEATURE EXTRACTION ----------------

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Statistical features
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    var_val = np.var(gray)
    
    # Entropy calculation
    # Flatten the image and calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    ent_val = entropy(hist_norm, base=2)
    
    features = {
        'Mean Intensity': round(float(mean_val), 4),
        'Standard Deviation': round(float(std_val), 4),
        'Variance': round(float(var_val), 4),
        'Entropy (Shannon)': round(float(ent_val), 4),
    }
    
    return features


# ---------------- PREDICTION WITH OOD DETECTION ----------------

def prediction(request):

    context = {}

    model_path = os.path.join(settings.MEDIA_ROOT, "adhd_model.h5")

    if not os.path.exists(model_path):
        context['predicted_class'] = "Model not trained yet"
        return render(request, 'users/prediction.html', context)

    model = load_model(model_path)

    if request.method == 'POST' and request.FILES.get('image'):

        image_file = request.FILES['image']

        image_path = os.path.join(settings.MEDIA_ROOT, image_file.name)

        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        # ---------- MRI IMAGE VALIDATION ----------

        img_cv = cv2.imread(image_path)

        if img_cv is None:
            context['predicted_class'] = "Invalid Image"
            return render(request, 'users/prediction.html', context)

        # split color channels
        b, g, r = cv2.split(img_cv)

        # calculate channel difference
        rg_diff = np.mean(np.abs(r - g))
        rb_diff = np.mean(np.abs(r - b))
        gb_diff = np.mean(np.abs(g - b))

        color_difference = (rg_diff + rb_diff + gb_diff) / 3

        # Relaxed check: allow colorful images if they are spectrograms (common in EEG)
        # We only warn instead of blocking, or just skip if it's likely a real image
        is_mri = color_difference <= 15
        
        # ---------- FEATURE EXTRACTION ----------
        features = extract_features(image_path)
        # context['features'] = features

        # ---------- ADHD PREDICTION ----------

        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img = img_to_array(img) / 255.0
        img = apply_thresholding(img)
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0][0]

        if pred < 0.5:
            context['predicted_class'] = "ADHD-Hyperactive"
        else:
            context['predicted_class'] = "Typically Developing Child"

        if not is_mri:
            context['warning'] = "Note: This image appears to be a color EEG spectrogram rather than a grayscale MRI, but it has been processed normally."

        context['image_url'] = f"/media/{image_file.name}"
    return render(request, 'users/prediction.html', context)

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        file = request.FILES.get('file')

        if not file:
            return JsonResponse({"error": "No file found"})

        # Create media folder if not exists
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

        file_path = os.path.join(settings.MEDIA_ROOT, file.name)

        # Save file
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        return JsonResponse({
            "message": "File uploaded successfully",
            "filename": file.name
        })

    return JsonResponse({"error": "Invalid request"})

@csrf_exempt
def predict_api(request):
    if request.method == 'GET':
        filename = request.GET.get('filename')
        import glob
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        
        if filename:
            image_path = os.path.join(settings.MEDIA_ROOT, filename)
            if not os.path.exists(image_path):
                files = glob.glob(os.path.join(settings.MEDIA_ROOT, f"*{filename}*"))
                if files:
                    image_path = files[0]
                else:
                    return JsonResponse({"error": f"File {filename} not found", "prediction": "Error"})
        else:
            files = glob.glob(os.path.join(settings.MEDIA_ROOT, '*'))
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                return JsonResponse({"error": "No image found", "prediction": "No Image"})
            image_path = max(image_files, key=os.path.getctime)
        
        model_path = os.path.join(settings.MEDIA_ROOT, "adhd_model.h5")
        if not os.path.exists(model_path):
            return JsonResponse({"error": "Model missing", "prediction": "Model Missing"})
            
        try:
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            img = load_img(image_path, target_size=(48, 48))
            img = img_to_array(img) / 255.0
            img = apply_thresholding(img)
            img = np.expand_dims(img, axis=0)
            pred = model.predict(img)[0][0]
            result = "ADHD-Hyperactive" if pred < 0.5 else "Typically Developing Child"
            return JsonResponse({"prediction": result, "filename": os.path.basename(image_path)})
        except Exception as e:
            return JsonResponse({"error": str(e), "prediction": "Error"})
    
    return JsonResponse({"error": "Invalid request"})