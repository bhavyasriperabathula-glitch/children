import os
import cv2
import numpy as np
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel

IMG_SIZE = 48


# ---------------- USER REGISTER ----------------
def is_valid_medical_image(img_path):
    """
    Check if the image is likely a Brain MRI or EEG Spectrogram.
    MRI: Predominantly grayscale, specific shape.
    Spectrogram: Specific frequency patterns, high texture density.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
            
        # --- HEURISTIC 1: Grayscale Dominance (Typically Brain MRI) ---
        # Check if R, G, B channels are close to each other
        b, g, r = cv2.split(img)
        diff_rg = np.abs(r.astype(int) - g.astype(int))
        diff_gb = np.abs(g.astype(int) - b.astype(int))
        
        is_grayscale = np.mean(diff_rg) < 15 and np.mean(diff_gb) < 15
        
        # --- HEURISTIC 2: Texture/Edge Density (Typically EEG Spectrogram) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255)
        
        # --- HEURISTIC 3: Aspect Ratio & Padding ---
        # Medical scans often have black padding or specific aspect ratios
        h, w = gray.shape
        corner_pixels = [gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1]]
        has_black_padding = np.mean(corner_pixels) < 50

        # Logic: If it's grayscale (MRI) or has high frequency detail (Spectrogram)
        # and has medical-like characteristics (padding/metadata)
        if (is_grayscale and has_black_padding) or (edge_density > 0.08):
            return True
            
        return False
    except Exception as e:
        print(f"Validation Error: {e}")
        return False

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)

        if form.is_valid():
            form.save()
            messages.success(request, 'Registered successfully')
            return redirect('UserRegister')
        else:
            # Show specific errors (e.g., email already exists, password too short)
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field.capitalize()}: {error}")

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
                return redirect('UserHome')
            else:
                messages.error(request, 'Account not activated')

        except UserRegistrationModel.DoesNotExist:
            messages.error(request, 'Invalid login credentials')

    return render(request, 'UserLogin.html')


# ---------------- USER HOME ----------------
def UserHome(request):
    return render(request, 'users/UserHome.html')


import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG19

def apply_thresholding(image):
    # Matches logic in ADHD.ipynb
    image_uint8 = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Convert back to 3-channel
    thresholded_rgb = np.stack([thresholded] * 3, axis=-1)
    return thresholded_rgb / 255.0

# Global model cache to prevent reloading on every request
MODEL_PATH = os.path.join(settings.MEDIA_ROOT, 'adhd_model.h5')
_model = None

def get_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            try:
                # 🏗️ ROBUST FIX: Rebuild architecture manually to bypass Keras 3 InputLayer bug
                base_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
                _model = Sequential([
                    base_model,
                    GlobalAveragePooling2D(),
                    Dense(128, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                
                # Load weights only (much more compatible)
                # Note: if the H5 contains the full model, load_weights will automatically extract the weights
                _model.load_weights(MODEL_PATH)
                
                _model.compile(optimizer='adam', 
                               loss='binary_crossentropy', 
                               metrics=['accuracy'])
                print("✅ Model reconstructed and weights loaded successfully")
            except Exception as e:
                print(f"❌ Error during model reconstruction: {e}")
                _model = None
        else:
            print(f"⚠️ Warning: Weights file not found at {MODEL_PATH}")
    return _model


# ---------------- LIVE PREDICTION ----------------
def prediction(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        
        # Ensure media folder exists and save file
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        temp_path = os.path.join(settings.MEDIA_ROOT, image.name)
        
        with open(temp_path, 'wb+') as f:
            for chunk in image.chunks():
                f.write(chunk)

        # 🛑 IMAGE VALIDATION LAYER
        if not is_valid_medical_image(temp_path):
            messages.error(request, "Invalid Image: Only Brain MRI or EEG Spectrogram images are accepted.")
            return render(request, 'users/prediction.html', {
                'error_message': "The uploaded image does not appear to be a Brain MRI or EEG Spectrogram. Please upload a valid medical scan."
            })

        # 🧠 ML INFERENCE
        model = get_model()
        if model:
            try:
                # Preprocess (Resizing -> Normalizing -> Thresholding)
                img = load_img(temp_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = img_to_array(img) / 255.0
                img_array = apply_thresholding(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                prediction_val = model.predict(img_array)[0][0]
                
                # Binary mapping based on train.py: 
                # 0: ADHD-Hyperactive, 1: Typically Developing Children
                if prediction_val < 0.5:
                    result = "ADHD-Hyperactive"
                else:
                    result = "Typically Developing Children"
                
                return render(request, 'users/prediction.html', {
                    'predicted_class': result,
                    'image_url': f"{settings.MEDIA_URL}{image.name}",
                    'confidence': f"{ (1 - prediction_val if prediction_val < 0.5 else prediction_val) * 100:.2f}%"
                })
            except Exception as e:
                print(f"Prediction Error: {e}")
                messages.error(request, f"Model error: {e}")
        else:
            messages.error(request, "ML Model file (adhd_model.h5) is missing.")

    return render(request, 'users/prediction.html')


# ---------------- TRAINING RESULTS ----------------
def training(request):

    # Dummy results for display
    context = {
        'accuracy': '92.5%',
        'val_accuracy': '89.2%'
    }
    return render(request, 'users/training.html', context)