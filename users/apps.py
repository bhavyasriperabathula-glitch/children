import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("adhd_model.h5")

# Prediction function
def predict(image):
    try:
        # Resize image (change size if your model uses different input)
        img = image.resize((224, 224))
        
        # Convert to array
        img = np.array(img)
        
        # Normalize
        img = img / 255.0
        
        # Expand dimensions
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img)

        # Output result
        if prediction[0][0] > 0.5:
            return "ADHD Detected"
        else:
            return "No ADHD Detected"

    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="ADHD Detection System",
    description="Upload an image to detect ADHD using ML model"
)

# Run app
interface.launch()