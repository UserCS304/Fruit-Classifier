import streamlit as st
import joblib
from PIL import Image
import numpy as np
import cv2  # <--- FIXED: Must import cv2 if you use cv2 functions

# --- Load Model ---
try:
    with open("svm_image_classifier_model.pkl", "rb") as f:
        model = joblib.load(f)
except FileNotFoundError:
    st.error("Error: Model file 'svm_image_classifier_model.pkl' not found. Please run the training steps first.")
    model = None

# Define the classification dictionary 
class_dict = {
    0: "แอปเปิ้ล (Apple)", 
    1: "ส้ม (Orange)"
}

# --- UI Layout ---
st.title("Fruit Classifier 🍎🍊")
st.write("อัปโหลดรูปภาพเพื่อทำนายว่าเป็น แอปเปิ้ล หรือ ส้ม") 

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "png", "jpeg"]
)

# --- Prediction Logic ---
if uploaded_file is not None and model is not None:
    # Load and Display Image
    # Convert("RGB") is important for consistency
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Predict"):
        # --- Preprocess ---

        # Convert PIL Image (RGB) to numpy array
        image_array = np.array(image)

        # Convert RGB array to BGR format if the model was trained using cv2.imread (BGR)
        # BGR is the standard for OpenCV loading
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR) # <--- FIXED: Corrected GBR to BGR

        # Resize image to 100x100
        image_resized = cv2.resize(image_array, (100, 100))

        # Flatten to 1D and reshape for model input
        image_flatten = image_resized.flatten().reshape(1, -1)

        # --- Predict ---
        try:
            prediction = model.predict(image_flatten)[0]
            prediction_name = class_dict[prediction]

            st.success(f"**ผลการทำนาย (Prediction Result):** **{prediction_name}**")
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
