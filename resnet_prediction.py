import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

label_map = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevi',
    5: 'Melanoma',
    6: 'Vascular lesions'
}

model = tf.keras.models.load_model("resnet_model.h5")

if 'history_df' not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=["Image", "Predicted Label", "Confidence"])

def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict(image):
    if image is None:
        return None, None, None
    img_array = load_and_preprocess_image(image)
    probabilities = model.predict(img_array)[0]
    predicted_label = np.argmax(probabilities)
    confidence = probabilities[predicted_label]
    return label_map[predicted_label], confidence, probabilities

st.title("Skin Cancer Prediction")

# Sidebar
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

# Button
make_prediction_btn = st.sidebar.button("Make Prediction")
show_probabilities_btn = st.sidebar.button("Show Probabilities")
show_history_btn = st.sidebar.button("Show Prediction History")

if uploaded_file and not (make_prediction_btn or show_probabilities_btn or show_history_btn):
    # Nếu chưa nhấn nút, hiển thị ảnh
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Xử lý sự kiện khi nhấn các nút
if uploaded_file:
    image = Image.open(uploaded_file)

    # Nếu nhấn nút "Make Prediction"
    if make_prediction_btn:
        predicted_label, confidence, probabilities = predict(image)
        st.markdown("## Prediction Results")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.markdown(f"**Predicted Label:** {predicted_label}")
        st.markdown(f"**Confidence:** {confidence:.2f}")


        new_entry = {
            "Image": uploaded_file.name,
            "Predicted Label": predicted_label,
            "Confidence": f"{confidence:.2f}"
        }
        st.session_state.history_df = pd.concat(
            [st.session_state.history_df, pd.DataFrame([new_entry])],
            ignore_index=True
        )

    # Nếu nhấn nút "Show Probabilities"
    elif show_probabilities_btn:
        _, _, probabilities = predict(image)
        probabilities_df = pd.DataFrame({
            "Class": list(label_map.values()),
            "Probability": probabilities
        }).sort_values(by="Probability", ascending=False)

        st.markdown("## Probabilities for Each Class")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(probabilities_df["Class"], probabilities_df["Probability"], color="skyblue")
        ax.set_xlabel("Probability")
        ax.set_ylabel("Class")
        ax.set_title("Class Probabilities")
        for i, (prob, label) in enumerate(zip(probabilities_df["Probability"], probabilities_df["Class"])):
            ax.text(prob + 0.01, i, f"{prob:.2f}", va="center")
        st.pyplot(fig)

    # Nếu nhấn nút "Show Prediction History"
    elif show_history_btn:
        st.markdown("## Prediction History")
        if st.session_state.history_df.empty:
            st.write("No predictions have been made yet.")
        else:
            st.dataframe(st.session_state.history_df)

else:
    st.write("Please upload an image to proceed.")