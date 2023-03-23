import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Define the layout of the app
st.set_page_config(page_title="BinSpy", page_icon=":camera:")


st.header("BinSpy: An AI Model for Accurate Sorting of Recyclable Items.")
st.write("Try clicking an item image and watch how an AI Model will detect its label.")

st.caption(
    "The application will infer the one label out of 6 labels, as follows: Glass, Metal, Paper, Plastic, Trash, Carboard."
)

st.caption(
    "Warning: Do not click Submit Image button before uploading/clicking a image. It will result in error."
)

with st.sidebar:
    st.header("BinSpy")
    img = Image.open("./Images/recycycling_bins.jpg")
    st.image(img)
    st.subheader("About BinSpy")
    st.write(
        "BinSpy is an advanced AI model built, trained, and exported on Teachable Machine. It is designed to accurately sort various recyclable items such as plastic bottles, cans, paper, and cardboard, among others. BinSpy utilizes state-of-the-art machine learning algorithms to analyze and classify different materials, ensuring proper sorting and disposal."
    )

    st.write(
        "The AI model is easy to install and can be integrated with existing recycling systems, making the sorting process more efficient and effective. BinSpy is equipped with a user-friendly interface that allows recycling plant operators to monitor the sorting process in real-time. This helps to minimize contamination, reduce waste, and increase the recovery of valuable materials."
    )


camera_image = st.camera_input(
    label="Click the image of the item you have",
    label_visibility="visible",
)


if camera_image is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")


def detect_recyclable_item(img_path):

    best_model = load_model("./keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(img_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = best_model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    Detection_Result = f"The model has detected {class_name[2:]}, with Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%."
    return Detection_Result, prediction


submit = st.button(label="Submit Item Image")
if submit:
    st.subheader("Output")
    classified_label, prediction = detect_recyclable_item(camera_image)
    with st.spinner(text="This may take a moment..."):
        st.write(classified_label)

        class_names = open("labels.txt", "r").readlines()

        data = {
            "Class": class_names,
            "Confidence Score": prediction[0],
        }

        df = pd.DataFrame(data)

        df["Confidence Score"] = df["Confidence Score"].apply(
            lambda x: f"{str(np.round(x*100))[:-2]}%"
        )

        df["Class"] = df["Class"].apply(lambda x: x.split(" ")[1])

        st.subheader("Confidence Scores on other classes:")
        st.write(df)

footer = """
<div style="text-align: center; font-size: medium; margin-top:50px;">
    If you find BinSpy useful or interesting, please consider starring it on GitHub.
    <hr>
    <a href="https://github.com/SaiJeevanPuchakayala/BinSpy" target="_blank">
    <img src="https://img.shields.io/github/stars/SaiJeevanPuchakayala/BinSpy.svg?style=social" alt="GitHub stars">
  </a>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
