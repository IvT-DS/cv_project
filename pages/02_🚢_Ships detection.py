# Contents of ~/my_app/pages/page_2.py
# import os

# Import of libraries
import streamlit as st
import requests
import time
from ultralytics import YOLO
from io import BytesIO
from PIL import Image

# Specify the paths, where a model and weights are located
weights_path = "Models/best_yolo_weights.pt"

# Use decorator to cache a model using
@st.cache_resource
def load_model():
    return YOLO(weights_path)

# Sidebar of the page
st.sidebar.markdown("## Используй навигацию между страницами выше ⬆️")
st.sidebar.markdown("# YOLOv8 page -->>")

# Main area of the page
st.markdown("### Детекция кораблей на аэрокосмических снимках с помощью YOLOv8 🚢")
st.markdown("##### Информация о модели:")
st.markdown(
    """
- Используемая модель - YOLOv8m: 
"""
)
comparison_img = Image.open("Metrics/yolo_comparison.png")

col1, col2 = st.columns((2, 1))  # Set width col1 at 1 & col2 at 3
with col1:
    st.image(comparison_img, use_column_width=True)

st.markdown(
    """
- Параметры:
    - batch size - 8;
    - cache -True;
    - epochs - 80, 20;
    - device - gpu.
- Число эпох обучения: 80 -> 20;
- Объем обучающей выборки: 9697 изображений;
- Объем валидационной выборки: 2165 изображений;
- Метрики (mAP, PR-AUC, confusion matrix):
"""
)

# Metrics
metrics_img_results_1 = Image.open("Metrics/yolo_1_results.png")
metrics_img_results_2 = Image.open("Metrics/yolo_2_results.png")
# metrics_img_results_3 = Image.open("Metrics/yolo_3_results.png")

metrics_img_PR_1 = Image.open("Metrics/yolo_1_PR_curve.png")
metrics_img_PR_2 = Image.open("Metrics/yolo_2_PR_curve.png")
# metrics_img_PR_3 = Image.open("Metrics/yolo_3_PR_curve.png")

metrics_img_matrix_1 = Image.open("Metrics/yolo_1_confusion_matrix_normalized.png")
metrics_img_matrix_2 = Image.open("Metrics/yolo_2_confusion_matrix_normalized.png")
# metrics_img_matrix_3 = Image.open("Metrics/yolo_3_confusion_matrix_normalized.png")

yolo_img_train_batch_1 = Image.open("Metrics/yolo_1_train_batch0.jpg")
yolo_img_train_batch_2 = Image.open("Metrics/yolo_1_train_batch84840.jpg")

yolo_img_val_labels = Image.open("Metrics/yolo_1_val_batch0_labels.jpg")
yolo_img_val_pred = Image.open("Metrics/yolo_1_val_batch0_pred.jpg")


st.markdown("###### 1st training cycle")
st.image(metrics_img_results_1, use_column_width=True)

st.markdown("###### 2nd training cycle")
st.image(metrics_img_results_2, use_column_width=True)

# st.markdown("###### 3rd training cycle")
# st.image(metrics_img_results_3, use_column_width=True)

col1, col2 = st.columns(2)

with col1:
    st.write("PR curve:")
    st.image(metrics_img_PR_1, use_column_width=True)

with col2:
    st.write("Diffusion matrix:")
    st.image(metrics_img_matrix_1, use_column_width=True)

# Separator
st.write("---")


button_style = """
    <style>
    .center-align {
        display: flex;
        justify-content: center;
    }
    </style>
    """

# Menu for choosing how to upload an image: by link or from a file
image_source = st.radio("Choose the option of uploading the image:", ("File", "URL"))

# Main code block
try:
    if image_source == "File":
        uploaded_files = st.file_uploader(
            "Upload the image",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
        )
        # Creating two columns on the main page
        col1, col2 = st.columns(2)
        # Load images
        if uploaded_files:
            with col1:
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded image", use_column_width=True)

            try:
                # model = YOLO(weights_path)
                model = load_model()

            except Exception as ex:
                st.error(
                    f"Unable to load model. Check the specified path: {weights_path}"
                )
                st.error(ex)

            # Clean images if the button was pressed
            if st.button(f"Dedect ships"):

                overall_elapsed_time = 0

                with col2:
                    for uploaded_file in uploaded_files:
                        image = Image.open(uploaded_file)
                        # The start of the countdown of the model's operation
                        start_time = time.time()
                        # Start prediction of a model
                        result = model.predict(image)
                        # The end of the countdown of the model
                        end_time = time.time()
                        # The working time of the model in 1 image
                        elapsed_time = end_time - start_time
                        # The total working time of the model in all images
                        overall_elapsed_time += elapsed_time
                        # Get the coordinates of the frames detected in the image by the model.
                        boxes = result[0].boxes
                        # An image is created with frames drawn on it.
                        result_plotted = result[0].plot()[:, :, ::-1]
                        # Show an image via streamlit
                        st.image(
                            result_plotted,
                            caption="Detected image",
                            use_column_width=True,
                        )
                st.info(
                    f"The working time of the model in all images is: {overall_elapsed_time:.4f} sec."
                )
    # If URL was chosen
    else:
        url = st.text_input("Enter the URL of image...")
        # Creating two columns on the main page
        col1, col2 = st.columns(2)
        # Adding image to the first column if image is uploaded
        with col1:
            if url:
                response = requests.get(url)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    st.image(image, caption="Uploaded image", use_column_width=True)

                else:
                    st.error(
                        "An error occurred while receiving the image. Make sure that the correct link is entered."
                    )
        try:
            # model = YOLO(weights_path)
            model = load_model()

        except Exception as ex:
            st.error(f"Unable to load model. Check the specified path: {weights_path}")
            st.error(ex)

        if st.button(f"Dedect ships"):
            # The start of the countdown of the model's operation
            start_time = time.time()
            # Start prediction of a model
            result = model.predict(image)
            # The end of the countdown of the model
            end_time = time.time()
            # The working time of the model in 1 image
            elapsed_time = end_time - start_time
            # Get the coordinates of the frames detected in the image by the model.
            boxes = result[0].boxes
            # An image is created with frames drawn on it.
            result_plotted = result[0].plot()[:, :, ::-1]

            with col2:
                # Show an image via streamlit
                st.image(
                    result_plotted, caption="Detected image", use_column_width=True
                )
            st.info(f"The working time of the model is: {elapsed_time:.4f} sec.")


except Exception as e:
    st.error(f"An error occurred while processing the image {str(e)}")
