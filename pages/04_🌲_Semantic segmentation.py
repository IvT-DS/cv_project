# Contents of ~/my_app/pages/page_3.py
import streamlit as st
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
import requests
import time

from ultralytics import YOLO
from io import BytesIO
from requests.models import MissingSchema

from PIL import Image
from torchvision import transforms as T
from Models.semantic_segmentation_model import UNet

# Specify the paths, where a model and weights are located
# model_path = "Models"
# weights_path = "Models/unet_weights.pt"

# Save weight for each layer
# for name, layer in model.named_children():
#     torch.save(layer.state_dict(), f"weights/{name}_weights.pth")


# Use decorator to cache a model using
@st.cache_resource
def load_model():
    model = UNet()  # load a model
    # Choose device
    device = torch.device("cpu")
    # Set weights
    for name, layer in model.named_children():
        layer_weights = torch.load(f"Models/{name}_weights.pth", map_location=device)
        layer.load_state_dict(layer_weights)

    # model.load_state_dict(torch.load(weights_path, map_location=device))

    # Set the model to evaluation mode
    model.eval()

    return model


st.sidebar.markdown("## Используй навигацию между страницами выше ⬆️")
st.sidebar.markdown("# Semantic segmentation page -->>")

st.markdown("## Семантическая сегментация леса на аэрокосмических снимках 🌲")

st.markdown("##### Информация о модели, обучении и метриках:")
st.markdown(
    """
- Модель U-net состоит из 10 слоев: 4 upscaling, 4 downscaling, bottleneck, выходной слой.
- Число эпох обучения: 1;
- Объем обучающей выборки: 5000 изображений;
- Объем валидационной выборки: 5000 изображений.
"""
)

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
    transform = T.ToTensor()  # transformator to tensor
    to_pil = T.ToPILImage()  # transformator to image
    if image_source == "File":
        uploaded_files = st.file_uploader(
            "Upload the image",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
        )

# # Main code block
# uploaded_files = st.file_uploader(
#     "Upload the image", type=["jpg", "png", "jpeg"], accept_multiple_files=True
# )

        # Creating two columns on the main page
        col1, col2 = st.columns(2)


        # Load images
        if uploaded_files:
            with col1:
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded image", use_column_width=True)

            # Clean images if the button was pressed
            if st.button(f"Determine forest"):

                overall_elapsed_time = 0
                try:
                    # transform = T.ToTensor()  # transformator to tensor
                    # to_pil = T.ToPILImage()  # transformator to image

                    model = load_model()

                    try:
                        with col2:
                            for uploaded_file in uploaded_files:
                                image = Image.open(uploaded_file)
                                # The start of the countdown of the model's operation
                                start_time = time.time()
                                # Transform an image to tensor
                                input_tensor_image = transform(image).unsqueeze(0)
                                # Start prediction of a model
                                with torch.no_grad():
                                    pred = model(input_tensor_image)

                                    pred[pred < 0.5] = 0
                                    pred[pred >= 0.5] = 1

                                    mask_data = pred.squeeze(0).permute(1, 2, 0).numpy()
                                    mask_data_rgb = np.repeat(mask_data, 3, axis=-1)
                                    plt.imsave("mask.jpg", mask_data_rgb)

                                    # image = Image.open()
                                    mask = Image.open("mask.jpg").convert("L")
                                    mask = mask.resize(image.size, Image.LANCZOS)
                                    new_color = (0, 255, 0)

                                    colored_mask = Image.new("RGB", mask.size, new_color)

                                    alpha_mask = mask.point(lambda p: p > 0 and 128)

                                    transparent_image = Image.new(
                                        "RGBA", mask.size, (0, 0, 0, 0)
                                    )
                                    transparent_image.paste(colored_mask, (0, 0), alpha_mask)

                                    overlayed_image = Image.alpha_composite(
                                        image.convert("RGBA"), transparent_image
                                    )

                                    # Transform a tensor to an image
                                    # ready_pil_image = to_pil(ready_image.squeeze(0))

                                # The end of the countdown of the model
                                end_time = time.time()
                                # The working time of the model in 1 image
                                elapsed_time = end_time - start_time
                                # The total working time of the model in all images
                                overall_elapsed_time += elapsed_time
                                # Show cleaned images
                                st.image(
                                    overlayed_image,
                                    caption="Ready image",
                                    use_column_width=True,
                                )
                        st.info(
                            f"The working time of the model in all images is: {overall_elapsed_time:.4f} sec."
                        )
                    except Exception as ex:
                        st.error(f"The model cannot be applied. Check the settings.")
                        st.error(ex)

                except Exception as ex:
                    st.error(
                        f"The model cannot be loaded. Check the paths: {model_path}, {weights_path}"
                    )
                    st.error(ex)

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
            # # Start prediction of a model
            # result = model.predict(image)
            # Transform an image to tensor
            input_tensor_image = transform(image).unsqueeze(0)
            # Start prediction of a model
            with torch.no_grad():
                pred = model(input_tensor_image)

                pred[pred < 0.5] = 0
                pred[pred >= 0.5] = 1

                mask_data = pred.squeeze(0).permute(1, 2, 0).numpy()
                mask_data_rgb = np.repeat(mask_data, 3, axis=-1)
                plt.imsave("mask.jpg", mask_data_rgb)

                # image = Image.open()
                mask = Image.open("mask.jpg").convert("L")
                mask = mask.resize(image.size, Image.LANCZOS)
                new_color = (0, 255, 0)

                colored_mask = Image.new("RGB", mask.size, new_color)

                alpha_mask = mask.point(lambda p: p > 0 and 128)

                transparent_image = Image.new(
                    "RGBA", mask.size, (0, 0, 0, 0)
                )
                transparent_image.paste(colored_mask, (0, 0), alpha_mask)

                overlayed_image = Image.alpha_composite(
                    image.convert("RGBA"), transparent_image
                )    



            # The end of the countdown of the model
            end_time = time.time()
            # The working time of the model in 1 image
            elapsed_time = end_time - start_time
            # Get the coordinates of the frames detected in the image by the model.
            # boxes = result[0].boxes
            # # An image is created with frames drawn on it.
            # result_plotted = result[0].plot()[:, :, ::-1]

            with col2:
                # Show an image via streamlit
                st.image(
                    overlayed_image, caption="Detected image", use_column_width=True
                )
            st.info(f"The working time of the model is: {elapsed_time:.4f} sec.")


except Exception as e:
    st.error(f"An error occurred while processing the image {str(e)}")
