# Contents of ~/my_app/pages/page_3.py

# Import of libraries
import streamlit as st
import time
import torch
from requests.models import MissingSchema
from PIL import Image
from torchvision import transforms as T
from Models.denoiser_model import DenoiseEncoder

# Specify the paths, where a model and weights are located
model_path = "Models"
weights_path = "Models/best_denoiser_weights.pt"

# Use decorator to cache a model using
@st.cache_resource
def load_model():
    model = DenoiseEncoder()  # load a model
    # Choose device
    device = torch.device('cpu')
    # Set weights
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # Set the model to evaluation mode
    model.eval()

    return model

# Sidebar of the page
st.sidebar.markdown("## Используй навигацию между страницами выше ⬆️")
st.sidebar.markdown("# Autoencoder page -->>")

# Main area of the page
st.markdown("## Очищение документов от шума с помощью автоэнкодера 📖")
st.markdown("##### Информация о модели, обучении и метриках:")
st.markdown(
    """
- Модель-автоэнкодер состоит из энкодера и декодера: 
    - в энкодере применены 3 слоя, свертки, батч-нормализация, дропауты, функции активации SELU и Tanh;
    - в декодере применены 3 слоя, свертки, батч-нормализация, дропауты, функции активации SELU, LeakyReLU и Sigmoid.
- Число эпох обучения: ~50;
- Объем обучающей выборки: 981 изображений;
- Объем валидационной выборки: 290 изображений;
- Метрики (RMSE):
"""
)

# Metrics
metrics_img_1 = Image.open("Metrics/encoder_loss_rmse.png")
metrics_img_2 = Image.open("Metrics/encoder_rmse_digits.png")
metrics_img_3 = Image.open("Metrics/encoder_loss_digits.png")

st.write("Graphics")
st.image(metrics_img_1, use_column_width=True)

col1, col2 = st.columns((1, 3))  # Set width col1 at 1 & col2 at 3
with col1:
    st.write("RMSE")
    st.image(metrics_img_2, use_column_width=True)
    st.write("LOSS")
    st.image(metrics_img_3, use_column_width=True)


# Separator
st.write("---")

# Main code block
uploaded_files = st.file_uploader(
    "Upload the image", type=["jpg", "png", "jpeg"], accept_multiple_files=True
)

# Creating two columns on the main page
col1, col2 = st.columns(2)


# Function for check if uploaded image in grayscale
def is_grayscale(img_path):
    img = Image.open(img_path)
    if img.mode == "L" or "LA" in img.mode:
        return True
    else:
        return False


# Load images
if uploaded_files:
    with col1:
        for uploaded_file in uploaded_files:
            if is_grayscale(uploaded_file):  # Check if image is in grayscale
                image = Image.open(uploaded_file)
                st.image(image, caption="Dirty image", use_column_width=True)
            else:
                st.error("Uploaded file is not a grayscale image")

    # Clean images if the button was pressed
    if st.button(f"Clean images"):

        overall_elapsed_time = 0
        try:
            transform = T.ToTensor()  # transformator to tensor
            to_pil = T.ToPILImage()  # transformator to image

            model = load_model()

            # model = DenoiseEncoder()  # load a model

            # # Choose device
            # device = torch.device('cpu')

            # # Set weights
            # model.load_state_dict(torch.load(weights_path, map_location=device))

            # # Set the model to evaluation mode
            # model.eval()

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
                            cleaned_image = model(input_tensor_image)
                            # Transform a tensor to an image
                            cleaned_pil_image = to_pil(cleaned_image.squeeze(0))
                        # The end of the countdown of the model
                        end_time = time.time()
                        # The working time of the model in 1 image
                        elapsed_time = end_time - start_time
                        # The total working time of the model in all images
                        overall_elapsed_time += elapsed_time
                        # Show cleaned images
                        st.image(
                            cleaned_pil_image,
                            caption="Clean image",
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
