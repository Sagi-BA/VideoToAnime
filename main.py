import streamlit as st
from PIL import Image
import numpy as np
import cv2
import typing
from io import BytesIO
from utils.animegan import AnimeGAN
from utils.engine import Engine

def process_image(image: typing.Union[str, np.ndarray], model_name: str) -> np.ndarray:
    try:
        if isinstance(image, str):
            img_array = cv2.imread(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        else:
            img_array = image
        
        animegan = AnimeGAN(f"models/{model_name}.onnx")
        engine = Engine(custom_objects=[animegan])
        result_array = engine.run(img_array)
        return result_array
    except Exception as e:
        st.error(f"Error processing image with {model_name} model: {str(e)}")
        return None

def main():
    st.title("AnimeGAN Image Processor")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            st.image(image, caption='Original Image', use_column_width=True)

            models = ['Hayao_64', 'Hayao-60', 'Paprika_54', 'Shinkai_53']
            
            for model in models:
                st.subheader(f"Result for {model}")
                result_image = process_image(image, model)
                if result_image is not None:
                    st.image(result_image, caption=f'Processed with {model}', use_column_width=True)
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")

if __name__ == "__main__":
    main()