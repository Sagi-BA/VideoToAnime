from math import e
import os
import tempfile
import streamlit as st
from PIL import Image
import io
import numpy as np
from utils.engine import Engine
from utils.animegan import AnimeGAN

def save_uploaded_file(uploaded_file):
    # Get the file extension from the uploaded file
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    # Create a temporary file with the correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tfile:
        tfile.write(uploaded_file.read())
        temp_file_path = tfile.name
    
    return temp_file_path

def process_image(image, model_name):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Process the image with AnimeGAN
    animegan = AnimeGAN(f"models/{model_name}.onnx")
    engine = Engine(custom_objects=[animegan], show=False)
    
    # Assume Engine.run() now takes an image array and returns the processed image array
    result_array = engine.run(img_array)
    
    # Convert the result back to a PIL Image
    result_image = Image.fromarray(result_array.astype('uint8'))
    
    return result_image

def main():
    st.title("AnimeGAN Image Processor")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            temp_file_path = save_uploaded_file(uploaded_file)
            image = Image.open(uploaded_file)
            st.image(image, caption='Original Image', use_column_width=True)

            models = ['Hayao_64', 'Hayao-60', 'Paprika_54', 'Shinkai_53']
            
            for model in models:
                with st.spinner(f"Processed with {model}..."):
                    animegan = AnimeGAN(f"models/{model}.onnx")
                    engine = Engine(image_path=temp_file_path, show=False, output_extension=str(model), custom_objects=[animegan])
                    result_image = engine.process_image(temp_file_path)

                    st.subheader(f"Result for {model}")            
                    st.image(result_image, caption=f'Processed with {model}', use_column_width=True)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            st.balloons()
        
        
        
if __name__ == "__main__":
    main()