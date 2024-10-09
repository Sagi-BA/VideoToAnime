import streamlit as st
from PIL import Image
import numpy as np
from utils.engine import Engine
from utils.animegan import AnimeGAN
import onnxruntime as ort
import cv2
from utils.counter import initialize_user_count, increment_user_count, get_user_count
from utils.TelegramSender import TelegramSender
from utils.init import initialize

# Set page config for better mobile responsiveness
st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title="המרת תמונות סטנדרטיות לסגנון אנימה", page_icon="🎥")

# Initialize session state
if 'state' not in st.session_state:
    st.session_state.state = {
        'telegram_sender': TelegramSender(),
        'counted': False,
    }

def reduce_image_resolution(image, scale_factor=0.5):
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return image.resize((new_width, new_height), Image.LANCZOS)

def is_cuda_available():
    return 'CUDAExecutionProvider' in ort.get_available_providers()

def process_image(image, model_name, use_cpu=False):
    if use_cpu:
        image = reduce_image_resolution(image)
        st.info("Image resolution reduced for CPU processing.")
    
    # Convert PIL Image to numpy array (RGB)
    img_array = np.array(image)
    # Convert RGB to BGR (OpenCV format)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    try:
        animegan = AnimeGAN(f"models/{model_name}.onnx")
        engine = Engine(show=False, custom_objects=[animegan])
        
        # Process the image without saving to file
        result_array = engine.custom_processing(img_array)
        
        # Convert BGR back to RGB
        result_array = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
        
        # Convert the result numpy array to PIL Image
        result_image = Image.fromarray(result_array)
        
        if use_cpu:
            result_image = result_image.resize(image.size, Image.LANCZOS)
        
        return result_image
    except Exception as e:
        st.error(f"Error processing image with {model_name}: {str(e)}")
        return None

def load_html_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        return f.read()

def show_examples():
    st.header("🎬 תוצאות")
    
    # Load and display the custom examples HTML
    examples_html = load_html_file('examples.html')
    st.html(examples_html)  
    
    if st.button(f"💖 אהבתי", key=f"like", use_container_width=True):
        st.balloons()

def main():
    title, image_path, footer_content = initialize()
    st.title(title)
    
    # Load and display the custom expander HTML
    expander_html = load_html_file('expander.html')
    st.html(expander_html)  

    # Initialize session state for tracking Telegram message sent
    if 'telegram_message_sent' not in st.session_state:
            st.session_state.telegram_message_sent = False        
        
    tab1, tab2 = st.tabs(["🚀 נסו בעצמכם", "🌟 ראו דוגמאות"])
    
    with tab1:
        uploaded_file = st.file_uploader("העלו תמונה...", type=["jpg", "jpeg", "png", "webp", ".jfif"], key="hidden_uploader")

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Original Image', use_column_width=True)

                use_cpu = not is_cuda_available()
                if use_cpu:
                    st.warning("CUDA is not available. Using CPU for processing with reduced image resolution.")

                models = ['Hayao_64', 'Hayao-60', 'Paprika_54', 'Shinkai_53']
                
                for model in models:
                    with st.spinner(f"Processing with {model}..."):
                        result_image = process_image(image, model, use_cpu)
                        if result_image is not None:
                            st.subheader(f"Result for {model}")            
                            st.image(result_image, caption=f'Processed with {model}', use_column_width=True)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error(f"Error type: {type(e).__name__}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
        else:
            st.warning('☝️ העלו תמונה') 
    with tab2:
        show_examples()
        
    # Display footer content
    st.markdown(footer_content, unsafe_allow_html=True)

    # Display user count after the chatbot
    user_count = get_user_count(formatted=True)
    st.markdown(f"<div class='user-count' style='color: #4B0082;'>סה\"כ משתמשים: {user_count}</div>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()