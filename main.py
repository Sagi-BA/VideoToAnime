import asyncio
from functools import wraps

import base64
import gc
import math
import streamlit as st
import numpy as np
import torch
from encoded_video import EncodedVideo, write_video
from torchvision.transforms.functional import center_crop, to_tensor
import streamlit.components.v1 as components
import os
import tempfile
import uuid
from moviepy.editor import VideoFileClip
import warnings
import io
import contextlib
from utils.counter import initialize_user_count, increment_user_count, get_user_count
from utils.TelegramSender import TelegramSender
from utils.init import initialize
# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set page config for better mobile responsiveness
st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title="המרה של וידאו לאנימציה", page_icon="🎥")

# Initialize session state
if 'state' not in st.session_state:
    st.session_state.state = {
        'telegram_sender': TelegramSender(),
        'counted': False,
    }

async def send_telegram_message_and_file(message, original_image, sketch_image, video_base64=None):
    sender = TelegramSender()
    try:
        # Verify the bot token
        if await sender.verify_bot_token():
            # Send the original and sketch images
            await sender.sketch_image(original_image, sketch_image, caption=message)
            
            # If video_base64 is provided, send the video as well
            if video_base64:
                video_bytes = base64.b64decode(video_base64)
                video_buffer = io.BytesIO(video_bytes)
                await sender.send_video(video_buffer, caption=message)
        else:
            raise Exception("Bot token verification failed")
    except Exception as e:
        st.error(f"Failed to send Telegram message: {str(e)}")
    finally:
        await sender.close_session()

def load_html_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        return f.read()
    
def load_footer():
    footer_path = os.path.join('utils', 'footer.md')
    if os.path.exists(footer_path):
        with open(footer_path, 'r', encoding='utf-8') as footer_file:
            return footer_file.read()
    return None  # Return None if the file doesn't exist
            
def load_model():
    with st.spinner("טוען מודל AI... זה עלול לקחת דקה."):
        print("🧠 טוען מודל...")
        # Temporarily redirect stdout and stderr
        temp_stdout = io.StringIO()
        temp_stderr = io.StringIO()
        with contextlib.redirect_stdout(temp_stdout), contextlib.redirect_stderr(temp_stderr):
            model = torch.hub.load(
                "AK391/animegan2-pytorch:main",
                "generator",
                pretrained=True,
                device=device,
                progress=True,
            )
    return model.to(device)

@st.cache_resource
def get_model():
    return load_model()

def handle_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            st.warning("The app encountered an error. Please try refreshing the page. If the problem persists, please contact support.")
    return wrapper

# Don't load the model immediately
model = None

def uniform_temporal_subsample(x: torch.Tensor, num_samples: int, temporal_dim: int = -3) -> torch.Tensor:
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, temporal_dim, indices)

def short_side_scale(x: torch.Tensor, size: int, interpolation: str = "bilinear") -> torch.Tensor:
    assert len(x.shape) == 4
    assert x.dtype == torch.float32
    c, t, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))
    return torch.nn.functional.interpolate(x, size=(new_h, new_w), mode=interpolation, align_corners=False)

def inference_step(vid, start_sec, duration, out_fps):
    global model
    if model is None:
        model = load_model()

    clip = vid.get_clip(start_sec, start_sec + duration)
    video_arr = torch.from_numpy(clip['video']).permute(3, 0, 1, 2)
    audio_arr = np.expand_dims(clip['audio'], 0) if 'audio' in clip else None
    audio_fps = None if not vid._has_audio else vid._container.streams.audio[0].sample_rate

    x = uniform_temporal_subsample(video_arr, duration * out_fps)
    x = center_crop(short_side_scale(x, 512), 512)
    x /= 255.0
    x = x.permute(1, 0, 2, 3)
    with torch.no_grad():
        x = x.to(device)
        output = model(x).detach().cpu()
        output = (output * 0.5 + 0.5).clip(0, 1) * 255.0
        output_video = output.permute(0, 2, 3, 1).numpy()

    return output_video, audio_arr, out_fps, audio_fps

def get_video_duration(video_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_file.read())
        tmpfile_path = tmpfile.name

    video = VideoFileClip(tmpfile_path)
    duration = video.duration
    video.close()
    os.unlink(tmpfile_path)
    return int(duration)

def predict_fn(video, start_sec, duration):
    out_fps = 12
    
    # Generate a unique identifier for this processing job
    job_id = str(uuid.uuid4())
    
    # Create temporary input and output file names
    temp_input_path = f"temp_input_{job_id}.mp4"
    temp_output_path = f"temp_output_{job_id}.mp4"
    
    try:
        # Save the uploaded video temporarily
        with open(temp_input_path, "wb") as f:
            f.write(video)
        
        vid = EncodedVideo.from_path(temp_input_path)
        video_all = None
        audio_all = None
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        model = get_model()  # Get the cached model here
        
        for i in range(duration):
            # Update progress bar and status text
            progress = (i + 1) / duration
            progress_bar.progress(progress)
            status_text.text(f"🖼️ מעבד שלב {i + 1}/{duration}...")
            
            video, audio, fps, audio_fps = inference_step(vid=vid, start_sec=i + start_sec, duration=1, out_fps=out_fps)
            gc.collect()
            if i == 0:
                video_all = video
                audio_all = audio
            else:
                video_all = np.concatenate((video_all, video))
                if audio is not None and audio_all is not None:
                    audio_all = np.hstack((audio_all, audio))

        status_text.text("💾 כותב את הווידאו הסופי...")
        write_video(temp_output_path, video_all, fps=fps, audio_array=audio_all, audio_fps=audio_fps, audio_codec='aac')

        status_text.text("✅ העיבוד הסתיים!")
        progress_bar.progress(1.0)
        
        # Read the output video into memory
        with open(temp_output_path, "rb") as f:
            output_video = f.read()
        
        return output_video
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        
        del video_all
        del audio_all
        gc.collect()

def show_examples():
    st.header("🎬 דוגמאות להמרות")
    
    # Load and display the custom examples HTML
    examples_html = load_html_file('examples.html')
    st.html(examples_html)  
    
    if st.button(f"💖 אהבתי", key=f"like", use_container_width=True):
        st.balloons()

def html5_slider(label, min_value, max_value, value, key):
    # Initialize session state for this slider if it doesn't exist
    if key not in st.session_state:
        st.session_state[key] = value #sagi
        
    # Create the HTML slider
    components.html(
        f"""
        <label for="{key}">{label}: <span id="{key}_value">{st.session_state[key]}</span></label>
        <input type="range" id="{key}" 
               min="{min_value}" max="{max_value}" 
               value="{st.session_state[key]}" 
               style="width: 100%;"
               oninput="
                   document.getElementById('{key}_value').innerHTML = this.value;
                   document.getElementById('{key}_hidden').value = this.value;
               ">
        <input type="hidden" id="{key}_hidden" name="{key}">
        """,
        height=70,
    )

    # Use a hidden number_input to get the value on the server side
    return st.number_input(label, min_value, max_value, st.session_state[key], key=f"{key}_hidden", label_visibility="collapsed")

@handle_error
def process_video(uploaded_file, start_sec, duration):
    # Reset file pointer to the beginning
    uploaded_file.seek(0)
    with st.spinner('🔮 מעבד וידאו... הקסם באנימציה בעיצומו!'):
        output_video = predict_fn(uploaded_file.read(), start_sec, duration)
    st.subheader("✨ וידאו מומר לסגנון אנימציה")
    st.video(output_video)
    st.snow()
    st.success("🎉 ההמרה הושלמה! איך זה נראה?")
    st.toast('ההמרה הושלמה! איך זה נראה', icon='🎉')

@handle_error
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
        uploaded_file = st.file_uploader("העלו קובץ וידאו", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # Display the original uploaded video
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("וידאו מקורי")
                st.video(data=uploaded_file, format="video/mp4")
            with col2:
                st.subheader("הגדרות")
                with st.form(key='video_form'):
                    video_duration = get_video_duration(uploaded_file)
                    start_sec = html5_slider("🕐 זמן התחלה (שניות)", 0, max(0, video_duration - 1), 0, "start_sec")
                    remaining_duration = video_duration - start_sec
                    duration = html5_slider("⏱️ משך (שניות)", 1, min(remaining_duration, 30), min(remaining_duration, 1), "duration")
                    
                    # Display current values
                    print(f"Start time: {start_sec}")
                    print(f"Duration: {duration}")

                    submit_button = st.form_submit_button(label='🎨 המר לאנימציה', use_container_width=True)

            if submit_button:
                process_video(uploaded_file, start_sec, duration)

    with tab2:
        show_examples()
    
    # Display footer content
    st.markdown(footer_content, unsafe_allow_html=True)    

    # Display user count after the chatbot
    user_count = get_user_count(formatted=True)
    st.markdown(f"<div class='user-count' style='color: #4B0082;'>סה\"כ משתמשים: {user_count}</div>", unsafe_allow_html=True)
    
if __name__ == "__main__":
    if 'counted' not in st.session_state:
        st.session_state.counted = True
        increment_user_count()
    initialize_user_count()
    main()