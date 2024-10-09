import asyncio
import base64
import gc
import math
import time
import streamlit as st
import numpy as np
import torch
from encoded_video import EncodedVideo, write_video
from torchvision.transforms.functional import resize, center_crop
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

# Set page config for better mobile responsiveness
st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title="המרה של וידאו לאנימציה", page_icon="🎥")

# Initialize session state
if 'state' not in st.session_state:
    st.session_state.state = {
        'telegram_sender': TelegramSender(),
        'counted': False,
    }

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"cuda.is_available():{torch.cuda.is_available()}")

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
                
@st.cache_resource
def load_model():
    model_path = "animegan2_model.pth"
    with st.spinner("טוען מודל AI... זה עלול לקחת מספר שניות."):
        print("🧠 טוען מודל...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model = torch.hub.load(
                "AK391/animegan2-pytorch:main",
                "generator",
                pretrained=False,
                device=device,
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print("Model loaded successfully from local file.")
            return model.to(device)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            st.error(f"Failed to load model: {str(e)}")
            return None

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

import torch
import numpy as np
from torchvision.transforms.functional import resize, center_crop

def process_video(video_arr, target_size=256):
    # Ensure video_arr is a torch tensor
    if not isinstance(video_arr, torch.Tensor):
        video_arr = torch.from_numpy(video_arr).float()
    
    # Ensure the tensor has the shape (frames, height, width, channels)
    if video_arr.dim() == 3:
        video_arr = video_arr.unsqueeze(-1)  # Add channel dimension if it's missing
    
    # Move channels to the second dimension
    video_arr = video_arr.permute(0, 3, 1, 2)
    
    # Ensure we have 3 channels (RGB)
    if video_arr.shape[1] != 3:
        if video_arr.shape[1] == 1:  # Grayscale
            video_arr = video_arr.repeat(1, 3, 1, 1)
        else:  # More than 3 channels
            video_arr = video_arr[:, :3, :, :]
    
    # Resize the video frames
    t, c, h, w = video_arr.shape
    if h > target_size or w > target_size:
        if h > w:
            new_h, new_w = int(h * target_size / w), target_size
        else:
            new_h, new_w = target_size, int(w * target_size / h)
        video_arr = resize(video_arr, (new_h, new_w))
    
    # Center crop to ensure square frames
    video_arr = center_crop(video_arr, target_size)
    
    return video_arr

def inference_step(vid, start_sec, duration, out_fps, target_size=256):
    global model
    if model is None:
        model = load_model()
    if model is None:
        st.error("Failed to load the model. Please try again later.")
        return None
    
    try:
        clip = vid.get_clip(start_sec, start_sec + duration)
        video_arr = clip['video']
        audio_arr = np.expand_dims(clip['audio'], 0) if 'audio' in clip else None
        audio_fps = None if not vid._has_audio else vid._container.streams.audio[0].sample_rate

        # Process and resize the video
        x = uniform_temporal_subsample(torch.from_numpy(video_arr), duration * out_fps)
        x = process_video(x, target_size)
        x = x.float() / 255.0

        with torch.no_grad():
            x = x.to(device)
            # Process each frame individually
            output_frames = []
            for frame in x:
                frame = frame.unsqueeze(0)  # Add batch dimension
                output = model(frame).detach().cpu()
                output_frames.append(output)
            output = torch.cat(output_frames, dim=0)
            output = (output * 0.5 + 0.5).clip(0, 1) * 255.0
            output_video = output.permute(0, 2, 3, 1).numpy().astype(np.uint8)

        return output_video, audio_arr, out_fps, audio_fps
    except Exception as e:
        st.error(f"Error in inference step: {str(e)}")
        raise  # Re-raise the exception for debugging

def save_uploaded_file(uploaded_file):
    # Get the file extension from the uploaded file
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    # Create a temporary file with the correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tfile:
        tfile.write(uploaded_file.read())
        temp_file_path = tfile.name
    
    return temp_file_path

def get_video_duration(file_path):
    with VideoFileClip(file_path) as video:
        duration = video.duration
    return int(duration)

def predict_fn(video_path, start_sec, duration, target_size=256):
    out_fps = 12
    
    # Generate a unique identifier for this processing job
    job_id = str(uuid.uuid4())
    
    # Create temporary output file name
    temp_output_path = f"temp_output_{job_id}.mp4"
    
    try:
        vid = EncodedVideo.from_path(video_path)
        video_all = None
        audio_all = None
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(duration):
            # Update progress bar and status text
            progress = (i + 1) / duration
            progress_bar.progress(progress)
            status_text.text(f"🖼️ מעבד שלב {i + 1}/{duration}...")
            
            result = inference_step(vid=vid, start_sec=i + start_sec, duration=1, out_fps=out_fps, target_size=target_size)
            if result is None:
                st.error("Failed to process video segment")
                return None
            video, audio, fps, audio_fps = result
            
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
        # Clean up temporary output file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        
        del video_all
        del audio_all
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # If using CUDA

def show_examples():
    st.header("🎬 דוגמאות להמרות")
    
    # Load and display the custom examples HTML
    examples_html = load_html_file('examples.html')
    st.html(examples_html)  
    
    if st.button(f"💖 אהבתי", key=f"like", use_container_width=True):
        st.balloons()

async def main():
    try:
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
                temp_file_path = save_uploaded_file(uploaded_file)
                try:
                    # Display the original uploaded video
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("וידאו מקורי")
                        st.video(temp_file_path)
                    with col2:
                        st.subheader("הגדרות")
                        with st.form(key='video_form'):
                            video_duration = get_video_duration(temp_file_path)                        
                            start_sec = st.slider("זמן התחלה (שניות)", 0, max(0, video_duration - 1), 0)
                            
                            remaining_duration = video_duration - start_sec
                            duration = st.slider("משך זמן (שניות)", 1, remaining_duration, min(remaining_duration, 10))
                            
                            submit_button = st.form_submit_button(label='🎨 המר לאנימציה', use_container_width=True)

                    if submit_button:
                        print(f"Start time: {start_sec}")
                        print(f"Duration: {duration}")
                        with st.spinner('🔮 מעבד וידאו... הקסם באנימציה בעיצומו!'):
                            output_video = predict_fn(temp_file_path, start_sec, duration)
                        st.subheader("✨ וידאו מומר לסגנון אנימציה")
                        st.video(output_video)
                        st.snow()
                        st.success("🎉 ההמרה הושלמה! איך זה נראה?")
                        st.toast('ההמרה הושלמה! איך זה נראה', icon='🎉')
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
            else:
                st.warning('☝️ העלו קובץ וידאו')
        
        with tab2:
            show_examples()
        
        # Display footer content
        st.markdown(footer_content, unsafe_allow_html=True)

        # Display user count after the chatbot
        user_count = get_user_count(formatted=True)
        st.markdown(f"<div class='user-count' style='color: #4B0082;'>סה\"כ משתמשים: {user_count}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    if 'counted' not in st.session_state:
        st.session_state.counted = True
        increment_user_count()
    initialize_user_count()
    asyncio.run(main())