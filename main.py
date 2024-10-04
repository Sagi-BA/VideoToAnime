import asyncio
import base64
import gc
import math
import streamlit as st
import numpy as np
import torch
from encoded_video import EncodedVideo, write_video
from torchvision.transforms.functional import center_crop, to_tensor
import json
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
st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title="המרה של וידאו לאנימציה", page_icon="🎬")

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
            
@st.cache_resource
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

@st.cache_resource
def load_examples():
    file_name = "examples.json"
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Default example if JSON file doesn't exist
        return [{
            "name": "Gaya",
            "original": "examples/Gaya.mp4",
            "anime": "examples/Gaya_anime.mp4",
            "description": "המרת הווידאו של גאיה לסגנון אנימציה"
        }]


def show_examples():
    st.header("🎬 דוגמאות להמרות")
    examples = load_examples()
    
    # Custom CSS for cool design
    st.markdown("""
    <style>
        .video-container {
            display: flex;
            justify-content: space-between;
            align-items: top;
            margin-bottom: 2rem;
            background: linear-gradient(45deg, #f3ec78, #af4261);
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .video-wrapper {
            width: 48%;
        }
        .video-title {
            text-align: center;
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: white;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }
        .description {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 1rem;
            border-radius: 5px;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    for example in examples:
        st.write(f"""
        <div class="video-container">
            <div class="video-wrapper">
                <div class="video-title">מקורי</div>
                <video width="100%" controls>
                    <source src="{example['original']}" type="video/mp4">
                    הדפדפן שלך לא תומך בתגית הווידאו.
                </video>
            </div>
            <div class="video-wrapper">
                <div class="video-title">סגנון אנימציה</div>
                <video width="100%" controls>
                    <source src="{example['anime']}" type="video/mp4">
                    הדפדפן שלך לא תומך בתגית הווידאו.
                </video>
            </div>
        </div>        
        """, unsafe_allow_html=True)
        
        if st.button(f"💖 אהבת את ההמרה של {example['name']}", key=f"like_{example['name']}"):
            st.balloons()
            st.success(f"אהבת את ההמרה של {example['name']}!")


async def main():
    title, image_path, footer_content = initialize()
    st.title(title)

    # Load and display the custom expander HTML
    expander_html = load_html_file('expander.html')
    st.markdown(expander_html, unsafe_allow_html=True)  

     # Initialize session state for tracking Telegram message sent
    if 'telegram_message_sent' not in st.session_state:
        st.session_state.telegram_message_sent = False        
    
    tab1, tab2 = st.tabs(["🚀 נסו בעצמכם", "🌟 ראו דוגמאות"])
    
    with tab1:
        uploaded_file = st.file_uploader("העלו קובץ וידאו", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            try:
                # Display the original uploaded video
                st.subheader("וידאו מקורי")
                st.video(uploaded_file)

                video_duration = get_video_duration(uploaded_file)
                start_sec = st.slider("🕐 זמן התחלה (שניות)", 0, max(0, video_duration - 1), 0)
                
                remaining_duration = video_duration - start_sec
                duration = st.slider("⏱️ משך (שניות)", 1, min(remaining_duration, 30), min(remaining_duration, 10))
                
                if st.button('🎨 המר לאנימציה'):
                    # Reset file pointer to the beginning
                    uploaded_file.seek(0)
                    with st.spinner('🔮 מעבד וידאו... הקסם באנימציה בעיצומו!'):
                        output_video = predict_fn(uploaded_file.read(), start_sec, duration)
                    st.subheader("✨ וידאו מומר לסגנון אנימציה")
                    st.video(output_video)
                    st.success("🎉 ההמרה הושלמה! איך זה נראה?")
                    # video_url = uploader.upload_media_to_imgur(output_video, "video", english_captioning, hebrew_captioning)

            except Exception as e:
                st.error(f"😢 אופס! התרחשה שגיאה: {str(e)}")
                st.error("אנא נסה להעלות וידאו אחר או בדוק את הפורמט.")
    
    with tab2:
        show_examples()
    
    # Display footer content
    st.markdown(footer_content, unsafe_allow_html=True)    

    # Display user count after the chatbot
    user_count = get_user_count(formatted=True)
    st.markdown(f"<p class='user-count' style='color: #4B0082;'>סה\"כ משתמשים: {user_count}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    if 'counted' not in st.session_state:
        st.session_state.counted = True
        increment_user_count()
    initialize_user_count()
    asyncio.run(main())