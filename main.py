import gc
import math
import streamlit as st
import gradio as gr
import numpy as np
import torch
from encoded_video import EncodedVideo, write_video
from torchvision.transforms.functional import center_crop, to_tensor
import json
import os
import tempfile
import uuid
from moviepy.editor import VideoFileClip

import io
import contextlib

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    print("🧠 Loading Model...")
    # Temporarily redirect stdout and stderr
    temp_stdout = io.StringIO()
    temp_stderr = io.StringIO()
    with contextlib.redirect_stdout(temp_stdout), contextlib.redirect_stderr(temp_stderr):
        model = torch.hub.load(
            "AK391/animegan2-pytorch:main",
            "generator",
            pretrained=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            progress=True,
        )
    return model.to(device)

model = load_model()


import gc
import math
import streamlit as st
import gradio as gr
import numpy as np
import torch
from encoded_video import EncodedVideo, write_video
from torchvision.transforms.functional import center_crop, to_tensor

@st.cache_resource
def load_model():
    print("🧠 Loading Model...")
    return torch.hub.load(
        "AK391/animegan2-pytorch:main",
        "generator",
        pretrained=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        progress=True,
    )

model = load_model()

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

def predict_fn(video, start_sec, duration):
    out_fps = 12
    # Save the uploaded video temporarily
    with open("temp_input.mp4", "wb") as f:
        f.write(video)
    
    vid = EncodedVideo.from_path("temp_input.mp4")
    video_all = None
    audio_all = None
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(duration):
        # Update progress bar and status text
        progress = (i + 1) / duration
        progress_bar.progress(progress)
        status_text.text(f"🖼️ Processing step {i + 1}/{duration}...")
        
        video, audio, fps, audio_fps = inference_step(vid=vid, start_sec=i + start_sec, duration=1, out_fps=out_fps)
        gc.collect()
        if i == 0:
            video_all = video
            audio_all = audio
        else:
            video_all = np.concatenate((video_all, video))
            if audio is not None and audio_all is not None:
                audio_all = np.hstack((audio_all, audio))

    status_text.text("💾 Writing output video...")
    output_path = 'out.mp4'
    write_video(output_path, video_all, fps=fps, audio_array=audio_all, audio_fps=audio_fps, audio_codec='aac')

    status_text.text("✅ Processing complete!")
    progress_bar.progress(1.0)
    
    del video_all
    del audio_all

    return output_path

def gradio_interface():
    with gr.Blocks() as iface:
        gr.Markdown("# AnimeGANV2 On Videos")
        gr.Markdown("Applying AnimeGAN-V2 to frames from video clips")
        
        with gr.Row():
            input_video = gr.Video(label="Input Video")
            output_video = gr.Video(label="Output Video")
        
        with gr.Row():
            start_sec = gr.Slider(minimum=0, maximum=300, step=1, value=0, label="Start Time (seconds)")
            duration = gr.Slider(minimum=1, maximum=4, step=1, value=2, label="Duration (seconds)")
        
        process_btn = gr.Button("Process Video")
        process_btn.click(fn=predict_fn, inputs=[input_video, start_sec, duration], outputs=output_video)
        
        gr.Markdown("""
        <p style='text-align: center'>
            <a href='https://github.com/bryandlee/animegan2-pytorch' target='_blank'>Github Repo Pytorch</a>
        </p>
        """)
    
    return iface

def load_examples():
    if os.path.exists('examples.json'):
        with open('examples.json', 'r') as f:
            return json.load(f)
    else:
        # Default example if JSON file doesn't exist
        return [{
            "name": "Gaya",
            "original": "examples/Gaya.mp4",
            "anime": "examples/Gaya_anime.mp4",
            "description": "Transformation of Gaya video to anime style"
        }]

def show_examples():
    st.header("Example Transformations")
    examples = load_examples()
    
    for example in examples:
        with st.expander(f"{example['name']} Example"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.video(example['original'])
            with col2:
                st.subheader("Anime Style")
                st.video(example['anime'])
            st.write(example['description'])

import gc
import math
import streamlit as st
import gradio as gr
import numpy as np
import torch
from encoded_video import EncodedVideo, write_video
from torchvision.transforms.functional import center_crop, to_tensor
import json
import os
import tempfile
import uuid
from moviepy.editor import VideoFileClip

# ... [Previous code remains unchanged] ...

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
            status_text.text(f"🖼️ Processing step {i + 1}/{duration}...")
            
            video, audio, fps, audio_fps = inference_step(vid=vid, start_sec=i + start_sec, duration=1, out_fps=out_fps)
            gc.collect()
            if i == 0:
                video_all = video
                audio_all = audio
            else:
                video_all = np.concatenate((video_all, video))
                if audio is not None and audio_all is not None:
                    audio_all = np.hstack((audio_all, audio))

        status_text.text("💾 Writing output video...")
        write_video(temp_output_path, video_all, fps=fps, audio_array=audio_all, audio_fps=audio_fps, audio_codec='aac')

        status_text.text("✅ Processing complete!")
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

def main():
    st.title('AnimeGANV2 On Videos')
    st.write("Applying AnimeGAN-V2 to frames from video clips")
    
    tab1, tab2 = st.tabs(["Try It Yourself", "View Examples"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # Display the original uploaded video
            st.subheader("Original Video")
            st.video(uploaded_file)

            video_duration = get_video_duration(uploaded_file)
            start_sec = st.slider("Start Time (seconds)", 0, max(0, video_duration - 1), 0)
            
            remaining_duration = video_duration - start_sec
            duration = st.slider("Duration (seconds)", 1, remaining_duration, min(remaining_duration, 10))
            
            if st.button('Process Video'):
                # Reset file pointer to the beginning
                uploaded_file.seek(0)
                with st.spinner('Processing video...'):
                    output_video = predict_fn(uploaded_file.read(), start_sec, duration)
                st.subheader("Processed Video")
                st.video(output_video)
        else:
            st.write("Please upload a video to begin.")
    
    with tab2:
        show_examples()

if __name__ == "__main__":
    main()