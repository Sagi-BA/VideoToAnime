import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.hub import download_url_to_file

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

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        pad_layer = {
            "zero":    nn.ZeroPad2d,
            "same":    nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()
        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch*expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))
        layers.extend([
            ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, padding=1),
            nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)

class Generator(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.block_a = nn.Sequential(
            ConvNormLReLU(3,  32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, kernel_size=3, stride=2, padding=1),
            ConvNormLReLU(64, 64)
        )
        self.block_b = nn.Sequential(
            ConvNormLReLU(64,  128, kernel_size=3, stride=2, padding=1),
            ConvNormLReLU(128, 128)
        )
        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )
        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )
        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64,  64),
            ConvNormLReLU(64,  32, kernel_size=7, padding=3)
        )
        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(nn.functional.interpolate(x, scale_factor=2))
        x = self.block_e(nn.functional.interpolate(x, scale_factor=2))
        x = self.out_layer(x)
        return x
    
@st.cache_resource
def load_model():
    print(f"🧠 Loading Model on {device}...")
    model = torch.hub.load(
        "bryandlee/animegan2-pytorch:main",
        "generator",
        pretrained=True,
        progress=True,
    )
    return model.to(device)

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
    x = x / 127.5 - 1.0  # Normalize to [-1, 1]
    x = x.permute(1, 0, 2, 3)
    with torch.no_grad():
        x = x.to(device)
        output = model(x).detach().cpu()
        output = (output * 0.5 + 0.5).clip(0, 1) * 255.0
        output_video = output.permute(0, 2, 3, 1).numpy().astype(np.uint8)

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
            video_duration = get_video_duration(uploaded_file)
            start_sec = st.slider("Start Time (seconds)", 0, max(0, video_duration - 1), 0)
            
            remaining_duration = video_duration - start_sec
            duration = st.slider("Duration (seconds)", 1, remaining_duration, min(remaining_duration, 10))
            
            if st.button('Process Video'):
                # Reset file pointer to the beginning
                uploaded_file.seek(0)
                output_video = predict_fn(uploaded_file.read(), start_sec, duration)
                st.video(output_video)
        else:
            st.write("Please upload a video to begin.")
    
    with tab2:
        show_examples()

if __name__ == "__main__":
    main()