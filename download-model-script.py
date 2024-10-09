import torch
import os

def download_model():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download the model
    model = torch.hub.load(
        "AK391/animegan2-pytorch:main",
        "generator",
        pretrained=True,
        device=device,
        progress=True,
    )
    
    # Save the model
    torch.save(model.state_dict(), "animegan2_model.pth")
    print("Model saved successfully!")

if __name__ == "__main__":
    download_model()
