import os
import sys
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from torchvision import transforms
from transformers import AutoConfig, AutoModel

# Adjust path to import model_davit correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.model import Model

st.set_page_config(page_title="Pneumonia Detection AI", layout="wide")

MODEL_PATH = os.path.join(parent_dir, 'outputs', 'model_best.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Google Drive model file ID - Replace this with your actual file ID
GOOGLE_DRIVE_MODEL_ID = "1Tzhfq17ytcZAcK4-6FOmjB_YeUtcnV0b"  # Your model file

def download_model_from_google_drive(file_id, destination):
    """Download model from Google Drive using direct usercontent URL"""
    import urllib.request
    import urllib.error
    
    # Use usercontent.google.com for direct download (more reliable)
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=true"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=300) as response:
            # Save the file
            with open(destination, 'wb') as f:
                chunk_size = 32768
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    
    except urllib.error.URLError as e:
        raise Exception(f"Download failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to download model: {str(e)}")

class DummyArgs:
    pass

class DummyProcessor:
    def __init__(self):
        self.size = {"shortest_edge": 224}

@st.cache_resource
def load_model():
    # Create outputs directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Download model if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... This may take a few minutes (first time only)")
        try:
            download_model_from_google_drive(GOOGLE_DRIVE_MODEL_ID, MODEL_PATH)
            st.success("Model downloaded successfully!")
        except Exception as e:
            # If download fails, remove any corrupted file
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            st.error(f"Failed to download model: {e}")
            return None
    
    try:
        config = AutoConfig.from_pretrained('facebook/dinov2-small')
        vit = AutoModel.from_pretrained('facebook/dinov2-small')
        args = DummyArgs()
        processor = DummyProcessor()
        
        model = Model(vit, processor, args, num_labels=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        # If model loading fails, remove corrupted file and retry
        st.warning(f"Model file corrupted. Redownloading... Error: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        st.rerun()

model = load_model()

if model is None:
    st.error(f"Model file not found at: {MODEL_PATH}. Please run training first.")
    st.stop()

# Define Preprocessing
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_mean, std=image_std)
])

# Grad-CAM specific
activations = None
gradients = None

def get_activations_hook(module, input, output):
    global activations
    activations = output

def get_gradients_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# Clear any previously registered hooks to prevent duplicate hook execution during Streamlit reloads
cnn_layer = model.cnn
cnn_layer._forward_hooks.clear()
cnn_layer._backward_hooks.clear()

cnn_layer.register_forward_hook(get_activations_hook)
cnn_layer.register_full_backward_hook(get_gradients_hook)

def generate_gradcam(img_tensor, pred_class, original_image):
    global activations, gradients
    
    model.eval()
    img_tensor.requires_grad = True
    model.zero_grad()
    
    probs = model(pixel_values=img_tensor)
    score = probs[0, pred_class]
    score.backward()
    
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # Detach and clone to prevent in-place modification of computation graph tensors
    act = activations.detach().clone()
    pooled_grad = pooled_gradients.detach().clone()
    
    for i in range(act.shape[1]):
        act[:, i, :, :] *= pooled_grad[i]
        
    heatmap = torch.mean(act, dim=1).squeeze()
    heatmap = heatmap.cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    
    if np.max(heatmap) == 0:
        heatmap_normalized = heatmap
    else:
        heatmap_normalized = heatmap / np.max(heatmap)
    
    heatmap_resized = cv2.resize(heatmap_normalized, (original_image.width, original_image.height))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed_img = np.float32(heatmap_colored) * 0.4 + np.float32(original_image)
    superimposed_img = superimposed_img / np.max(superimposed_img)
    return np.uint8(255 * superimposed_img)

st.title("Pneumonia Detection AI")
st.markdown("Upload chest X-ray images to predict the probability of Pneumonia and visualize the model's focus using **Grad-CAM**.")

uploaded_files = st.file_uploader("Choose X-ray images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.subheader(f"Results for: `{uploaded_file.name}`")
        
        try:
            image = Image.open(uploaded_file).convert('RGB')
        except Exception as e:
            st.error(f"Could not open image {uploaded_file.name}. Error: {str(e)}")
            continue
        
        # Inference
        clean_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            probs_clean = model(pixel_values=clean_tensor)[0].cpu().numpy()
            
        pred_class = np.argmax(probs_clean)
        confidence = probs_clean[pred_class]
        classes = ['NORMAL', 'PNEUMONIA']
        pred_label = classes[pred_class]
        
        if confidence > 0.8:
            risk = "High"
            color = "red" if pred_class == 1 else "green"
        elif confidence >= 0.6:
            risk = "Medium"
            color = "orange"
        else:
            risk = "Low"
            color = "gray"

        # Generate Grad-CAM for the predicted class
        img_tensor_grad = preprocess(image).unsqueeze(0).to(DEVICE)
        superimposed_img = generate_gradcam(img_tensor_grad, pred_class, image)
        
        # UI Layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original X-Ray", use_container_width=True)
            st.markdown(f"### Prediction: <span style='color:{color}'>{pred_label}</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence Score:** {confidence*100:.2f}%")
            st.markdown(f"**Risk Level:** {risk}")
            
        with col2:
            st.image(superimposed_img, caption="Grad-CAM Heatmap", use_container_width=True)
            
            # Probability Bar Chart
            st.write("**Class Probabilities**")
            fig_bar, ax_bar = plt.subplots(figsize=(5, 3))
            sns.barplot(x=probs_clean * 100, y=classes, hue=classes, palette=["green", "red"], legend=False, ax=ax_bar)
            ax_bar.set_xlim([0, 100])
            ax_bar.set_xlabel('Probability (%)')
            st.pyplot(fig_bar)
            plt.close()
        
        st.write("**Pixel Intensity Histogram (Grayscale)**")
        gray_img = np.array(image.convert('L'))
        fig_hist, ax_hist = plt.subplots(figsize=(10, 3))
        ax_hist.hist(gray_img.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
        ax_hist.set_xlim([0, 256])
        ax_hist.set_xlabel('Pixel Intensity')
        ax_hist.set_ylabel('Frequency')
        st.pyplot(fig_hist)
        plt.close()
