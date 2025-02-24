import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
from PIL import Image
import numpy as np
import warnings
import plotly.graph_objects as go
from utils import preprocess_image, resize_and_colorize_mask, FPN_Segmenter, CLASS_COLORS

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# 🔹 Configuration du bucket GCS
BUCKET_NAME = "p9-dashboard-storage"
IMAGE_FOLDER = "Dataset/images"
MASK_FOLDER = "Dataset/masks"

# 🔹 Suppression de google.cloud.storage pour éviter les erreurs
# 🔹 On accède directement aux images via URL publique
@st.cache_data
def get_available_images_and_masks():
    """Liste les images et masques disponibles en utilisant les URL GCS publiques."""
    available_images = [
        "lindau_000001_000019_leftImg8bit.png",
        "lindau_000002_000019_leftImg8bit.png",
        "lindau_000003_000019_leftImg8bit.png",
        "lindau_000004_000019_leftImg8bit.png",
        "lindau_000005_000019_leftImg8bit.png",
    ]

    available_masks = [
        "lindau_000001_000019_gtFine_color.png",
        "lindau_000002_000019_gtFine_color.png",
        "lindau_000003_000019_gtFine_color.png",
        "lindau_000004_000019_gtFine_color.png",
        "lindau_000005_000019_gtFine_color.png",
    ]

    return available_images, available_masks

available_images, available_masks = get_available_images_and_masks()

# 🔹 Page Test des modèles
if page == "Test des modèles":
    st.title("Test de Segmentation avec les Modèles")

    image_choice = st.selectbox("Choisissez une image à segmenter", available_images)
    model_choice = st.radio("Choisissez le modèle", ["FPN", "Mask2Former"])

    image_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{IMAGE_FOLDER}/{image_choice}"
    mask_filename = image_choice.replace("leftImg8bit", "gtFine_color")
    mask_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{MASK_FOLDER}/{mask_filename}"

    image = Image.open(urllib.request.urlopen(image_url)).convert("RGB")
    st.image(image, caption="Image d'entrée", use_column_width=True)

    input_size = (512, 512)
    image_resized, original_size = preprocess_image(image, input_size)
    tensor_image = torch.tensor(image_resized).permute(0, 3, 1, 2).float().unsqueeze(0)

    output = fpn_model(tensor_image) if model_choice == "FPN" else mask2former_model(tensor_image)
    mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    mask_colorized = resize_and_colorize_mask(mask, original_size, CLASS_COLORS)

    st.image(mask_colorized, caption="Masque segmenté", use_column_width=True)
    st.image(Image.open(urllib.request.urlopen(mask_url)).convert("RGB"), caption="Masque réel", use_column_width=True)
