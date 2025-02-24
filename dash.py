import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import warnings
import plotly.graph_objects as go
from google.cloud import storage  # Importation du client GCS
from utils import preprocess_image, resize_and_colorize_mask, FPN_Segmenter, CLASS_COLORS

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# 🔹 Initialisation du client Google Cloud Storage
storage_client = storage.Client()
BUCKET_NAME = "p9-dashboard-storage"

# 🔹 Mise en cache des modèles pour éviter les rechargements inutiles
@st.cache_resource
def load_models():
    fpn_model_path = "fpn_best.pth"
    mask2former_model_path = "mask2former_best.pth"

    fpn_url = f"https://storage.googleapis.com/{BUCKET_NAME}/Models/fpn_best.pth"
    mask2former_url = f"https://storage.googleapis.com/{BUCKET_NAME}/Models/mask2former_best.pth"

    if not os.path.exists(fpn_model_path):
        os.system(f"wget {fpn_url} -O {fpn_model_path}")
    if not os.path.exists(mask2former_model_path):
        os.system(f"wget {mask2former_url} -O {mask2former_model_path}")

    fpn_model = torch.load(fpn_model_path, map_location=torch.device("cpu"))
    fpn_model.eval()
    mask2former_model = torch.load(mask2former_model_path, map_location=torch.device("cpu"))
    mask2former_model.eval()
    return fpn_model, mask2former_model

fpn_model, mask2former_model = load_models()
st.write("Modèles chargés avec succès")

# 🔹 Sidebar Navigation
st.sidebar.title("Menu")
page = st.sidebar.radio("Aller à :", ["EDA", "Résultats des modèles", "Test des modèles"])

# 🔹 Fonction pour télécharger une image depuis GCS
@st.cache_data
def download_image_from_gcs(image_name):
    """Télécharge une image depuis GCS et la retourne en tant qu'objet PIL."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"Dataset/images/{image_name}")
    
    try:
        image_bytes = blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        st.error(f"❌ Erreur lors du téléchargement de l'image {image_name} : {e}")
        return None

# 🔹 Chargement de la liste des images et masques
@st.cache_data
def get_available_images_and_masks():
    """Charge les listes d'images et de masques à partir des fichiers CSV sur GCS."""
    image_csv_url = f"https://storage.googleapis.com/{BUCKET_NAME}/image_list.csv"
    mask_csv_url = f"https://storage.googleapis.com/{BUCKET_NAME}/mask_list.csv"
    
    try:
        df_images = pd.read_csv(image_csv_url)
        available_images = df_images["image_name"].tolist()

        df_masks = pd.read_csv(mask_csv_url)
        available_masks = df_masks["mask_name"].tolist()

        return available_images, available_masks
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des fichiers CSV : {e}")
        return [], []

available_images, available_masks = get_available_images_and_masks()

# 🔹 Page Test des modèles
if page == "Test des modèles":
    st.title("Test de Segmentation avec les Modèles")

    image_choice = st.selectbox("Choisissez une image à segmenter", available_images)
    model_choice = st.radio("Choisissez le modèle", ["FPN", "Mask2Former"])

    # 🔹 Téléchargement de l’image
    image = download_image_from_gcs(image_choice)
    if image:
        st.image(image, caption="Image d'entrée", use_column_width=True)
        
        # 🔹 Chargement du masque réel correspondant
        mask_filename = image_choice.replace("leftImg8bit", "gtFine_color")
        mask_url = f"https://storage.googleapis.com/{BUCKET_NAME}/Dataset/masks/{mask_filename}"
        
        # 🔹 Prétraitement et passage dans le modèle
        input_size = (512, 512)
        image_resized, original_size = preprocess_image(image, input_size)
        tensor_image = torch.tensor(image_resized).permute(0, 3, 1, 2).float().unsqueeze(0)

        output = fpn_model(tensor_image) if model_choice == "FPN" else mask2former_model(tensor_image)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        mask_colorized = resize_and_colorize_mask(mask, original_size, CLASS_COLORS)

        st.image(mask_colorized, caption="Masque segmenté", use_column_width=True)
        st.image(Image.open(urllib.request.urlopen(mask_url)).convert("RGB"), caption="Masque réel", use_column_width=True)
