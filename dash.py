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
import gcsfs
from utils import preprocess_image, resize_and_colorize_mask, FPN_Segmenter, CLASS_COLORS

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# 🔹 Mise en cache des modèles pour éviter les rechargements inutiles
@st.cache_resource
def load_models():
    fpn_model_path = "fpn_best.pth"
    mask2former_model_path = "mask2former_best.pth"

    fpn_url = "https://storage.googleapis.com/p9-dashboard-storage/Models/fpn_best.pth"
    mask2former_url = "https://storage.googleapis.com/p9-dashboard-storage/Models/mask2former_best.pth"

    if not os.path.exists(fpn_model_path):
        urllib.request.urlretrieve(fpn_url, fpn_model_path)
    
    if not os.path.exists(mask2former_model_path):
        urllib.request.urlretrieve(mask2former_url, mask2former_model_path)

    try:
        fpn_model = torch.load(fpn_model_path, map_location=torch.device("cpu"))
        fpn_model.eval()

        mask2former_model = torch.load(mask2former_model_path, map_location=torch.device("cpu"))
        mask2former_model.eval()

        st.write("✅ Modèles chargés avec succès.")
        return fpn_model, mask2former_model

    except Exception as e:
        st.error(f"⚠ Erreur lors du chargement des modèles : {e}")
        return None, None

fpn_model, mask2former_model = load_models()

# 🔹 Sidebar Navigation
st.sidebar.title("Menu")
page = st.sidebar.radio("Aller à :", ["EDA", "Résultats des modèles", "Test des modèles"])

# 🔹 Chargement des images et masques depuis GCS
@st.cache_data
def get_available_images():
    """Liste les images disponibles sur GCS."""
    try:
        fs = gcsfs.GCSFileSystem()
        image_files = fs.ls("p9-dashboard-storage/Dataset/images")
        available_images = [img.split("/")[-1] for img in image_files if img.endswith(".png")]
        return available_images
    except Exception as e:
        st.error(f"⚠ Erreur lors de la récupération des images : {e}")
        return []

available_images = get_available_images()

# 🔹 Page EDA
if page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    # 🔹 Structure du dataset
    st.header("Structure des Dossiers et Fichiers")
    folders = {"Images": ["train", "val", "test"], "Masques": ["train", "val", "test"]}
    for key, values in folders.items():
        st.write(f"**{key}**: {', '.join(values)}")

    dataset_info = {
        "Ensemble": ["Train", "Validation", "Test"],
        "Images": [2975, 500, 1525],
        "Masques": [2975, 500, 1525]
    }
    df_info = pd.DataFrame(dataset_info)
    st.table(df_info)

# 🔹 Page Résultats des modèles
@st.cache_data
def load_results():
    try:
        fpn_results = pd.read_csv("https://storage.googleapis.com/p9-dashboard-storage/Resultats/fpn_results.csv")
        mask2former_results = pd.read_csv("https://storage.googleapis.com/p9-dashboard-storage/Resultats/mask2former_results.csv")
        return fpn_results, mask2former_results
    except Exception as e:
        st.error(f"⚠ Erreur lors du chargement des résultats : {e}")
        return None, None

fpn_results, mask2former_results = load_results()

if page == "Résultats des modèles" and fpn_results is not None and mask2former_results is not None:
    st.title("Analyse des Résultats des Modèles")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpn_results["Epoch"], y=fpn_results["Val Loss"], mode='lines', name='FPN - Validation Loss'))
    fig.add_trace(go.Scatter(x=fpn_results["Epoch"], y=fpn_results["Val IoU"], mode='lines', name='FPN - Validation IoU Score'))
    fig.add_trace(go.Scatter(x=mask2former_results["Epoch"], y=mask2former_results["Val Loss"], mode='lines', name='Mask2Former - Validation Loss'))
    fig.add_trace(go.Scatter(x=mask2former_results["Epoch"], y=mask2former_results["Val IoU"], mode='lines', name='Mask2Former - Validation IoU Score'))

    st.plotly_chart(fig)

# 🔹 Page Test des modèles
if page == "Test des modèles":
    st.title("Test de Segmentation avec les Modèles")

    if not available_images:
        st.error("⚠ Aucune image trouvée sur GCS.")
    else:
        image_choice = st.selectbox("Choisissez une image à segmenter", available_images)
        model_choice = st.radio("Choisissez le modèle", ["FPN", "Mask2Former"])

        image_url = f"https://storage.googleapis.com/p9-dashboard-storage/Dataset/images/{image_choice}"

        try:
            image = Image.open(urllib.request.urlopen(image_url)).convert("RGB")
            st.image(image, caption="Image d'entrée", use_container_width=True)
        except Exception as e:
            st.error(f"⚠ Erreur lors du chargement de l'image : {e}")

        if fpn_model is None or mask2former_model is None:
            st.error("⚠ Les modèles ne sont pas disponibles. Impossible d'effectuer la segmentation.")
        else:
            input_size = (512, 512)
            image_resized, original_size = preprocess_image(image, input_size)
            tensor_image = torch.tensor(image_resized).permute(0, 3, 1, 2).float().unsqueeze(0)

            try:
                output = fpn_model(tensor_image) if model_choice == "FPN" else mask2former_model(tensor_image)
                mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
                mask_colorized = resize_and_colorize_mask(mask, original_size, CLASS_COLORS)

                st.image(mask_colorized, caption="Masque segmenté", use_container_width=True)

                # Masque réel
                mask_filename = image_choice.replace("leftImg8bit", "gtFine_color")
                mask_url = f"https://storage.googleapis.com/p9-dashboard-storage/Dataset/masks/{mask_filename}"

                st.image(Image.open(urllib.request.urlopen(mask_url)).convert("RGB"), caption="Masque réel", use_container_width=True)

            except Exception as e:
                st.error(f"⚠ Erreur lors de la segmentation : {e}")
