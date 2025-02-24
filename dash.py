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
from google.cloud import storage  # ✅ Utilisation de google.cloud.storage

from utils import preprocess_image, resize_and_colorize_mask, FPN_Segmenter, CLASS_COLORS

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# 🔹 Définition du bucket GCS
BUCKET_NAME = "p9-dashboard-storage"
IMAGE_FOLDER = "Dataset/images"
MASK_FOLDER = "Dataset/masks"

# 🔹 Initialisation du client GCS
storage_client = storage.Client.create_anonymous_client()  # ✅ Utilisation sans authentification car bucket public
bucket = storage_client.bucket(BUCKET_NAME)

# 🔹 Mise en cache des modèles pour éviter les rechargements inutiles
@st.cache_resource
def load_models():
    fpn_model_path = "fpn_best.pth"
    mask2former_model_path = "mask2former_best.pth"

    fpn_url = f"https://storage.googleapis.com/{BUCKET_NAME}/Models/fpn_best.pth"
    mask2former_url = f"https://storage.googleapis.com/{BUCKET_NAME}/Models/mask2former_best.pth"

    if not os.path.exists(fpn_model_path):
        urllib.request.urlretrieve(fpn_url, fpn_model_path)

    if not os.path.exists(mask2former_model_path):
        urllib.request.urlretrieve(mask2former_url, mask2former_model_path)

    fpn_model = torch.load(fpn_model_path, map_location=torch.device("cpu"), weights_only=True)
    fpn_model.eval()

    mask2former_model = torch.load(mask2former_model_path, map_location=torch.device("cpu"), weights_only=True)
    mask2former_model.eval()

    return fpn_model, mask2former_model

fpn_model, mask2former_model = load_models()
st.write("✅ Modèles chargés avec succès.")

# 🔹 Fonction pour récupérer la liste des images disponibles dans le bucket
@st.cache_data
def get_available_images():
    """Liste toutes les images disponibles dans le bucket GCS."""
    try:
        blobs = bucket.list_blobs(prefix=IMAGE_FOLDER + "/")
        image_files = [blob.name.split("/")[-1] for blob in blobs if blob.name.endswith(".png")]

        if not image_files:
            st.error("❌ Aucune image trouvée dans le bucket. Vérifiez le stockage GCS.")
            return []

        return image_files
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des images : {e}")
        return []

available_images = get_available_images()

# 🔹 Vérification si la liste d'images est vide
if not available_images:
    st.error("⚠ Aucune image disponible. Arrêt du script.")
    st.stop()  # ✅ Stopper l'exécution pour éviter d'autres erreurs

st.write(f"✅ {len(available_images)} images disponibles.")

# 🔹 Sidebar Navigation
st.sidebar.title("Menu")
page = st.sidebar.radio("Aller à :", ["EDA", "Résultats des modèles", "Test des modèles"])

# 🔹 Page EDA
if page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
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
    fpn_results = pd.read_csv(f"https://storage.googleapis.com/{BUCKET_NAME}/Resultats/fpn_results.csv")
    mask2former_results = pd.read_csv(f"https://storage.googleapis.com/{BUCKET_NAME}/Resultats/mask2former_results.csv")
    return fpn_results, mask2former_results

fpn_results, mask2former_results = load_results()

if page == "Résultats des modèles":
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

    image_choice = st.selectbox("Choisissez une image à segmenter", available_images)
    model_choice = st.radio("Choisissez le modèle", ["FPN", "Mask2Former"])

    # 🔹 Vérification de la sélection d'image
    if not image_choice:
        st.error("⚠ Aucune image sélectionnée.")
        st.stop()

    # 🔹 URL directe des fichiers GCS (aucun téléchargement local nécessaire)
    image_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{IMAGE_FOLDER}/{image_choice}"
    mask_filename = image_choice.replace("leftImg8bit", "gtFine_color")
    mask_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{MASK_FOLDER}/{mask_filename}"

    try:
        image = Image.open(urllib.request.urlopen(image_url)).convert("RGB")
        st.image(image, caption="Image d'entrée", use_container_width=True)
    except Exception as e:
        st.error(f"⚠ Erreur lors du chargement de l'image : {e}")

    try:
        input_size = (512, 512)
        image_resized, original_size = preprocess_image(image, input_size)
        tensor_image = torch.tensor(image_resized).permute(0, 3, 1, 2).float().unsqueeze(0)

        output = fpn_model(tensor_image) if model_choice == "FPN" else mask2former_model(tensor_image)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        mask_colorized = resize_and_colorize_mask(mask, original_size, CLASS_COLORS)

        st.image(mask_colorized, caption="Masque segmenté", use_container_width=True)
        st.image(Image.open(urllib.request.urlopen(mask_url)).convert("RGB"), caption="Masque réel", use_container_width=True)

    except Exception as e:
        st.error(f"⚠ Erreur lors de la segmentation : {e}")
