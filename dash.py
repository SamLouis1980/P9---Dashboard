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
import gc
import threading
import time
from utils import preprocess_image, resize_and_colorize_mask, FPN_Segmenter, FPN_ConvNeXtV2_Segmenter, CLASS_COLORS

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

st.markdown(
    """
    <style>
        /* Changer la couleur des titres */
        h1, h2, h3, h4, h5, h6 {
            color: #1E90FF !important;
        }

        /* Style des tableaux */
        table {
            background-color: #121212 !important;
            color: #FFFFFF !important;
            border-radius: 10px;
        }

        /* Style des images pour les rendre visibles */
        img {
            border-radius: 10px;
            background-color: #000000;
        }

        /* Style des boutons */
        button {
            background-color: #1E90FF !important;
            color: #FFFFFF !important;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 🔹 Configuration du bucket GCS (Public)
BUCKET_NAME = "p9-dashboard-storage"
IMAGE_FOLDER = "Dataset/images"
MASK_FOLDER = "Dataset/masks"

# 🔹 Chemins vers les modèles sur GCS
FPN_MODEL_URL = f"https://storage.googleapis.com/{BUCKET_NAME}/Models/fpn_best.pth"
CONVNEXT_MODEL_URL = f"https://storage.googleapis.com/{BUCKET_NAME}/Models/convnext_model_fp16.pth"

# 🔹 Téléchargement et chargement des modèles
@st.cache_resource
def load_models():
    """Télécharge et charge les modèles depuis Google Cloud Storage."""
    fpn_model_path = "fpn_best.pth"
    convnext_model_path = "convnext_model_fp16.pth"

    # Télécharger les fichiers depuis GCS s'ils ne sont pas déjà présents
    if not os.path.exists(fpn_model_path):
        urllib.request.urlretrieve(FPN_MODEL_URL, fpn_model_path)
    
    if not os.path.exists(convnext_model_path):
        urllib.request.urlretrieve(CONVNEXT_MODEL_URL, convnext_model_path)

    # Charger les modèles
    fpn_model = torch.load(fpn_model_path, map_location=torch.device("cpu"))
    fpn_model.eval()  # Mettre en mode évaluation

    convnext_model = torch.load(convnext_model_path, map_location=torch.device("cpu"))
    convnext_model.eval()  # Mettre en mode évaluation

    return fpn_model, convnext_model

# Charger les modèles
fpn_model, convnext_model = load_models()

# 🔹 Liste manuelle des images (évite les appels à GCS)
@st.cache_data
def get_available_images_and_masks():
    """Retourne les noms des images et masques présents dans GCS."""
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

# Stocker les résultats de segmentation et l'état du traitement
if "segmentation_result" not in st.session_state:
    st.session_state.segmentation_result = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# 🔹 Sidebar Navigation
st.sidebar.title("Menu")
page = st.sidebar.radio("Aller à :", ["EDA", "Résultats des modèles", "Test des modèles"])

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

    # 🔹 Distribution des classes
    st.header("Distribution des Classes dans les Masques")
    class_distribution = {
        "ID": [7, 11, 21, 26, 8, 1, 23, 3, 4, 2, 6, 17, 24, 22, 13, 9, 12, 20, 33, 15],
        "Classe": ["road", "fence", "truck", "void", "sidewalk", "ego vehicle", "train", "out of roi", "static", "rectification border",
                    "ground", "sky", "motorcycle", "bus", "traffic light", "building", "pole", "car", "void", "vegetation"],
        "Pixels": [2036416525, 1260636120, 879783988, 386328286, 336090793, 286002726, 221979646, 94111150, 83752079, 81359604,
                    75629728, 67789506, 67326424, 63949536, 48454166, 39065130, 36199498, 30448193, 22861233, 17860177]
    }
    df_classes = pd.DataFrame(class_distribution)
    st.table(df_classes.head(10))

    # 🔹 Affichage du graphique de répartition des classes
    fig, ax = plt.subplots()
    ax.bar(df_classes["Classe"], df_classes["Pixels"], color="skyblue")
    plt.xticks(rotation=90)
    plt.xlabel("Classes")
    plt.ylabel("Nombre de Pixels")
    plt.title("Répartition des Pixels par Classe")
    st.pyplot(fig)

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

def run_segmentation(tensor_image, original_size):
    """Exécute la segmentation avec les deux modèles en parallèle."""
    print("🚀 Début de la segmentation...")  # Debug
    st.session_state.processing = True  

    with torch.no_grad():
        # FPN (FP32)
        output_fpn = fpn_model(tensor_image)
        mask_fpn = torch.argmax(output_fpn, dim=1).squeeze().cpu().numpy()
        mask_fpn_colorized = resize_and_colorize_mask(mask_fpn, original_size, CLASS_COLORS)

        # ConvNeXt (FP16)
        output_convnext = convnext_model(tensor_image.half())
        mask_convnext = torch.argmax(output_convnext, dim=1).squeeze().cpu().numpy()
        mask_convnext_colorized = resize_and_colorize_mask(mask_convnext, original_size, CLASS_COLORS)

    # Stocker les résultats
    st.session_state.segmentation_fpn = mask_fpn_colorized
    st.session_state.segmentation_convnext = mask_convnext_colorized
    st.session_state.processing = False

    print("Segmentation terminée.")  # Debug

# Initialisation des variables dans session_state si elles n'existent pas encore
if "segmentation_fpn" not in st.session_state:
    st.session_state.segmentation_fpn = None

if "segmentation_convnext" not in st.session_state:
    st.session_state.segmentation_convnext = None

# 🔹 Page Test des modèles
if page == "Test des modèles":
    st.title("Test de Segmentation avec les Modèles")

    image_choice = st.selectbox("Choisissez une image à segmenter", available_images)

    # 🔹 URL de l’image d’entrée
    image_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{IMAGE_FOLDER}/{image_choice}"

    try:
        # 🔹 Chargement et affichage de l’image d’entrée
        image = Image.open(urllib.request.urlopen(image_url)).convert("RGB")
        st.image(image, caption="Image d'entrée", use_container_width=True)

        # 🔹 Prétraitement avant segmentation
        input_size = (512, 512)
        image_resized, original_size = preprocess_image(image, input_size)
        tensor_image = torch.tensor(image_resized).permute(0, 3, 1, 2).float()

        # 🔹 Bouton pour lancer la segmentation
        if st.button("Lancer la segmentation"):
            print("Bouton cliqué !")  # Debug
            st.session_state.segmentation_fpn = None
            st.session_state.segmentation_convnext = None

            with st.spinner("Segmentation en cours..."):
                threading.Thread(target=run_segmentation, args=(tensor_image, original_size)).start()

            print("Segmentation lancée en arrière-plan !")  # Debug

        # 🔹 Affichage du statut
        if st.session_state.processing:
            st.info("Segmentation en cours... Vous pouvez naviguer librement.")

        # 🔹 Afficher les images segmentées superposées uniquement si elles existent
        if st.session_state.segmentation_fpn is not None and st.session_state.segmentation_convnext is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(st.session_state.segmentation_fpn, caption="Superposition - FPN", use_container_width=True)

            with col2:
                st.image(st.session_state.segmentation_convnext, caption="Superposition - ConvNeXt", use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors du chargement des images : {e}")
