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

# 🔹 Configuration du layout
st.set_page_config(layout="wide")

# 🔹 Titre principal
st.title("Dashboard")

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

# Initialisation des variables dans session_state si elles n'existent pas encore
for var in ["overlay_fpn", "overlay_convnext"]:
    if var not in st.session_state:
        st.session_state[var] = None
        
# Configuration du bucket GCS (Public)
BUCKET_NAME = "p9-dashboard-storage"
IMAGE_FOLDER = "Dataset/images"
MASK_FOLDER = "Dataset/masks"
AUGMENTED_FOLDER = "Dataset/transformed_images"

# Chemins vers les modèles sur GCS
FPN_MODEL_URL = f"https://storage.googleapis.com/{BUCKET_NAME}/Models/fpn_best.pth"
CONVNEXT_MODEL_URL = f"https://storage.googleapis.com/{BUCKET_NAME}/Models/convnext_model_fp16.pth"

# Téléchargement et chargement des modèles
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

# Liste manuelle des images (évite les appels à GCS)
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

    available_augmented_images = [
        "lindau_000001_000019_augmented.png",
        "lindau_000002_000019_augmented.png",
        "lindau_000003_000019_augmented.png",
        "lindau_000004_000019_augmented.png",
        "lindau_000005_000019_augmented.png",
    ]
    
    return available_images, available_masks, available_augmented_images

available_images, available_masks, available_augmented_images = get_available_images_and_masks()

# Stocker les résultats de segmentation et l'état du traitement
if "segmentation_result" not in st.session_state:
    st.session_state.segmentation_result = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# 🔹 Menu déroulant
page = st.selectbox("", ["Menu", "EDA", "Résultats des modèles", "Test des modèles"], key="menu_selection", label_visibility="collapsed")

# Page Menu
if page == "Menu":
    
    # 🔹 Création de la mise en page en 2x2 avec des colonnes
    col1, col2 = st.columns(2)  # 2 colonnes pour chaque ligne

    # 🔹 Première ligne (Présentation du projet & EDA)
    with col1:
        st.markdown(
            """
            <div style="
                background-color: #2C2F33;
                padding: 20px;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 5px;">
                <h2>🏆 Présentation du projet</h2>
                <p>Ce projet compare les performances des modèles <b>FPN avec ResNet</b> et <b>ConvNeXt</b> pour la segmentation d'images.</p>
            </div>
            """, unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="
                background-color: #2C2F33;
                padding: 20px;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 5px;">
                <h2>📊 Exploration des Données (EDA)</h2>
                <p>Analyse du dataset Cityscapes : <b>distribution des classes</b>, visualisation des images, et effets de la <b>data augmentation</b>.</p>
            </div>
            """, unsafe_allow_html=True
        )

    # 🔹 Espacement entre les deux lignes
    st.markdown("<br>", unsafe_allow_html=True)

    # 🔹 Deuxième ligne (Résultats des modèles & Test des modèles)
    col3, col4 = st.columns(2)  # Nouvelle ligne avec 2 colonnes

    with col3:
        st.markdown(
            """
            <div style="
                background-color: #2C2F33;
                padding: 20px;
                border-radius: 10px;
                color: white;
                text-align: center;">
                <h2>📈 Résultats des Modèles</h2>
                <p>Comparaison des performances : <b>IoU, Dice Score</b>, et <b>courbes d'apprentissage</b> des modèles testés.</p>
            </div>
            """, unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div style="
                background-color: #2C2F33;
                padding: 20px;
                border-radius: 10px;
                color: white;
                text-align: center;">
                <h2>🖼️ Test des Modèles</h2>
                <p>Testez la segmentation en direct : <b>sélectionnezz une image</b> et observez le résultat du modèle.</p>
            </div>
            """, unsafe_allow_html=True
        )
    
# 🔹 Page EDA

# 🔹 Génération des URLs complètes des images et masques en utilisant les variables existantes
image_urls = [f"https://storage.googleapis.com/{BUCKET_NAME}/{IMAGE_FOLDER}/{img}" for img in available_images]
mask_urls = [f"https://storage.googleapis.com/{BUCKET_NAME}/{MASK_FOLDER}/{mask}" for mask in available_masks]
augmented_image_urls = [f"https://storage.googleapis.com/{BUCKET_NAME}/Dataset/transformed_images/{img.replace('_leftImg8bit.png', '_augmented.png')}" for img in available_images]

if page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    # 🔹 Chargement du fichier CSV depuis Google Cloud Storage
    @st.cache_data
    def load_class_distribution():
        """Charge le fichier CSV contenant la distribution des classes."""
        CSV_URL = "https://storage.googleapis.com/p9-dashboard-storage/Dataset/class_distribution/cityscapes_class_distribution.csv"
        return pd.read_csv(CSV_URL)

    df_classes = load_class_distribution()

    # 🔹 Slider interactif pour choisir combien de classes afficher
    num_classes = st.slider("Nombre de classes à afficher :", min_value=10, max_value=34, value=20, step=5)
    df_filtered = df_classes.head(num_classes)

    # 🔹 Titre unique pour l'ensemble des blocs
    st.markdown("### 📊 Distribution des Classes dans Cityscapes")

    # 🔹 Affichage en 2 colonnes (tableau à gauche, graphique à droite)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div style="
                background-color: #2C2F33;
                padding: 15px;
                border-radius: 10px;
                color: white;">
            """, unsafe_allow_html=True
        )
        st.dataframe(df_filtered, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div style="
                background-color: #2C2F33;
                padding: 15px;
                border-radius: 10px;
                color: white;">
            """, unsafe_allow_html=True
        )
        # 🔹 Création du graphique interactif avec Plotly
        import plotly.express as px
        fig = px.bar(
            df_filtered,
            x="Class Name", 
            y="Pixel Count", 
            title="",
            labels={"Pixel Count": "Nombre de Pixels", "Class Name": "Classe"},
            color="Pixel Count",
            color_continuous_scale="blues"
        )
        fig.update_layout(xaxis_tickangle=-45)

        # 🔹 Affichage du graphique interactif
        st.plotly_chart(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # 🔹 Affichage du carrousel interactif des images et masques
    st.markdown("### 🎠 Exemples d'Images et Masques Segmentés")
    
    # Sélecteur d’image avec un slider
    img_index = st.slider("Sélectionnez une image :", min_value=0, max_value=len(image_urls)-1, value=0)

    # Chargement des images sélectionnées
    image = Image.open(urllib.request.urlopen(image_urls[img_index]))
    mask = Image.open(urllib.request.urlopen(mask_urls[img_index]))

    # Affichage en deux colonnes
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="📸 Image originale", use_container_width=True)

    with col2:
        st.image(mask, caption="🎭 Masque segmenté", use_container_width=True)

    # 🔹 Affichage du carrousel interactif des images augmentées
    st.markdown("### 🎭 Effets de la Data Augmentation")

    # 🔹 Sélecteur d’image avec un slider
    img_index_aug = st.slider("Sélectionnez une image :", min_value=0, max_value=len(augmented_image_urls)-1, value=0, key="aug_slider")

    # Chargement des images sélectionnées
    original_image = Image.open(urllib.request.urlopen(image_urls[img_index_aug]))
    augmented_image = Image.open(urllib.request.urlopen(augmented_image_urls[img_index_aug]))

    # 🔹 Affichage en deux colonnes équilibrées comme pour les masques
    col1, col2 = st.columns(2)

    with col1:
        st.image(original_image, caption="📸 Image originale", use_container_width=True)

    with col2:
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        st.image(augmented_image, caption="🛠️ Image après Data Augmentation", width=int(original_image.width * 0.8))  
        st.markdown("</div>", unsafe_allow_html=True)

# Page Résultats des modèles
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
    """Exécute la segmentation avec les deux modèles en parallèle et met à jour l'interface."""
    print("Début de la segmentation...")  # Debug
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

    # Forcer la mise à jour de l'interface
    time.sleep(0.5)  # Petit délai pour éviter une mise à jour trop rapide
    st.rerun()

# Initialisation des variables dans session_state si elles n'existent pas encore
if "segmentation_fpn" not in st.session_state:
    st.session_state.segmentation_fpn = None

if "segmentation_convnext" not in st.session_state:
    st.session_state.segmentation_convnext = None

# Page Test des modèles
if page == "Test des modèles":
    st.title("Test de Segmentation avec les Modèles")

    image_choice = st.selectbox("Choisissez une image à segmenter", available_images)

    # URL de l’image
    image_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{IMAGE_FOLDER}/{image_choice}"

    try:
        # Chargement et affichage de l’image d’entrée
        image = Image.open(urllib.request.urlopen(image_url)).convert("RGB")
        st.image(image, caption="Image d'entrée", use_container_width=True)

        # Prétraitement de l’image avant passage dans le modèle
        input_size = (512, 512)
        image_resized, original_size = preprocess_image(image, input_size)
        tensor_image = torch.tensor(image_resized).permute(0, 3, 1, 2).float()

        # Bouton pour lancer la segmentation avec les deux modèles
        if st.button("Lancer la segmentation"):
            print("Bouton cliqué !")  # Debug

            # Réinitialiser les résultats précédents
            st.session_state.overlay_fpn = None
            st.session_state.overlay_convnext = None

            # Exécuter la segmentation en parallèle
            with st.spinner("Segmentation en cours..."):
                with torch.no_grad():
                    # FPN - Prédiction et post-traitement
                    output_fpn = fpn_model(tensor_image)  # FPN en FP32
                    mask_fpn = torch.argmax(output_fpn, dim=1).squeeze().cpu().numpy()
                    mask_fpn_colorized = resize_and_colorize_mask(mask_fpn, original_size, CLASS_COLORS)

                    # ConvNeXt - Prédiction et post-traitement
                    output_convnext = convnext_model(tensor_image.half())  # ConvNeXt en FP16
                    mask_convnext = torch.argmax(output_convnext, dim=1).squeeze().cpu().numpy()
                    mask_convnext_colorized = resize_and_colorize_mask(mask_convnext, original_size, CLASS_COLORS)

                    # Superposition des masques sur l'image d'origine
                    overlay_fpn = Image.blend(image, mask_fpn_colorized, alpha=0.5)  # Transparence 50%
                    overlay_convnext = Image.blend(image, mask_convnext_colorized, alpha=0.5)  # Transparence 50%

                # Sauvegarder uniquement les superpositions dans la session
                st.session_state.overlay_fpn = overlay_fpn
                st.session_state.overlay_convnext = overlay_convnext

                # Libérer la mémoire après inférence
                torch.cuda.empty_cache()
                del tensor_image, output_fpn, output_convnext
                gc.collect()

            print("Segmentation terminée !")  # Debug

        # Affichage des superpositions uniquement
        if st.session_state.overlay_fpn is not None and st.session_state.overlay_convnext is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.overlay_fpn, caption="Superposition - FPN", use_container_width=True)
            with col2:
                st.image(st.session_state.overlay_convnext, caption="Superposition - ConvNeXt", use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors du chargement des images : {e}")
