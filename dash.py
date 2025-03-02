import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import urllib.request
from PIL import Image
import numpy as np
import warnings
import plotly.graph_objects as go
import gc
from utils import preprocess_image, resize_and_colorize_mask, FPN_Segmenter, FPN_ConvNeXtV2_Segmenter, CLASS_COLORS

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# 🔹 Configuration du layout avec sidebar
st.set_page_config(layout="wide")

# 🔹 Appliquer le style WCAG (contrastes et lisibilité)
st.markdown("""
    <style>
        body { color: black; background-color: white; }
        .stSidebar { border: 2px solid black !important; }
        h1, h2, h3, h4, h5, h6 { color: #1E90FF !important; }
    </style>
""", unsafe_allow_html=True)

# 🔹 Sidebar pour la navigation
page = st.sidebar.radio("Navigation", ["Accueil", "Analyse exploratoire", "Résultats des modèles", "Test des modèles"])

# 🔹 Titre principal
st.title("Dashboard")

# 🔹 Charger les modèles depuis Google Cloud Storage
@st.cache_resource
def load_models():
    fpn_model_path = "fpn_best.pth"
    convnext_model_path = "convnext_model_fp16.pth"

    if not os.path.exists(fpn_model_path):
        urllib.request.urlretrieve("https://storage.googleapis.com/p9-dashboard-storage/Models/fpn_best.pth", fpn_model_path)

    if not os.path.exists(convnext_model_path):
        urllib.request.urlretrieve("https://storage.googleapis.com/p9-dashboard-storage/Models/convnext_model_fp16.pth", convnext_model_path)

    fpn_model = torch.load(fpn_model_path, map_location=torch.device("cpu"))
    fpn_model.eval()

    convnext_model = torch.load(convnext_model_path, map_location=torch.device("cpu"))
    convnext_model.eval()

    return fpn_model, convnext_model

fpn_model, convnext_model = load_models()

# 🔹 Charger la liste des images et masques
@st.cache_data
def get_available_images():
    return [
        "lindau_000001_000019_leftImg8bit.png",
        "lindau_000002_000019_leftImg8bit.png"
    ]

available_images = get_available_images()

# ✅ **Accueil**
if page == "Accueil":
    st.subheader("🏆 Présentation du projet")
    st.write("Comparaison des performances des modèles **FPN** et **ConvNeXt** pour la segmentation d'images.")

    st.subheader("📊 Exploration des Données")
    st.write("Analyse du dataset Cityscapes : **distribution des classes**, **visualisation des images** et **effets de la data augmentation**.")

    st.subheader("📈 Résultats des Modèles")
    st.write("Comparaison des performances sur les métriques : **IoU, Dice Score**, et **courbes d'apprentissage**.")

    st.subheader("🖼️ Test des Modèles")
    st.write("Testez la segmentation en direct en **sélectionnant une image**.")

# ✅ **Analyse exploratoire**
elif page == "Analyse exploratoire":
    st.subheader("📊 Distribution des Classes")
    
    # Charger la distribution des classes
    @st.cache_data
    def load_class_distribution():
        return pd.read_csv("https://storage.googleapis.com/p9-dashboard-storage/Dataset/class_distribution/cityscapes_class_distribution.csv")

    df_classes = load_class_distribution()
    st.dataframe(df_classes)

    st.subheader("🎠 Exemples d'Images et Masques")
    img_index = st.slider("Sélectionnez une image :", 0, len(available_images)-1, 0)
    image_url = f"https://storage.googleapis.com/p9-dashboard-storage/Dataset/images/{available_images[img_index]}"

    # Affichage avec description pour accessibilité
    st.image(image_url, caption="📸 Image originale du dataset Cityscapes", use_column_width=True)

# ✅ **Résultats des modèles**
elif page == "Résultats des modèles":
    st.subheader("📊 Courbes d'Apprentissage")

    # Charger les résultats
    @st.cache_data
    def load_results():
        return pd.read_csv("https://storage.googleapis.com/p9-dashboard-storage/Resultats/resnet_results.csv")

    df_results = load_results()
    st.line_chart(df_results[["Epoch", "Val IoU"]])

# ✅ **Test des modèles**
elif page == "Test des modèles":
    st.subheader("Test de Segmentation")

    image_choice = st.selectbox("Choisissez une image à segmenter", available_images)
    image_url = f"https://storage.googleapis.com/p9-dashboard-storage/Dataset/images/{image_choice}"

    try:
        image = Image.open(urllib.request.urlopen(image_url)).convert("RGB")
        st.image(image, caption="Image d'entrée pour la segmentation", use_column_width=True)

        # Traitement et segmentation
        input_size = (512, 512)
        image_resized, original_size = preprocess_image(image, input_size)
        tensor_image = torch.tensor(image_resized).permute(0, 3, 1, 2).float()

        if st.button("Lancer la segmentation"):
            with st.spinner("Segmentation en cours..."):
                with torch.no_grad():
                    output_fpn = fpn_model(tensor_image)
                    mask_fpn = torch.argmax(output_fpn, dim=1).squeeze().cpu().numpy()
                    mask_fpn_colorized = resize_and_colorize_mask(mask_fpn, original_size, CLASS_COLORS)

                    output_convnext = convnext_model(tensor_image.half())
                    mask_convnext = torch.argmax(output_convnext, dim=1).squeeze().cpu().numpy()
                    mask_convnext_colorized = resize_and_colorize_mask(mask_convnext, original_size, CLASS_COLORS)

                    overlay_fpn = Image.blend(image, mask_fpn_colorized, alpha=0.5)
                    overlay_convnext = Image.blend(image, mask_convnext_colorized, alpha=0.5)

                # Affichage avec descriptions WCAG
                col1, col2 = st.columns(2)
                with col1:
                    st.image(overlay_fpn, caption="Superposition de la segmentation avec FPN", use_column_width=True)
                with col2:
                    st.image(overlay_convnext, caption="Superposition de la segmentation avec ConvNeXt", use_column_width=True)

    except Exception as e:
        st.error(f"Erreur : {e}")
