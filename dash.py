import streamlit as st
import pandas as pd
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
from utils import preprocess_image, resize_and_colorize_mask, CLASS_COLORS

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# 🔹 Configuration du layout
st.set_page_config(page_title="Dashboard", layout="wide")

# 🔹 Titre principal
st.title("Dashboard de Segmentation d'Images")

# 🔹 Barre latérale pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à :", ["🏠 Accueil", "📊 Analyse exploratoire", "📈 Résultats des modèles", "🖼️ Test des modèles"])

# Configuration du bucket GCS
BUCKET_NAME = "p9-dashboard-storage"
IMAGE_FOLDER = "Dataset/images"
MASK_FOLDER = "Dataset/masks"

# 🔹 Chargement des modèles
@st.cache_resource
def load_models():
    """Télécharge et charge les modèles depuis Google Cloud Storage."""
    fpn_model_path = "fpn_best.pth"
    convnext_model_path = "convnext_model_fp16.pth"
    
    # Télécharger les fichiers s'ils ne sont pas présents
    urllib.request.urlretrieve(f"https://storage.googleapis.com/{BUCKET_NAME}/Models/fpn_best.pth", fpn_model_path)
    urllib.request.urlretrieve(f"https://storage.googleapis.com/{BUCKET_NAME}/Models/convnext_model_fp16.pth", convnext_model_path)
    
    # Charger les modèles
    fpn_model = torch.load(fpn_model_path, map_location=torch.device("cpu"))
    fpn_model.eval()
    convnext_model = torch.load(convnext_model_path, map_location=torch.device("cpu"))
    convnext_model.eval()
    
    return fpn_model, convnext_model

fpn_model, convnext_model = load_models()

# 🔹 Chargement des données d'entraînement
@st.cache_data
def load_results():
    resnet_results = pd.read_csv(f"https://storage.googleapis.com/{BUCKET_NAME}/Resultats/resnet_results.csv")
    convnext_results = pd.read_csv(f"https://storage.googleapis.com/{BUCKET_NAME}/Resultats/convnext_results.csv")
    resnet_pixel = pd.read_csv(f"https://storage.googleapis.com/{BUCKET_NAME}/Resultats/resnet_pixel.csv", encoding="ISO-8859-1")
    convnext_pixel = pd.read_csv(f"https://storage.googleapis.com/{BUCKET_NAME}/Resultats/convnext_pixels.csv", encoding="ISO-8859-1")
    return resnet_results, convnext_results, resnet_pixel, convnext_pixel

# 🔹 Navigation entre les pages
def accueil():
    st.header("🏠 Bienvenue sur le Dashboard")
    st.write("Ce projet compare les performances des modèles **FPN avec ResNet** et **ConvNeXt** pour la segmentation d'images.")

def analyse_exploratoire():
    st.header("📊 Analyse exploratoire des données")
    df_classes = pd.read_csv(f"https://storage.googleapis.com/{BUCKET_NAME}/Dataset/class_distribution/cityscapes_class_distribution.csv")
    st.write(df_classes.head(10))

def resultats_modeles():
    st.header("📈 Analyse des Résultats des Modèles")
    resnet_results, convnext_results, resnet_pixel, convnext_pixel = load_results()
    
    st.subheader("📊 Courbes d'Apprentissage")
    metric_choice = st.selectbox("Sélectionnez une métrique :", ["Loss", "IoU Score", "Dice Score"])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resnet_results["Epoch"], y=resnet_results[f"Val {metric_choice}"], mode='lines', name=f'ResNet {metric_choice}'))
    fig.add_trace(go.Scatter(x=convnext_results["Epoch"], y=convnext_results[f"Val {metric_choice}"], mode='lines', name=f'ConvNeXt {metric_choice}'))
    fig.update_layout(xaxis_title="Epochs", yaxis_title=metric_choice)
    st.plotly_chart(fig)

def test_modeles():
    st.header("🖼️ Test de Segmentation avec les Modèles")
    image_choice = st.selectbox("Choisissez une image à segmenter", ["lindau_000001_000019_leftImg8bit.png", "lindau_000002_000019_leftImg8bit.png"])
    
    image_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{IMAGE_FOLDER}/{image_choice}"
    image = Image.open(urllib.request.urlopen(image_url)).convert("RGB")
    st.image(image, caption="Image d'entrée", use_container_width=True)
    
    if st.button("Lancer la segmentation"):
        with torch.no_grad():
            tensor_image = torch.tensor(preprocess_image(image, (512, 512))).permute(0, 3, 1, 2).float()
            mask_fpn = torch.argmax(fpn_model(tensor_image), dim=1).squeeze().cpu().numpy()
            mask_convnext = torch.argmax(convnext_model(tensor_image.half()), dim=1).squeeze().cpu().numpy()
            
            overlay_fpn = Image.blend(image, resize_and_colorize_mask(mask_fpn, image.size, CLASS_COLORS), alpha=0.5)
            overlay_convnext = Image.blend(image, resize_and_colorize_mask(mask_convnext, image.size, CLASS_COLORS), alpha=0.5)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(overlay_fpn, caption="Superposition - FPN", use_container_width=True)
            with col2:
                st.image(overlay_convnext, caption="Superposition - ConvNeXt", use_container_width=True)
            
            torch.cuda.empty_cache()
            gc.collect()

# 🔹 Exécution de la page sélectionnée
if page == "🏠 Accueil":
    accueil()
elif page == "📊 Analyse exploratoire":
    analyse_exploratoire()
elif page == "📈 Résultats des modèles":
    resultats_modeles()
elif page == "🖼️ Test des modèles":
    test_modeles()

    except Exception as e:
        st.error(f"Erreur lors du chargement des images : {e}")
