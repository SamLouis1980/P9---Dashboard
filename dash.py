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
import warnings
import plotly.graph_objects as go
import numpy as np
import requests
from io import BytesIO

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Définition du modèle FPN
class FPN_Segmenter(nn.Module):
    def __init__(self, num_classes=8):
        super(FPN_Segmenter, self).__init__()
        self.fpn_backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1").backbone
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        fpn_features = self.fpn_backbone(x)
        p2 = fpn_features['0']
        output = self.final_conv(p2)
        output = F.interpolate(output, size=(512, 512), mode="bilinear", align_corners=False)
        return output

# Mise en cache du chargement des modèles
@st.cache_resource
def load_models():
    fpn_model_path = "fpn_best.pth"
    mask2former_model_path = "mask2former_best.pth"

    fpn_url = "https://storage.googleapis.com/p9-dashboard-storage/Models/fpn_best.pth"
    mask2former_url = "https://storage.googleapis.com/p9-dashboard-storage/Models/mask2former_best.pth"

    # Vérifier si les fichiers existent déjà
    if not os.path.exists(fpn_model_path):
        urllib.request.urlretrieve(fpn_url, fpn_model_path)
    
    if not os.path.exists(mask2former_model_path):
        urllib.request.urlretrieve(mask2former_url, mask2former_model_path)

    # Charger les modèles
    fpn_model = torch.load(fpn_model_path, map_location=torch.device("cpu"), weights_only=False)
    fpn_model.eval()
    
    mask2former_model = torch.load(mask2former_model_path, map_location=torch.device("cpu"), weights_only=False)

    return fpn_model, mask2former_model

# Charger les modèles une seule fois
fpn_model, mask2former_model = load_models()
st.write("Modèles chargés avec succès")

# Création de la sidebar
st.sidebar.title("Menu")
page = st.sidebar.radio("Aller à :", [
    "EDA", 
    "Résultats des modèles", 
    "Test des modèles"
])

# Affichage de la page EDA
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
    
    st.header("Exemples d'Images et Masques")
    sample_image_url = "https://storage.googleapis.com/p9-dashboard-storage/Dataset/images/lindau_000000_000019_leftImg8bit.png"
    sample_mask_url = "https://storage.googleapis.com/p9-dashboard-storage/Dataset/masks/lindau_000000_000019_gtFine_color.png"
    col1, col2 = st.columns(2)
    with col1:
        st.image(sample_image_url, caption="Image Originale", use_container_width=True)
    with col2:
        st.image(sample_mask_url, caption="Masque Correspondant", use_container_width=True)

# Mise en cache du chargement des fichiers CSV
@st.cache_data
def load_results():
    fpn_results = pd.read_csv("https://storage.googleapis.com/p9-dashboard-storage/Resultats/fpn_results.csv")
    mask2former_results = pd.read_csv("https://storage.googleapis.com/p9-dashboard-storage/Resultats/mask2former_results.csv")
    return fpn_results, mask2former_results

# Charger les résultats une seule fois
fpn_results, mask2former_results = load_results()

# Page Résultats des modèles
if page == "Résultats des modèles":
    st.title("Analyse des Résultats des Modèles")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpn_results["Epoch"], y=fpn_results["Val Loss"], mode='lines', name='FPN - Validation Loss'))
    fig.add_trace(go.Scatter(x=fpn_results["Epoch"], y=fpn_results["Val IoU"], mode='lines', name='FPN - Validation IoU Score'))
    fig.add_trace(go.Scatter(x=mask2former_results["Epoch"], y=mask2former_results["Val Loss"], mode='lines', name='Mask2Former - Validation Loss'))
    fig.add_trace(go.Scatter(x=mask2former_results["Epoch"], y=mask2former_results["Val IoU"], mode='lines', name='Mask2Former - Validation IoU Score'))
    fig.update_layout(title="Évolution des métriques", xaxis_title="Epoch", yaxis_title="Score", template="plotly_white")
    st.plotly_chart(fig)

# Page Test des modèles
if page == "Test des modèles":
    st.title("Test des modèles de segmentation")
    image_list = ["lindau_000000_000019_leftImg8bit.png", "frankfurt_000000_000576_leftImg8bit.png"]
    selected_image = st.selectbox("Choisissez une image :", image_list)
    img_url = f"https://storage.googleapis.com/p9-dashboard-storage/Dataset/images/{selected_image}"
    mask_url = f"https://storage.googleapis.com/p9-dashboard-storage/Dataset/masks/{selected_image.replace('leftImg8bit', 'gtFine_color')}"
    response = requests.get(img_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    st.image(image, caption="Image Originale", use_container_width=True)
    
    model_choice = st.radio("Sélectionnez un modèle :", ["FPN", "Mask2Former"])
    model = fpn_model if model_choice == "FPN" else mask2former_model
    input_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor)
    prediction = prediction.argmax(dim=1).squeeze().numpy()
    st.image(prediction, caption="Masque Prédit", use_column_width=True)
