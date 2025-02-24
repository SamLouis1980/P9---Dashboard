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

# Chargement et mise en cache des modèles
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

    fpn_model = torch.load(fpn_model_path, map_location=torch.device("cpu"))
    fpn_model.eval()
    
    mask2former_model = torch.load(mask2former_model_path, map_location=torch.device("cpu"))
    mask2former_model.eval()

    return fpn_model, mask2former_model

fpn_model, mask2former_model = load_models()
st.write("Modèles chargés avec succès")

# Création de la sidebar
st.sidebar.title("Menu")
page = st.sidebar.radio("Aller à :", [
    "EDA", 
    "Résultats des modèles", 
    "Test des modèles"
])

# Page EDA
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
    
    fig, ax = plt.subplots()
    ax.bar(df_classes["Classe"], df_classes["Pixels"], color="skyblue")
    plt.xticks(rotation=90)
    plt.xlabel("Classes")
    plt.ylabel("Nombre de Pixels")
    plt.title("Répartition des Pixels par Classe")
    st.pyplot(fig)

# Chargement des résultats mis en cache
@st.cache_data
def load_results():
    fpn_results = pd.read_csv("https://storage.googleapis.com/p9-dashboard-storage/Resultats/fpn_results.csv")
    mask2former_results = pd.read_csv("https://storage.googleapis.com/p9-dashboard-storage/Resultats/mask2former_results.csv")
    return fpn_results, mask2former_results

fpn_results, mask2former_results = load_results()

if page == "Résultats des modèles":
    st.title("Analyse des Résultats des Modèles")
    st.subheader("Comparaison des métriques d'entraînement")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpn_results["Epoch"], y=fpn_results["Val Loss"], mode='lines', name='FPN - Validation Loss'))
    fig.add_trace(go.Scatter(x=fpn_results["Epoch"], y=fpn_results["Val IoU"], mode='lines', name='FPN - Validation IoU Score'))
    fig.add_trace(go.Scatter(x=mask2former_results["Epoch"], y=mask2former_results["Val Loss"], mode='lines', name='Mask2Former - Validation Loss'))
    fig.add_trace(go.Scatter(x=mask2former_results["Epoch"], y=mask2former_results["Val IoU"], mode='lines', name='Mask2Former - Validation IoU Score'))
    
    fig.update_layout(title="Évolution des métriques", xaxis_title="Epoch", yaxis_title="Score", template="plotly_white")
    st.plotly_chart(fig)

if page == "Test des modèles":
    st.title("Test de Segmentation avec les Modèles")
    
    image_list = ["image1.png", "image2.png", "image3.png"]
    image_choice = st.selectbox("Choisissez une image à segmenter", image_list)
    
    model_choice = st.radio("Choisissez le modèle", ["FPN", "Mask2Former"])
    
    st.write(f"Vous avez sélectionné {image_choice} avec le modèle {model_choice}")
    st.image(f"https://storage.googleapis.com/p9-dashboard-storage/Dataset/images/{image_choice}", caption="Image d'entrée")
    
    st.write("Masque segmenté")
    st.image(f"https://storage.googleapis.com/p9-dashboard-storage/Dataset/masks/{image_choice}", caption="Masque segmenté")
