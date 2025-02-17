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

    if not os.path.exists(fpn_model_path):
        urllib.request.urlretrieve(fpn_url, fpn_model_path)
    
    if not os.path.exists(mask2former_model_path):
        urllib.request.urlretrieve(mask2former_url, mask2former_model_path)

    fpn_model = torch.load(fpn_model_path, map_location=torch.device("cpu"))
    fpn_model.eval()
    
    mask2former_model = torch.load(mask2former_model_path, map_location=torch.device("cpu"))

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
    
    st.header("Dimensions des Images et Masques")
    st.write("**2048x1024 : 5000 occurrences**")
    
    st.header("Analyse des Formats de Fichiers")
    file_formats = {"Format": [".png (images)", ".png (masques)", ".json (masques)"], "Nombre": [5000, 15000, 5000]}
    df_formats = pd.DataFrame(file_formats)
    st.table(df_formats)
    
    st.header("Exemples d'Images et Masques")
    sample_image_url = "https://storage.googleapis.com/p9-dashboard-storage/Dataset/images/lindau_000000_000019_leftImg8bit.png"
    sample_mask_url = "https://storage.googleapis.com/p9-dashboard-storage/Dataset/masks/lindau_000000_000019_gtFine_color.png"
    col1, col2 = st.columns(2)
    with col1:
        st.image(sample_image_url, caption="Image Originale", use_container_width=True)
    with col2:
        st.image(sample_mask_url, caption="Masque Correspondant", use_container_width=True)

# Ajout de la page Test des modèles
if page == "Test des modèles":
    st.title("Test des Modèles de Segmentation")
    image_url = st.selectbox("Sélectionner une image :", [
        "lindau_000000_000019_leftImg8bit.png",
        "frankfurt_000000_000294_leftImg8bit.png",
        "munster_000000_000019_leftImg8bit.png"
    ])
    
    image_path = f"https://storage.googleapis.com/p9-dashboard-storage/Dataset/images/{image_url}"
    
    image = Image.open(urllib.request.urlopen(image_path))
    image_tensor = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
    
    output = fpn_model(image_tensor)[0]
    segmentation_map = torch.argmax(output, dim=0).cpu().detach().numpy()
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_path, caption="Image Originale", use_container_width=True)
    with col2:
        st.image(segmentation_map, caption="Masque Prédit", use_column_width=True)
    
    st.write("Prédiction effectuée avec succès !")
