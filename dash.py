import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
from PIL import Image
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

# Définition des URLs des modèles sur GCS
fpn_url = "https://storage.googleapis.com/p9-dashboard-storage/Models/fpn_best.pth"
mask2former_url = "https://storage.googleapis.com/p9-dashboard-storage/Models/mask2former_best.pth"

# Téléchargement et chargement des modèles depuis GCS
fpn_model_path = "fpn_best.pth"
mask2former_model_path = "mask2former_best.pth"

urllib.request.urlretrieve(fpn_url, fpn_model_path)
urllib.request.urlretrieve(mask2former_url, mask2former_model_path)

fpn_model = torch.load(fpn_model_path, map_location=torch.device("cpu"), weights_only=False)
fpn_model.eval()
mask2former_model = torch.load(mask2former_model_path, map_location=torch.device("cpu"), weights_only=False)

st.write("Modèles chargés avec succès !")

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

# Page Résultats des modèles
if page == "Résultats des modèles":
    st.title("Analyse des Résultats des Modèles")
    st.subheader("Comparaison des métriques d'entraînement")
    
    # Chargement des résultats
    fpn_results = pd.read_csv("https://storage.googleapis.com/p9-dashboard-storage/Models/fpn_results.csv")
    mask2former_results = pd.read_csv("https://storage.googleapis.com/p9-dashboard-storage/Models/mask2former_results.csv")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpn_results["Epoch"], y=fpn_results["Val Loss"], mode='lines', name='FPN - Validation Loss'))
    fig.add_trace(go.Scatter(x=fpn_results["Epoch"], y=fpn_results["Val IoU"], mode='lines', name='FPN - Validation IoU Score'))
    fig.add_trace(go.Scatter(x=mask2former_results["Epoch"], y=mask2former_results["Val Loss"], mode='lines', name='Mask2Former - Validation Loss'))
    fig.add_trace(go.Scatter(x=mask2former_results["Epoch"], y=mask2former_results["Val IoU"], mode='lines', name='Mask2Former - Validation IoU Score'))
    
    fig.update_layout(title="Évolution des métriques", xaxis_title="Epoch", yaxis_title="Score", template="plotly_white")
    st.plotly_chart(fig)
