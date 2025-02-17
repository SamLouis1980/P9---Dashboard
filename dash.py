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

# Récupération dynamique des images disponibles sur GCS
@st.cache_data
def get_image_list():
    fs = gcsfs.GCSFileSystem()
    image_prefix = "p9-dashboard-storage/Dataset/images/"
    
    # Liste tous les fichiers disponibles dans le bucket sous le dossier 'images'
    images = fs.ls(image_prefix)
    
    # Ne garder que les noms des fichiers (sans le chemin complet)
    image_list = [img.split("/")[-1] for img in images if img.endswith(".png")]
    
    return sorted(image_list)  # Trier les images pour une meilleure lisibilité

# Charger la liste des images une seule fois
image_list = get_image_list()

# Interface utilisateur pour le test des modèles
if page == "Test des modèles":
    st.title("Test de Segmentation avec les Modèles")

    # Sélecteur d’image parmi celles disponibles sur GCS
    image_choice = st.selectbox("Choisissez une image à segmenter", image_list)

    # Sélecteur du modèle à utiliser
    model_choice = st.radio("Choisissez le modèle", ["FPN", "Mask2Former"])

    st.write(f"Vous avez sélectionné **{image_choice}** avec le modèle **{model_choice}**")

    # Chargement de l'image d'entrée
    image_url = f"https://storage.googleapis.com/p9-dashboard-storage/Dataset/images/{image_choice}"
    st.image(image_url, caption="Image d'entrée", use_column_width=True)

    # Exécution du modèle sélectionné
    model = fpn_model if model_choice == "FPN" else mask2former_model

    # Prétraitement de l'image pour le modèle
    image_pil = Image.open(urllib.request.urlopen(image_url))
    input_tensor, original_size = preprocess_image(image_pil, (512, 512))  # Redimensionner au format du modèle
    
    with torch.no_grad():
        input_tensor = torch.tensor(input_tensor).permute(0, 3, 1, 2).float()  # Réorganiser les dimensions
        output = model(input_tensor)  # Passage dans le modèle
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Extraction du masque final

    # Post-traitement et colorisation
    segmented_mask = resize_and_colorize_mask(mask, original_size, CLASS_COLORS)

    # Affichage du masque segmenté
    st.image(segmented_mask, caption="Masque segmenté par le modèle", use_column_width=True)

    # Chargement du vrai masque correspondant
    mask_choice = image_choice.replace("leftImg8bit", "gtFine_color")  # Adapter le nom du fichier masque
    mask_url = f"https://storage.googleapis.com/p9-dashboard-storage/Dataset/masks/{mask_choice}"
    st.image(mask_url, caption="Masque réel", use_column_width=True)
