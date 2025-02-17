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

# 🎨 Palette de Cityscapes
CITYSCAPES_PALETTE = {
    0: (128, 64, 128), 1: (244, 35, 232), 2: (70, 70, 70), 3: (102, 102, 156),
    4: (190, 153, 153), 5: (153, 153, 153), 6: (250, 170, 30), 7: (220, 220, 0),
    8: (107, 142, 35), 9: (152, 251, 152), 10: (0, 130, 180), 11: (220, 20, 60),
    12: (255, 0, 0), 13: (0, 0, 142), 14: (0, 60, 100), 15: (0, 80, 100),
    16: (0, 0, 230), 17: (119, 11, 32)
}

# 📥 Prétraitement de l'image
def preprocess_image(image, input_size=(512, 512)):
    image = image.convert("RGB")
    original_size = image.size
    image = image.resize(input_size, Image.BILINEAR)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0), original_size

# 🎨 Post-traitement du masque
def resize_and_colorize_mask(mask, original_size, palette):
    mask = Image.fromarray(mask.astype(np.uint8))
    mask = mask.resize(original_size, Image.NEAREST)
    flat_palette = [value for color in palette.values() for value in color]
    mask.putpalette(flat_palette)
    return mask.convert("RGB")

# 🖼️ Définition du modèle FPN
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

# ⚡ Mise en cache du chargement des modèles
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

    fpn_model = torch.load(fpn_model_path, map_location=torch.device("cpu"), weights_only=False)
    fpn_model.eval()
    
    mask2former_model = torch.load(mask2former_model_path, map_location=torch.device("cpu"), weights_only=False)

    return fpn_model, mask2former_model

# Charger les modèles une seule fois
fpn_model, mask2former_model = load_models()

# 🎛️ Sidebar Menu
st.sidebar.title("Menu")
page = st.sidebar.radio("Aller à :", ["EDA", "Résultats des modèles", "Test des modèles"])

# 📊 Page EDA
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
    df_classes = pd.DataFrame({
        "Classe": ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                   "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"],
        "Pixels": [2036416525, 1260636120, 879783988, 386328286, 336090793, 286002726, 221979646, 94111150, 83752079,
                   81359604, 75629728, 67789506, 67326424, 63949536, 48454166, 39065130, 36199498, 30448193, 22861233]
    })
    st.table(df_classes)

    fig, ax = plt.subplots()
    ax.bar(df_classes["Classe"], df_classes["Pixels"], color="skyblue")
    plt.xticks(rotation=90)
    plt.xlabel("Classes")
    plt.ylabel("Nombre de Pixels")
    plt.title("Répartition des Pixels par Classe")
    st.pyplot(fig)

# 📈 Page Résultats des modèles
if page == "Résultats des modèles":
    st.title("Analyse des Résultats des Modèles")

    @st.cache_data
    def load_results():
        fpn_results = pd.read_csv("https://storage.googleapis.com/p9-dashboard-storage/Resultats/fpn_results.csv")
        mask2former_results = pd.read_csv("https://storage.googleapis.com/p9-dashboard-storage/Resultats/mask2former_results.csv")
        return fpn_results, mask2former_results

    fpn_results, mask2former_results = load_results()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpn_results["Epoch"], y=fpn_results["Val Loss"], mode='lines', name='FPN - Validation Loss'))
    fig.add_trace(go.Scatter(x=fpn_results["Epoch"], y=fpn_results["Val IoU"], mode='lines', name='FPN - Validation IoU Score'))
    fig.add_trace(go.Scatter(x=mask2former_results["Epoch"], y=mask2former_results["Val Loss"], mode='lines', name='Mask2Former - Validation Loss'))
    fig.add_trace(go.Scatter(x=mask2former_results["Epoch"], y=mask2former_results["Val IoU"], mode='lines', name='Mask2Former - Validation IoU Score'))

    fig.update_layout(title="Évolution des métriques", xaxis_title="Epoch", yaxis_title="Score", template="plotly_white")
    st.plotly_chart(fig)

# 🛠️ Page Test des modèles
if page == "Test des modèles":
    st.title("Tester les modèles de segmentation")

    model_choice = st.selectbox("Choisissez un modèle :", ["FPN", "Mask2Former"])
    image_choices = [f"lindau_000000_0000{i}_leftImg8bit.png" for i in range(10, 30)]
    image_name = st.selectbox("Choisissez une image :", image_choices)

    image_url = f"https://storage.googleapis.com/p9-dashboard-storage/Dataset/images/{image_name}"
    image = Image.open(urllib.request.urlopen(image_url))

    processed_image, original_size = preprocess_image(image)

    model = fpn_model if model_choice == "FPN" else mask2former_model
    with torch.no_grad():
        pred_mask = model(torch.tensor(processed_image).permute(0, 3, 1, 2).float()).squeeze()
        pred_mask = torch.argmax(pred_mask, dim=0).cpu().numpy()

    processed_mask = resize_and_colorize_mask(pred_mask, original_size, CITYSCAPES_PALETTE)

    st.image(image, caption="Image Originale", use_container_width=True)
    st.image(processed_mask, caption=f"Masque Prédit - {model_choice}", use_container_width=True)
