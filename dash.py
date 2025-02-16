import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import urllib.request
from PIL import Image

# Définition des URLs des modèles sur GCS
fpn_url = "https://storage.googleapis.com/p9-dashboard-storage/Models/fpn_best.pth"
mask2former_url = "https://storage.googleapis.com/p9-dashboard-storage/Models/mask2former_best.pth"

# Téléchargement et chargement des modèles depuis GCS
fpn_model_path = "fpn_best.pth"
mask2former_model_path = "mask2former_best.pth"

urllib.request.urlretrieve(fpn_url, fpn_model_path)
urllib.request.urlretrieve(mask2former_url, mask2former_model_path)

fpn_model = torch.load(fpn_model_path, map_location=torch.device("cpu"))
mask2former_model = torch.load(mask2former_model_path, map_location=torch.device("cpu"))

st.write("Modèles chargés avec succès !")

# Création de la sidebar
st.sidebar.title("Menu")
page = st.sidebar.radio("Aller à :", [
    "EDA", 
    "Résultats du modèle FPN", 
    "Résultats du modèle Mask2Former", 
    "Comparaison des modèles", 
    "Test des modèles"
])

# Affichage de la page EDA
if page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    
    # Affichage de la structure des dossiers
    st.header("Structure des Dossiers et Fichiers")
    folders = {"Images": ["train", "val", "test"], "Masques": ["train", "val", "test"]}
    for key, values in folders.items():
        st.write(f"**{key}**: {', '.join(values)}")
    
    # Nombre d'images et masques
    dataset_info = {
        "Ensemble": ["Train", "Validation", "Test"],
        "Images": [2975, 500, 1525],
        "Masques": [2975, 500, 1525]
    }
    df_info = pd.DataFrame(dataset_info)
    st.table(df_info)
    
    # Distribution des classes dans les masques
    st.header("Distribution des Classes dans les Masques")
    class_distribution = {
        "ID": [7, 11, 21, 26, 8, 1, 23, 3, 4, 2, 6, 17, 24, 22, 13, 9, 12, 20, 33, 15],
        "Classe": ["road", "fence", "truck", "void", "sidewalk", "ego vehicle", "train", "out of roi", "static", "rectification border",
                    "ground", "sky", "motorcycle", "bus", "traffic light", "building", "pole", "car", "void", "vegetation"],
        "Pixels": [2036416525, 1260636120, 879783988, 386328286, 336090793, 286002726, 221979646, 94111150, 83752079, 81359604,
                    75629728, 67789506, 67326424, 63949536, 48454166, 39065130, 36199498, 30448193, 22861233, 17860177]
    }
    df_classes = pd.DataFrame(class_distribution)
    
    # Affichage du tableau
    st.table(df_classes.head(10))
    
    # Affichage du graphique
    fig, ax = plt.subplots()
    ax.bar(df_classes["Classe"], df_classes["Pixels"], color="skyblue")
    plt.xticks(rotation=90)
    plt.xlabel("Classes")
    plt.ylabel("Nombre de Pixels")
    plt.title("Répartition des Pixels par Classe")
    st.pyplot(fig)
    
    # Dimensions des images et masques
    st.header("Dimensions des Images et Masques")
    st.write("**2048x1024 : 5000 occurrences**")
    
    # Analyse des formats de fichiers
    st.header("Analyse des Formats de Fichiers")
    file_formats = {"Format": [".png (images)", ".png (masques)", ".json (masques)"], "Nombre": [5000, 15000, 5000]}
    df_formats = pd.DataFrame(file_formats)
    st.table(df_formats)
    
    # Affichage d'exemples d'images
    st.header("Exemples d'Images et Masques")

    # Définition des URLs des images et masques sur GCS
    sample_image_url = "https://storage.googleapis.com/p9-dashboard-storage/Dataset/images/lindau_000000_000019_leftImg8bit.png"
    sample_mask_url = "https://storage.googleapis.com/p9-dashboard-storage/Dataset/masks/lindau_000000_000019_gtFine_color.png"

    col1, col2 = st.columns(2)
    with col1:
        st.image(sample_image_url, caption="Image Originale", use_column_width=True)
    with col2:
        st.image(sample_mask_url, caption="Masque Correspondant", use_column_width=True)
