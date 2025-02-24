import streamlit as st
import os
import torch
import urllib.request
from google.cloud import storage
from PIL import Image
from utils import preprocess_image, resize_and_colorize_mask, CLASS_COLORS

# 🔹 Définition du bucket GCS
BUCKET_NAME = "p9-dashboard-storage"
MODEL_FOLDER = "Models"
IMAGE_FOLDER = "Dataset/images"
MASK_FOLDER = "Dataset/masks"

# 🔹 Initialisation du client GCS (anonyme car bucket public)
storage_client = storage.Client.create_anonymous_client()
bucket = storage_client.bucket(BUCKET_NAME)

# 🔹 Mise en cache des modèles pour éviter les rechargements inutiles
@st.cache_resource
def load_models():
    fpn_model_path = "fpn_best.pth"
    mask2former_model_path = "mask2former_best.pth"

    fpn_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{MODEL_FOLDER}/fpn_best.pth"
    mask2former_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{MODEL_FOLDER}/mask2former_best.pth"

    if not os.path.exists(fpn_model_path):
        urllib.request.urlretrieve(fpn_url, fpn_model_path)

    if not os.path.exists(mask2former_model_path):
        urllib.request.urlretrieve(mask2former_url, mask2former_model_path)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ Chargement du modèle ENTIER (et pas juste les poids)
        fpn_model = torch.load(fpn_model_path, map_location=device)
        fpn_model.eval()

        mask2former_model = torch.load(mask2former_model_path, map_location=device)
        mask2former_model.eval()

        st.write("✅ Modèles chargés avec succès.")
        return fpn_model, mask2former_model

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des modèles : {e}")
        return None, None

fpn_model, mask2former_model = load_models()

# 🔹 Vérification du bon chargement des modèles
if fpn_model is None or mask2former_model is None:
    st.error("❌ Impossible de charger les modèles. Vérifiez votre stockage GCS.")
    st.stop()

# 🔹 Fonction pour récupérer la liste des images disponibles dans le bucket
@st.cache_data
def get_available_images():
    try:
        blobs = bucket.list_blobs(prefix=IMAGE_FOLDER + "/")
        image_files = [blob.name.split("/")[-1] for blob in blobs if blob.name.endswith(".png")]

        if not image_files:
            st.error("❌ Aucune image trouvée dans le bucket. Vérifiez le stockage GCS.")
            return []

        return image_files
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des images : {e}")
        return []

available_images = get_available_images()

# 🔹 Vérification si la liste d'images est vide
if not available_images:
    st.error("⚠ Aucune image disponible. Arrêt du script.")
    st.stop()

st.write(f"✅ {len(available_images)} images disponibles.")

# 🔹 Sidebar Navigation
st.sidebar.title("Menu")
page = st.sidebar.radio("Aller à :", ["EDA", "Résultats des modèles", "Test des modèles"])

# 🔹 Page Test des modèles
if page == "Test des modèles":
    st.title("Test de Segmentation avec les Modèles")

    image_choice = st.selectbox("Choisissez une image à segmenter", available_images)
    model_choice = st.radio("Choisissez le modèle", ["FPN", "Mask2Former"])

    # 🔹 Vérification de la sélection d'image
    if not image_choice:
        st.error("⚠ Aucune image sélectionnée.")
        st.stop()

    # 🔹 URL directe des fichiers GCS (aucun téléchargement local nécessaire)
    image_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{IMAGE_FOLDER}/{image_choice}"
    mask_filename = image_choice.replace("leftImg8bit", "gtFine_color")
    mask_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{MASK_FOLDER}/{mask_filename}"

    try:
        image = Image.open(urllib.request.urlopen(image_url)).convert("RGB")
        st.image(image, caption="Image d'entrée", use_container_width=True)
    except Exception as e:
        st.error(f"⚠ Erreur lors du chargement de l'image : {e}")

    try:
        input_size = (512, 512)
        image_resized, original_size = preprocess_image(image, input_size)
        tensor_image = torch.tensor(image_resized).permute(0, 3, 1, 2).float().unsqueeze(0)

        output = fpn_model(tensor_image) if model_choice == "FPN" else mask2former_model(tensor_image)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        mask_colorized = resize_and_colorize_mask(mask, original_size, CLASS_COLORS)

        st.image(mask_colorized, caption="Masque segmenté", use_container_width=True)
        st.image(Image.open(urllib.request.urlopen(mask_url)).convert("RGB"), caption="Masque réel", use_container_width=True)

    except Exception as e:
        st.error(f"⚠ Erreur lors de la segmentation : {e}")
