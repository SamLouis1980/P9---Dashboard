import numpy as np
from PIL import Image

def preprocess_image(image, input_size):
    """Prépare une image pour le modèle.

    Args:
        image (PIL.Image): L'image à prétraiter.
        input_size (tuple): La taille d'entrée attendue par le modèle (largeur, hauteur).

    Returns:
        tuple: Tableau numpy normalisé prêt pour le modèle, et taille originale de l'image.
    """
    image = image.convert("RGB")  # S'assurer que l'image est en mode RGB
    original_size = image.size  # Sauvegarde de la taille originale
    image = image.resize(input_size, Image.BILINEAR)  # Redimensionnement
    image_array = np.array(image) / 255.0  # Normalisation entre 0 et 1
    return np.expand_dims(image_array, axis=0), original_size

def resize_and_colorize_mask(mask, original_size, palette):
    """Redimensionne et colorise un masque prédictif.

    Args:
        mask (numpy.ndarray): Le masque de classes (2D, valeurs entières).
        original_size (tuple): La taille originale de l'image d'entrée (largeur, hauteur).
        palette (dict): Un dictionnaire où les clés sont les indices de classes et les valeurs sont des tuples RGB.

    Returns:
        PIL.Image: Le masque colorisé redimensionné.
    """
    mask = Image.fromarray(mask.astype(np.uint8))  # Convertir le masque en image PIL
    mask = mask.resize(original_size, Image.NEAREST)  # Redimensionner à la taille originale

    # Appliquer la palette
    flat_palette = [value for color in palette.values() for value in color]
    mask.putpalette(flat_palette)  # Ajouter la palette

    return mask.convert("RGB")  # Convertir en RGB pour l'affichage
