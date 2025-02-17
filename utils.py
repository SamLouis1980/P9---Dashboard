import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# 🔹 Définition de la palette de couleurs Cityscapes
CLASS_COLORS = {
    0: (0, 0, 0),        # Void
    1: (128, 64, 128),   # Flat
    2: (70, 70, 70),     # Construction
    3: (153, 153, 153),  # Object
    4: (107, 142, 35),   # Nature
    5: (70, 130, 180),   # Sky
    6: (220, 20, 60),    # Human
    7: (0, 0, 142)       # Vehicle
}

# 🔹 Définition du modèle FPN
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

# 🔹 Prétraitement des images
def preprocess_image(image, input_size):
    """Prépare une image pour le modèle."""
    image = image.convert("RGB")  # S'assurer que l'image est en mode RGB
    original_size = image.size  # Sauvegarde de la taille originale
    image = image.resize(input_size, Image.BILINEAR)  # Redimensionnement
    image_array = np.array(image) / 255.0  # Normalisation entre 0 et 1
    return np.expand_dims(image_array, axis=0), original_size

# 🔹 Post-traitement : Colorisation et redimensionnement du masque
def resize_and_colorize_mask(mask, original_size, palette=CLASS_COLORS):
    """Redimensionne et colorise un masque segmenté."""
    mask = Image.fromarray(mask.astype(np.uint8))  # Convertir en image PIL
    mask = mask.resize(original_size, Image.NEAREST)  # Redimensionnement
    
    # Appliquer la palette
    flat_palette = [value for color in palette.values() for value in color]
    mask.putpalette(flat_palette)
    
    return mask.convert("RGB")  # Convertir en RGB pour affichage
