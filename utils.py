import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import timm
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

# 🔹 Définition du modèle FPN + Resnet50
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

# 🔹 Définition du modèle FPN + Resnet50
class FPN_ConvNeXtV2_Segmenter(nn.Module):
    def __init__(self, num_classes=8):
        super(FPN_ConvNeXtV2_Segmenter, self).__init__()

        # Charger ConvNeXt V2-Large pré-entraîné
        self.convnext_backbone = timm.create_model("convnextv2_large", pretrained=True, features_only=True)

        # Convolutions latérales 1x1 pour aligner les features avec FPN
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(192, 256, kernel_size=1),   # Feature 0 (128x128)
            nn.Conv2d(384, 256, kernel_size=1),   # Feature 1 (64x64)
            nn.Conv2d(768, 256, kernel_size=1),   # Feature 2 (32x32)
            nn.Conv2d(1536, 256, kernel_size=1),  # Feature 3 (16x16)
        ])

        # Convolutions 3x3 après fusion des features
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        ])

        # Convolution finale pour segmentation
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        """Passage avant du modèle FPN + ConvNeXtV2"""

        # Extraire les features de ConvNeXt V2-Large
        features = self.convnext_backbone(x)

        # Appliquer les convolutions latérales 1x1
        fpn_features = [conv(feat) for feat, conv in zip(features, self.lateral_convs)]

        # Fusionner les features FPN (du plus bas niveau au plus haut)
        for i in range(len(fpn_features) - 1, 0, -1):
            fpn_features[i - 1] += F.interpolate(fpn_features[i], scale_factor=2, mode="bilinear", align_corners=False)

        # Appliquer les convolutions FPN après fusion
        fpn_features = [conv(feat) for feat, conv in zip(fpn_features, self.fpn_convs)]

        # Prendre la feature la plus large (128x128)
        output = fpn_features[0]

        # Convolution finale pour segmentation
        output = self.final_conv(output)

        # Upsample à la taille de l'image d'entrée (512x512)
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
