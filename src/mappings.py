# Defines mapping / extraction functions to be used with SilhouetteDataset class.
import numpy as np

##############################
# Helper functions / objects #
##############################
label2int = { # labels as used in the VOC documentation / segment encoding
    "aeroplane": 1, # start with 1 (0 encodes background)
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}

int2label = {v: k for k, v in label2int.items()}


########################
# Extraction functions #
########################
# These can be plugged into a SilhouetteDataset to return different types of
# data (original image, silhouette, cropped silhouette, ...)
def get_image(img, seg, ann):
    "Extract image and label (like for classification)."
    label = ann['annotation']['object'][0]['name']
    target = label2int[label]
    return img, target

def get_silhouette_simple(img, seg, ann):
    "Extract silhouette (same dimension as image) and label."
    label = ann['annotation']['object'][0]['name']
    target = label2int[label]
    silhouette = np.expand_dims((seg != target), 2).repeat(3, axis=2)
    return silhouette, target


