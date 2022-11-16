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

def getobject(ann, index=0):
    return ann['annotation']['object'][index]

def gettarget(obj):
    return label2int[obj['name']]

def getbbox(obj, size, factor=1.0):
    # get corners from annotation
    xmin = int(obj['bndbox']['xmin'])
    xmax = int(obj['bndbox']['xmax'])
    ymin = int(obj['bndbox']['ymin'])
    ymax = int(obj['bndbox']['ymax'])
    # scale by factor
    xrange = xmax - xmin
    yrange = ymax - ymin
    xmin, xmax = round(xmax - factor * xrange), round(xmin + factor * xrange)
    ymin, ymax = round(ymax - factor * yrange), round(ymin + factor * yrange)
    # make sure new bounding box is in image range
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, int(size['width']))
    ymax = min(ymax, int(size['height']))
    return xmin, xmax, ymin, ymax


########################
# Extraction functions #
########################
# These can be plugged into a SilhouetteDataset to return different types of
# data (original image, silhouette, cropped silhouette, ...)
def get_image(img, seg, ann):
    "Extract image and label (like for classification)."
    obj = getobject(ann)
    target = gettarget(obj)
    return img, target

def get_silhouette_simple(img, seg, ann):
    "Extract silhouette (same dimension as image) and label."
    obj = getobject(ann)
    target = gettarget(obj)
    silhouette = np.expand_dims((seg != target), 2).repeat(3, axis=2)
    return silhouette, target

def get_image_bbox(img, seg, ann, factor=1.2):
    "Extract image restricted to bounding box of object, enlarged by `factor`."
    obj = getobject(ann)
    target = gettarget(obj)
    # get bounding box
    xmin, xmax, ymin, ymax = getbbox(obj, ann['annotation']['size'], factor)
    image = img[ymin:ymax, xmin:xmax, :]
    return image, target


def get_silhouette_bbox(img, seg, ann, factor=1.2):
    "Extract silhouette restricted to bounding box of object, enlarged by `factor`."
    obj = getobject(ann)
    target = gettarget(obj)
    # get bounding box
    xmin, xmax, ymin, ymax = getbbox(obj, ann['annotation']['size'], factor)
    silhouette = (seg != target)[ymin:ymax, xmin:xmax]
    silhouette = np.expand_dims(silhouette, 2).repeat(3, axis=2)
    return silhouette, target
