# Defines mapping / extraction functions to be used with SilhouetteDataset class.
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, center_of_mass

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

def getmask(seg, target, dtype=np.uint8):
    "Return boolean mask of object pixels"
    return np.expand_dims((seg == target).astype(dtype), 2).repeat(3, axis=2)

def swappatches(img, splits_per_dim):
    "Split img into (roughly) equal patches and swap them randomly."
    pass

def frankenstein(img):
    "Flip lower half of image along vertical axis. Then re-align to reduce artifacts."
    # find center of mass along y dimension; this is where we split the image and flip
    ymid, _, _ = center_of_mass(img) # this only works with silhouettes, not original images
    ymid = round(ymid)
    # find the borders of the object in the last row above the swap, so that they can be
    # re-aligned later
    object_indices = np.argwhere(img[ymid-1, :, 0])
    first = object_indices.min()
    last = object_indices.max()
    shift = first - (img.shape[1] - last)
    # flip lower half and re-align
    img[ymid:, :, :] = img[ymid:, ::-1, :] # flip
    if shift > 0: # shift right
        img[ymid:, shift:, :] = img[ymid:, :-shift, :]
        img[ymid:, :shift, :] = 0 # fill undefined area with 0
    else: # shift left
        img[ymid:, :shift, :] = img[ymid:, -shift:, :]
        img[ymid:, shift:, :] = 0 # fill undefined area with 0
    return img

def double_flip(img):
    "Flip lower half of image along vertical axis. Then flip right half of image along horizontal axis."
    ysize, xsize = img.shape[0], img.shape[1]
    yhalf, xhalf = ysize // 2, xsize // 2
    img[yhalf:, :, :] = img[yhalf:, ::-1, :]
    img[:, xhalf:, :] = img[::-1, xhalf:, :]
    return img


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

def get_image_bbox(img, seg, ann, factor=1.2):
    "Extract image restricted to bounding box of object, enlarged by `factor`."
    obj = getobject(ann)
    target = gettarget(obj)
    # get bounding box
    xmin, xmax, ymin, ymax = getbbox(obj, ann['annotation']['size'], factor)
    image = img[ymin:ymax, xmin:xmax, :]
    return image, target

def get_image_bg(img, seg, ann):
    "Extract image, but mask out the object."
    obj = getobject(ann)
    target = gettarget(obj)
    mask = getmask(seg, target)
    background = img * (1 - mask)
    return background, target

def get_image_bg_bbox(img, seg, ann, factor=1.2):
    "Extract image background, restricted to boundig box."
    obj = getobject(ann)
    target = gettarget(obj)
    xmin, xmax, ymin, ymax = getbbox(obj, ann['annotation']['size'], factor)
    image = img[ymin:ymax, xmin:xmax, :]
    mask = getmask(seg[ymin:ymax, xmin:xmax], target)
    background = image * (1 - mask)
    return background, target

def get_image_fg(img, seg, ann):
    "Extract image, but mask out the background."
    obj = getobject(ann)
    target = gettarget(obj)
    mask = getmask(seg, target)
    foreground = img * mask
    white_bg = (1 - mask) * 255
    return foreground + white_bg, target

def get_image_fg_bbox(img, seg, ann, factor=1.2):
    "Extract image, mask out background, and restrict to bounding box."
    obj = getobject(ann)
    target = gettarget(obj)
    xmin, xmax, ymin, ymax = getbbox(obj, ann['annotation']['size'], factor)
    image = img[ymin:ymax, xmin:xmax, :]
    mask = getmask(seg[ymin:ymax, xmin:xmax], target)
    foreground = image * mask
    white_bg = (1 - mask) * 255
    return foreground + white_bg, target

def get_silhouette_simple(img, seg, ann):
    "Extract silhouette (same dimension as image) and label."
    obj = getobject(ann)
    target = gettarget(obj)
    mask = getmask(seg, target)
    silhouette = (1 - mask) * 255
    return silhouette, target

def get_silhouette_bbox(img, seg, ann, factor=1.2):
    "Extract silhouette restricted to bounding box of object, enlarged by `factor`."
    obj = getobject(ann)
    target = gettarget(obj)
    # get bounding box
    xmin, xmax, ymin, ymax = getbbox(obj, ann['annotation']['size'], factor)
    mask = getmask(seg[ymin:ymax, xmin:xmax], target)
    silhouette = (1 - mask) * 255
    return silhouette, target

def get_silhouette_bbox_patchy(img, seg, ann, factor=1.2, splits_per_dim=2):
    "Extract silhouette restricted to bounding box, then scramble image patches."
    obj = getobject(ann)
    target = gettarget(obj)
    xmin, xmax, ymin, ymax = getbbox(obj, ann['annotation']['size'], factor)
    mask = getmask(seg[ymin:ymax, xmin:xmax], target)
    silhouette = (1 - mask) * 255
    silhouette = swappatches(img, splits_per_dim) # NOT YET IMPLEMENTED
    return silhouette, target

def get_silhouette_bbox_frankenstein(img, seg, ann, factor=1.2, splits_per_dim=2):
    "Extract silhouette restricted to bounding box, then flip image halves."
    obj = getobject(ann)
    target = gettarget(obj)
    xmin, xmax, ymin, ymax = getbbox(obj, ann['annotation']['size'], factor)
    mask = getmask(seg[ymin:ymax, xmin:xmax], target)
    silhouette = frankenstein(mask)
    silhouette = (1 - silhouette) * 255
    return silhouette, target

def get_silhouette_bbox_serrated(img, seg, ann, factor=1.2, borderwidth=5, sigma=2.0):
    "Extract silhouette restricted to bounding box, then corrupt border outline by noise."
    obj = getobject(ann)
    target = gettarget(obj)
    xmin, xmax, ymin, ymax = getbbox(obj, ann['annotation']['size'], factor)
    mask = getmask(seg[ymin:ymax, xmin:xmax], target)
    border = getmask(seg[ymin:ymax, xmin:xmax], 255, dtype=np.bool) # object outlines are encoded as 255
    border = binary_dilation(border, iterations=borderwidth // 2) # extend object outline by dilation
    noise = np.random.randn(ymax-ymin, xmax - xmin) # generate random noise to corrupt border
    noise = gaussian_filter(noise, sigma=sigma) > 0.0 # blur noise to have larger coherent bits; convert to binary
    noise_border = border * np.expand_dims(noise, 2).repeat(3, axis=2)
    result = (1 - mask) * (1 - border) + noise_border
    return result.astype(np.uint8) * 255, target





def preprocess(img, seg, ann, bbox=True, factor=1.2, mask_bg=False,
        mask_fg=False):
    """Preprocess images."""
    pass