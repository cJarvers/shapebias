# Defines dataset wrappers for images, silhouettes and related stimuli
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Compose, ConvertImageDtype
from torchvision.transforms import InterpolationMode
import mappings

def get_common_images(segmentation, detection):
    """
    Find the common images in the datasets `segmentation`(a VOCSegmentation object)
    and `detection`(a VOCDetection object).
    """
    s = 0 # index into segmentation dataset
    d = 0 # index into detection dataset
    sinds = []
    dinds = []
    while s < len(segmentation) and d < len(detection):
        sname = segmentation.images[s] # get names of input images for segmentation ...
        dname = detection.images[d]    # ... and detection for current indices
        syear, snum = sname.split("/")[-1].split(".")[0].split("_") # parse names into year and image index
        dyear, dnum = dname.split("/")[-1].split(".")[0].split("_")
        syear, snum = int(syear), int(snum)
        dyear, dnum = int(dyear), int(dnum)
        if syear < dyear: # if the segmentation year is to low, go to next s
            s += 1
        elif dyear < syear: # if the detection year is to low, go to next d
            d += 1
        elif snum < dnum: # if years are the same, but segmentation image number is too low, go to next s
            s += 1
        elif dnum < snum: # if years are the same, but detection image number is too low, go to next d
            d += 1
        else: # if years and image numbers are the same, save the indices and move on
            sinds.append(s)
            dinds.append(d)
            s += 1
            d += 1
    return sinds, dinds


class SilhouetteDataset(Dataset):
    """
    Dataset of images and silhouettes from PascalVOC. Silhouettes are derived
    from semantic segmentation data.

    Args:
        root: folder for VOC data / devkit
        year, image_set, download: passed to torchvision `VOCSegmentation` and `VOCDetection` constructors
        filters: list of str that controls how images are filtered Default: ["single"].
            Allowed values are:
            "single" - only images with single objects are used
            "occluded" - images of occluded objects are removed
            "truncated" - images of truncated objects are removed
        mapping: function that maps image, segmentation masks, and annotation onto desired output.
        transform: torchvision transform to be applied to image / silhouette (after `mapping`)
    """
    def __init__(self, root="../data", year="2012", image_set="train", download=False,
            sinds=None, dinds=None,
            filters=["single", "occluded", "truncated"], mapping=mappings.get_silhouette_simple,
            transform=Compose([ToTensor(),
                Resize((224, 224), interpolation=InterpolationMode.NEAREST), # nearest neighbor interpolation conserves narrow structures better
                ConvertImageDtype(torch.float32)
            ])):
        super().__init__()
        self.mapping = mapping
        self.transform = transform
        self.segmentation = datasets.VOCSegmentation(root=root, year=year, image_set=image_set, download=download)
        self.detection = datasets.VOCDetection(root=root, year=year, image_set=image_set, download=download)
        if (sinds is None) or (dinds is None):
            self.sinds, self.dinds = get_common_images(self.segmentation, self.detection)
            self.sinds, self.dinds = self._filter_inds(filters)
        else:
            self.sinds = sinds
            self.dinds = dinds

    def __getitem__(self, index):
        img, seg, ann = self._get_raw(index)
        x, y = self.mapping(img, seg, ann)
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.sinds)

    def _get_raw(self, index):
        img = self.segmentation[self.sinds[index]][0]
        seg = self.segmentation[self.sinds[index]][1]
        annotation = self.detection[self.dinds[index]][1]
        return np.array(img), np.array(seg), annotation

    def _filter_inds(self, filters):
        new_sinds = []
        new_dinds = []
        for i in range(len(self.sinds)):
            _, _, ann = self._get_raw(i)
            if "single" in filters and not len(ann['annotation']['object']) == 1:
                continue # filter out images with more than one object
            if "occluded" in filters:
                not_occluded = [o['occluded'] == '0' for o in ann['annotation']['object']]
                if not all(not_occluded):
                    continue # filter out images where an object is occluded
            if "truncated" in filters:
                not_truncated = [o['truncated'] == '0' for o in ann['annotation']['object']]
                if not all(not_truncated):
                    continue # filer out images where an object is truncated
            # otherwise, keep the image
            new_sinds.append(self.sinds[i])
            new_dinds.append(self.dinds[i])
        return new_sinds, new_dinds

    def _count_classes(self):
        class_counts = {}
        for i in range(len(self.sinds)):
            _, _, ann = self._get_raw(i)
            for o in ann['annotation']['object']:
                name = o['name']
                if name in class_counts:
                    class_counts[name] += 1
                else:
                    class_counts[name] = 1
        return class_counts


############################################
# Convenience function for loading dataset #
############################################
def loaddataset(imgtype, crop, **kwargs):
    """
    Loads a silhouette datasets.
    
    # Args:
    - imgtype: selects which images are returned (e.g., normal, masked, silhouettes, ...)
        Possible values are:
    - All other arguments are passed on to SilhouetteDataset constructor
    """
    if crop:
        if imgtype == "image":
            mapping = mappings.get_image_bbox
        elif imgtype == "fg":
            mapping = mappings.get_image_fg_bbox
        elif imgtype == "bg":
            mapping = mappings.get_image_bg_bbox
        elif imgtype == "silhouette":
            mapping = mappings.get_silhouette_bbox
        elif imgtype == "silhouette_frankenstein" or imgtype=="frankenstein":
            mapping = mappings.get_silhouette_bbox_frankenstein
        elif imgtype == "silhouette_serrated" or imgtype=="serrated":
            mapping = mappings.get_silhouette_bbox_serrated
        else:
            raise(ValueError(f"Unknown dataset / image type {imgtype}."))
    else:
        if imgtype is None or imgtype == "image":
            mapping = mappings.get_image
        elif imgtype == "fg":
            mapping = mappings.get_image_fg
        elif imgtype == "bg":
            mapping = mappings.get_image_bg
        elif imgtype == "silhouette" or imgtype == "silhouette_simple":
            mapping = mappings.get_silhouette_simple
        else:
            raise(ValueError(f"Unknown dataset / image type {imgtype}."))
    data = SilhouetteDataset(mapping=mapping, **kwargs)
    return data, data.sinds, data.dinds