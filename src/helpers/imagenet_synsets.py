from nltk.corpus import wordnet as wn
import numpy as np
import os
current_file = os.path.dirname(__file__)

# PascalVOC categories:
with open(current_file + "/pascalVOC_labels.txt", "r") as file:
    voclabels = {
        label: index
        for (index, label) in map(lambda line: line.strip("\n").split(" "), file)
    }
voc_synsets = {}
for label in voclabels.keys():
    # For most labels, the first noun synset is correct, BUT:
    if label == "diningtable":
        # "diningtable" has to be replaced by "dining_table" 
        synset = wn.synsets("dining_table", pos="n")[0]
    elif label == "pottedplant":
        # "pottedplant" has to be replaced by "pot_plant"
        # or by "flowerpot" (latter is used in ImageNet)
        synset = wn.synsets("flowerpot", pos="n")[0]
    elif label == "tvmonitor":
        # "tvmonitor" does not exist. Closest equivalents are:
        # - "monitor" (4th synset)
        # - "monitor" (5th synset) -> used in ImageNet
        # - "television" (2nd synset) -> used in ImageNet
        # The lowest common hypernym of these is "instrumentality.n.03"
        synset = wn.synsets("television", pos="n")[1]
    elif label == "cow":
        # we replace "cow" with "bovid" to be more generic (e.g., to catch "buffalo")
        synset = wn.synsets("bovid", pos="n")[0]
    elif label == "aeroplane":
        # several subtypes of plane (e.g., "warplane") are not captured
        # by synset "plane", but by its hypernym
        synset = wn.synsets("heavier-than-air_craft", pos="n")[0]
    elif label == "train":
        # ImageNet classes contain trains and locomotives;
        # They don't have a good common hypernym.
        # But "locomotive" seems to describe images in PascalVOC data better
        synset = wn.synsets("locomotive", pos="n")[0]
    else:
        synset = wn.synsets(label, pos="n")[0]
    voc_synsets[label] = synset

# ImageNet categories from file
with open(current_file + "/imagenet_categories.txt", "r") as file:
    categories = [line.split(" ")[0] for line in file]
# get synsets from 
synsets = [wn.synset_from_pos_and_offset('n', int(c[1:])) for c in categories]


voc2imgnet = {c: [] for c in voclabels.keys()}
imgnet2voc = {}
imagenet2voc = np.zeros(1000, dtype=np.int64)
for i, synset in enumerate(synsets):
    for vocclass, vocsynset in voc_synsets.items():
        #for hyper in syn.closure(lambda s: s.hypernyms()):
        #    if hyper in vocsynset:
        if vocsynset in synset.lowest_common_hypernyms(vocsynset):
            voc2imgnet[vocclass].append(i)
            imgnet2voc[i] = vocclass
            imagenet2voc[i] = voclabels[vocclass]