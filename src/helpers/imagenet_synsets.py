from nltk.corpus import wordnet as wn
import os
current_file = os.path.dirname(__file__)

# PascalVOC categories:
with open(current_file + "/pascalVOC_labels.txt", "r") as file:
    voclabels = {
        label: index
        for (index, label) in map(lambda line: line.strip("\n").split(" "), file)
    }
voc_synsets = {}
# For most labels, the first noun synset is correct, BUT:
# - "diningtable" has to be replaced by "dining_table" 
# - "pottedplant" has to be replaced by "pot_plant" or "flowerpot" (latter is used in ImageNet)
# - "tvmonitor" does not exist. Closest equivalents are:
#     - "monitor" (4th synset)
#     - "monitor" (5th synset) -> used in ImageNet
#     - "television" (2nd synset) -> used in ImageNet
#   The lowest common hypernym of these is "instrumentality.n.03"
# - we replace "cow" with "bovid" to be more generic (e.g., to catch "buffalo")
# - 
for label in voclabels.keys():
    if label == "diningtable":
        synset = wn.synsets("dining_table", pos="n")[0]
    elif label == "pottedplant":
        synset = wn.synsets("flowerpot", pos="n")[0]
    elif label == "tvmonitor":
        synset = wn.synsets("television", pos="n")[1]
    elif label == "cow":
        synset = wn.synsets("bovid", pos="n")[0]
    elif label == "aeroplane":
        synset = wn.synsets("heavier-than-air_craft", pos="n")[0]
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
for i, synset in enumerate(synsets):
    for vocclass, vocsynset in voc_synsets.items():
        #for hyper in syn.closure(lambda s: s.hypernyms()):
        #    if hyper in vocsynset:
        if vocsynset in synset.lowest_common_hypernyms(vocsynset):
            voc2imgnet[vocclass].append(i)
            imgnet2voc[i] = vocclass