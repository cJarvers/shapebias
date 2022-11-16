from matplotlib import pyplot as plt
import sys
sys.path.insert(0, "../src")
from datasets import SilhouetteDataset

data = SilhouetteDataset("../data", image_set="val", filters=["single", "occluded", "truncated"])

#img, seg, ann = data[4]
#plt.subplot(1,2,1)
#plt.imshow(img)
#plt.subplot(1,2,2)
#plt.imshow(seg)
#print(ann)
#plt.show()

class_counts = data._count_classes()

print("Total number of images:", len(data))
print(f"Class occurences ({len(class_counts)} classes):")
for c in class_counts:
    print(c.ljust(20, " "), class_counts[c])


