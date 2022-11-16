from matplotlib import pyplot as plt
import sys
sys.path.insert(0, "../src")
from datasets import SilhouetteDataset
from mappings import int2label, get_image, get_silhouette_simple

images = SilhouetteDataset("../data", image_set="val", filters=["single", "occluded", "truncated"], mapping=get_image)
silhouettes = SilhouetteDataset("../data", image_set="val", filters=["single", "occluded", "truncated"], mapping=get_silhouette_simple)

class_counts = images._count_classes()

print("Total number of images:", len(images))
print(f"Class occurences ({len(class_counts)} classes):")
for c in class_counts:
    print(c.ljust(20, " "), class_counts[c])

n = 6
for i in range(n):
    image, target = images[i]
    silhouette, _ = silhouettes[i]
    plt.subplot(2, n, i+1)
    plt.imshow(image)
    plt.title(f"{int2label[target]}")
    plt.subplot(2, n, n+i+1)
    plt.imshow(silhouette, cmap="gray")
plt.show()

