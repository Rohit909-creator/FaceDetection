import torch
import cv2
import os
import pandas as pd
import numpy as np

df = pd.read_csv("Dataset.csv")
# print(df.head())
# print(df.values[:,1])
# labels = df.values[:, 1]
# label_to_index = {label: idx for idx, label in enumerate(set(labels))}
# encoded_labels = torch.tensor([label_to_index[label] for label in labels])
# print(encoded_labels)  # tensor([0, 1, 0])
# torch.save(encoded_labels, "Labels.pt")

images = []
filenames = df.values[:, 0]

for filename in filenames:
    img = cv2.imread(f"./Faces/{filename}")
    images.append(img)
    
images_np = np.stack(images)
images_tensor = torch.tensor(images, dtype=torch.float32)
images_tensor = images_tensor.transpose(1, 2)

print(images_tensor.shape)
torch.save(images_tensor, "features.pt")