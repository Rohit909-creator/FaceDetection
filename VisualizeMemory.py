import cv2
import numpy as np
import pytorch_lightning as pl
from Model import MemoryBased
path = r"lightning_logs\version_1\checkpoints\epoch=71-step=2376.ckpt"

model = MemoryBased.load_from_checkpoint(path, width = 160, hieght = 160 ,output_size=2562)
model.eval()

memory = model.persistent_memory.x.detach().cpu().numpy()
memory = np.transpose(memory, (0, 2, 3, 1))  # Change from (batch_size, channels, height, width) to (batch_size, height, width, channels)
print("Memory shape: ", memory.shape)
img = cv2.resize(memory[0], (512, 512))  # Resize to 512x512 for better visualization
# only show one channel
img1 = img[:, :, 0]*255  # Take only the first channel for visualization
# cv2.imshow("Memory", memory[0])
# cv2.imshow("Memory", img1)
img2 = img[:, :, 1]*255  # Take only the second channel for visualization
# cv2.imshow("Memory2", img2)
img3 = img[:, :, 2]*255  # Take only the third channel for visualization
# cv2.imshow("Memory3", img3)
# cv2.waitKey(0)
cv2.imwrite("./NeuralMemory/Memory.png", img1)  # Save the image
cv2.imwrite("./NeuralMemory/Memory2.png", img2)  # Save the image
cv2.imwrite("./NeuralMemory/Memory3.png", img3)  # Save the image