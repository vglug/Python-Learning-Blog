import cv2
import torch
import torchvision.transforms as transforms
#load image
image = cv2.imread('image.jpg')
# Preprocess image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
image = transform(image)
