import cv2
import numpy as np
import torch

class TransposeAndOtsuThreshold:
    def __call__(self, image):
        image = np.transpose(image)
        image = image.astype(np.uint8)
        _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresholded = torch.from_numpy(thresholded).float()
        return thresholded 

