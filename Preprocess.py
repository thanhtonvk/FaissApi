import cv2
import numpy as np
def norm_mean_std(img):
    
    img = img / 255
    img = img.astype('float32')
    
    mean = np.mean(img, axis=(0, 1, 2))  # Per channel mean
    std = np.std(img, axis=(0, 1, 2))
    img = (img - mean) / std
    
    return img
def img_preprocess(img_fp, img_size=(384, 384), expand=True):
#     img = cv2.imread(img_fp)[:, :, ::-1]
    img = cv2.resize(img_fp, img_size)

    # normalize image
    img = norm_mean_std(img)

    if expand:
        img = np.expand_dims(img, axis=0)
    
    return img