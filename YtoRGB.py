

import os
import cv2
import numpy as np

path_img1 = "PET"
path_img2 = "fusion_results"
save_path = "final_output"
all_images_1 = os.listdir(path_img1)
all_images_2 = os.listdir(path_img2)
for image in all_images_1:
    image_path_1 = os.path.join(path_img1, image)
    image_path_2 = os.path.join(path_img2, image)
    img_1 = cv2.imread(image_path_1)
    img_2 = cv2.imread(image_path_2)
    img_yuv = cv2.cvtColor(img_1, cv2.COLOR_BGR2YUV)
    img_yuv2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)
    y2, u2, v2 = cv2.split(img_yuv2)
    y2 = np.expand_dims(y2,2)
    u = np.expand_dims(u,2)
    v = np.expand_dims(v,2)

    out = np.concatenate((y2,u, v),axis=2)
    out = cv2.cvtColor(out, cv2.COLOR_YUV2BGR)
    cv2.imwrite(os.path.join(save_path,image), out)
