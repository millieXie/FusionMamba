import cv2
import numpy as np
import os
import torch
import time
import torchvision.transforms as transforms
from models.vmamba_Fusion_efficross import VSSM_Fusion as net
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = net(in_channel=1)
model_path = "./madelpath.pth"
use_gpu = torch.cuda.is_available()

if use_gpu:
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
else:
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)


# def imresize(image, size, interp='bilinear'):
#     interp_methods = {
#         'nearest': cv2.INTER_NEAREST,
#         'bilinear': cv2.INTER_LINEAR,
#         'bicubic': cv2.INTER_CUBIC,
#         'lanczos': cv2.INTER_LANCZOS4
#     }
#     resized_image = cv2.resize(image, (size[1], size[0]), interpolation=interp_methods.get(interp, cv2.INTER_LINEAR))
#     return resized_image

def fusion(input_folder_ir, input_folder_vis, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tic = time.time()


    ir_images = sorted([f for f in os.listdir(input_folder_ir) if f.endswith('.png')])
    vis_images = sorted([f for f in os.listdir(input_folder_vis) if f.endswith('.png')])

    for ir_image, vis_image in zip(ir_images, vis_images):
        path1 = os.path.join(input_folder_ir, ir_image)
        path2 = os.path.join(input_folder_vis, vis_image)

        # 读取灰度图像
        img1 = cv2.imread(path1, 0)
        img2 = cv2.imread(path2, 0)

        # 归一化图像
        img1 = np.asarray(Image.fromarray(img1), dtype=np.float32) / 255.0
        img2 = np.asarray(Image.fromarray(img2), dtype=np.float32) / 255.0

        # 扩展维度
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)

        # 转换为张量
        img1_tensor = torch.from_numpy(img1).unsqueeze(0).to(device)
        img2_tensor = torch.from_numpy(img2).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            out = model(img1_tensor, img2_tensor)
            out = torch.clamp(out, 0, 1) 
            
            # 将张量转换为 NumPy 数组
            out_np = out.cpu().numpy()
            
            # 归一化输出
            out_np = (out_np - np.min(out_np)) / (np.max(out_np) - np.min(out_np))

        # 处理和保存结果
        d = np.squeeze(out_np)
        result = (d * 255).astype(np.uint8)

        output_path = os.path.join(output_folder, ir_image)
        cv2.imwrite(output_path, result)

    toc = time.time()
    print('Processing time: {}'.format(toc - tic))




if __name__ == '__main__':
  
    input_folder_1 = '/path1'
    input_folder_2 = '/path2'
    output_folder = './results'

    fusion(input_folder_2, input_folder_1, output_folder)


