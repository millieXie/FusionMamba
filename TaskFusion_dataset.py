# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os
from numpy import asarray

def imresize(arr, size, interp='bilinear', mode=None):
    # print('-----------',type(arr))
    numpydata = asarray(arr)
    # print('-----------', type(numpydata))
    im = Image.fromarray(numpydata, mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size)*percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size)*size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return np.array(imnew)


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None,length = 0):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.filepath_ir = []
        self.filenames_ir = []
        self.filepath_vis = []
        self.filenames_vis = []
        self.length = length
        if split == 'train':
            data_dir_vis = "vis_path"
            data_dir_ir = "ir_path"
            dir = os.listdir(data_dir_ir)
            dir.sort()
            for dir0 in dir:
                for dir1 in os.listdir(os.path.join(data_dir_ir, dir0)):
                    req_path = os.path.join(data_dir_ir, dir0, dir1, 'ir')
                    for file in os.listdir(req_path):
                        filepath_ir_ = os.path.join(req_path, file)
                        self.filepath_ir.append(filepath_ir_)
                        self.filenames_ir.append(file)
                        filepath_vis_ = filepath_ir_.replace('ir','vis')
                        self.filepath_vis.append(filepath_vis_)
                        self.filenames_vis.append(file)
                        self.split = split

        elif split == 'test':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split


    def __getitem__(self, index):
        if self.split=='train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]

            image_vis = cv2.imread(vis_path)  #For grayscale images, they can be directly used as input. However, when inputting a medical image, it must first be converted from RGB to YCbCr format, with the Y channel being used as the primary input.
            image_inf = cv2.imread(ir_path, 0)
            image_inf, image_vis = self.resize(image_inf, image_vis, [256, 256], [256, 256])

            image_vis = np.asarray(Image.fromarray(image_vis), dtype=np.float32) / 255.0
            image_vis = np.expand_dims(image_vis, axis=0)

            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)

            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),

                name,
            )
        elif self.split=='test':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis = cv2.imread(vis_path)

            image_inf = cv2.imread(ir_path, 0)

            image_vis = np.asarray(Image.fromarray(image_vis), dtype=np.float32) / 255.0
            image_vis = np.expand_dims(image_vis, axis=0)

            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
            )

    def __len__(self):
        return self.length

    def resize(self, data, label, crop_size_img, crop_size_label):
        data = imresize(data, crop_size_img, interp='bicubic')
        label = imresize(label, crop_size_label, interp='bicubic')
        return data, label

