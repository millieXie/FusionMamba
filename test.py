import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from TaskFusion_dataset import Fusion_dataset
from models import VSSM_Fusion as net


def main():
    fusion_model_path = './model_last/my_cross/CTMRI.pth'
    fusionmodel = net(output=1)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    
    if args.gpu >= 0:
        fusionmodel.to(device)
    
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('Fusion model loaded successfully!')

    ir_path = 'path1'
    vi_path = 'path2'
    
    test_dataset = Fusion_dataset('test', ir_path=ir_path, vi_path=vi_path, length=21)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    with torch.no_grad():
        for it, (images_vis, images_ir, name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)

            if args.gpu >= 0:
                images_vis = images_vis.to(device)
                images_ir = images_ir.to(device)

            image_vis_ycrcb = images_vis[:, 0:1, :, :]

            fusion_image = fusionmodel(image_vis_ycrcb, images_ir)
            fusion_image = torch.clamp(fusion_image, 0, 1)  

            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))  

            print(f"Shape of fused_image: {fused_image.shape}")

            fused_image = (fused_image - np.min(fused_image)) / (np.max(fused_image) - np.min(fused_image))
            fused_image = np.uint8(255.0 * fused_image)

            for k in range(len(name)):
                image = fused_image[k, :, :, :]
            
                print(f"Shape of image[{k}]: {image.shape}")
                
          
                if image.shape[-1] == 1:
                    image = np.squeeze(image, axis=-1)
                
                print(f"Shape of image before conversion: {image.shape}")

                try:
                    image = Image.fromarray(image)
                except TypeError as e:
                    print(f"Error converting image: {e}")
                    continue

                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print(f'Fusion {save_path} successfully!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MyFusion with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='net')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=1)
    args = parser.parse_args()

    fused_dir = 'fusion'
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    print('| Testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    main()
