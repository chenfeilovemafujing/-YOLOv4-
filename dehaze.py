import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob

use_gpu = False


def dehaze_image(image_path):

    data_hazy = Image.open(image_path)
    ori_img = data_hazy
    w, h = data_hazy.size
    data_hazy = data_hazy.resize((480,640), Image.ANTIALIAS)
    data_hazy = (np.asarray(data_hazy)/255.0)

    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2,0,1)
    if use_gpu:
        data_hazy = data_hazy.cuda().unsqueeze(0)
        dehaze_net = net.dehaze_net().cuda()
    else:
        data_hazy = data_hazy.unsqueeze(0)
        dehaze_net = net.dehaze_net()

    dehaze_net.load_state_dict(torch.load('snapshots/Epoch_1_dehazer.pth'))

    clean_image = dehaze_net(data_hazy)
    #torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "results/" + image_path.split("/")[-1])
    #clean_image = clean_image.resize((w, h), Image.ANTIALIAS)
    unloader = transforms.ToPILImage()
    clean_image = clean_image.cpu().clone()
    clean_image = clean_image.squeeze(0)
    clean_image = unloader(clean_image)
    clean_image = clean_image.resize((w, h), Image.ANTIALIAS)
    #torchvision.utils.save_image(clean_image, "results/" + image_path.split("/")[-1])
    clean_image = Image.fromarray(np.concatenate((ori_img, clean_image), axis=1))
    clean_image.save("results/" + image_path.split("/")[-1])


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    test_list = glob.glob("test_images/*")
    test_list = [x.replace('\\', '/') for x in test_list]

    if not os.path.exists('./results'):
        os.mkdir('./results')

    for image in test_list:
        dehaze_image(image)
        print(image, "done!")
