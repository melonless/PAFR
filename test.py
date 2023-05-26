import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image

from dataset import RescaleT
from dataset import ToTensorLab
from dataset import SalObjDataset

from model.PAFR import Net
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import matplotlib.pyplot as plt

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi + 1e-8)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')


if __name__ == '__main__':
    image_dir = './test_data/'
    prediction_dir = './test_data/'
    model_dir = './saved_models/premodel.pth'

    img_name_list = [image_dir + f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, gt_name_list=[], mask_name_list=[],
                                        transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    print("...load model...")
    model = Net(3)
    model.load_state_dict(torch.load(model_dir))
    model.cuda()
    model.eval()

    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split("/")[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, da = model(inputs_test)  # [1, 64, 256, 256]

        pred = d1[:, 0, :, :]  # [1, 256, 256]
        # pred = normPRED(pred)        # normalization [1, 256, 256]
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, da
