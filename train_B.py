import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import os, argparse

from ColorJitter import ColorJitter
from datetime import datetime
from dataset import RescaleT
from dataset import RandomCrop
from dataset import ToTensorLab
from dataset import SalObjDataset

from model.PAFR import Net
from src.G_smooth import SSIM
from src.G_crf import GatedCRF


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
opt = parser.parse_args()

model = Net(3)
model.cuda()
optimizer = optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
l = 0.3
CJ = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = SSIM(window_size=11, size_average=True)
loss_lsc = GatedCRF().cuda()

tra_image_dir = './train_data/EORRSD-S/img/'
tra_gt_dir = './train_data/EORRSD-S/gt/'
tra_mask_dir = './train_data/EORRSD-S/mask/'

tra_img_name_list = [tra_image_dir + f for f in os.listdir(tra_image_dir) if f.endswith('.jpg')]
tra_gt_name_list = [tra_gt_dir + f for f in os.listdir(tra_gt_dir) if f.endswith('.png')]
tra_mask_name_list = [tra_mask_dir + f for f in os.listdir(tra_mask_dir) if f.endswith('.png')]

print("---")
print("train images: ", len(tra_img_name_list))
print("---")

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    gt_name_list=tra_gt_name_list,
    mask_name_list=tra_mask_name_list,
    transform=transforms.Compose([
        RescaleT(256),
        RandomCrop(224),
        ToTensorLab(flag=0),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=0)

train_num = len(salobj_dataloader)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def sscloss(x, y, alpha):
    ssim_out = 1 - ssim_loss(x, y)
    l1_loss = torch.mean(torch.abs(x-y))
    loss_ssc = alpha*ssim_out + (1-alpha)*l1_loss
    return loss_ssc


def train(salobj_dataloader, model, optimizer, epoch):
    model.train()

    for i, data in enumerate(salobj_dataloader, start=1):
        optimizer.zero_grad()

        images, gts, masks = data['image'], data['gt'], data['mask']

        images_colorJ = CJ(images)
        images = images.type(torch.FloatTensor)
        gts    = gts.type(torch.FloatTensor)
        masks  = masks.type(torch.FloatTensor)

        images_colorJ = Variable(images_colorJ.cuda(), requires_grad=False)
        images = Variable(images.cuda(), requires_grad=False)
        gts    = Variable(gts.cuda(), requires_grad=False)
        masks  = Variable(masks.cuda(), requires_grad=False)

        img_size = images.size(2) * images.size(3) * images.size(0)
        ratio = img_size / torch.sum(masks)

        #############################################################################################
        image_scaleCJ = F.interpolate(images_colorJ, scale_factor=0.3, mode='bilinear', align_corners=True)  # [1, 3, 67, 67]
        d1, d2, d3, d4, d5, d6, d7 = model(images)
        d1_s, d2_s, d3_s, d4_s, d5_s, d6_s, d7_s = model(image_scaleCJ)

        d1_scale = F.interpolate(d1, size=d1_s.size()[2:], mode='bilinear', align_corners=True)  # [1, 1, 96, 96]
        d2_scale = F.interpolate(d2, size=d2_s.size()[2:], mode='bilinear', align_corners=True)

        loss_ssc1 = sscloss(d1_s, d1_scale, 0.85)
        loss_ssc2 = sscloss(d2_s, d2_scale, 0.85)
        loss_ssc = loss_ssc1 + loss_ssc2

        image_ = F.interpolate(images, scale_factor=0.25, mode='bilinear', align_corners=True)
        sample = {'rgb': image_}

        d1_ = F.interpolate(d1, scale_factor=0.25, mode='bilinear', align_corners=True)
        loss1_lsc = loss_lsc(d1_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
        loss1 = ratio*bce_loss(d1*masks, gts*masks) + l * loss1_lsc + loss_ssc

        d2_ = F.interpolate(d2, scale_factor=0.25, mode='bilinear', align_corners=True)
        loss2_lsc = loss_lsc(d2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
        loss2 = ratio*bce_loss(d2*masks, gts*masks) + l * loss2_lsc

        d3_ = F.interpolate(d3, scale_factor=0.25, mode='bilinear', align_corners=True)
        loss3_lsc = loss_lsc(d3_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
        loss3 = ratio*bce_loss(d3*masks, gts*masks) + l * loss3_lsc

        d4_ = F.interpolate(d4, scale_factor=0.25, mode='bilinear', align_corners=True)
        loss4_lsc = loss_lsc(d4_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
        loss4 = ratio*bce_loss(d4*masks, gts*masks) + l * loss4_lsc

        d5_ = F.interpolate(d5, scale_factor=0.25, mode='bilinear', align_corners=True)
        loss5_lsc = loss_lsc(d5_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
        loss5 = ratio*bce_loss(d5*masks, gts*masks) + l * loss5_lsc

        d6_ = F.interpolate(d6, scale_factor=0.25, mode='bilinear', align_corners=True)
        loss6_lsc = loss_lsc(d6_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
        loss6 = ratio*bce_loss(d6*masks, gts*masks) + l * loss6_lsc

        d7_ = F.interpolate(d7, scale_factor=0.25, mode='bilinear', align_corners=True)
        loss7_lsc = loss_lsc(d7_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
        loss7 = ratio*bce_loss(d7*masks, gts*masks) + l * loss7_lsc

        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 10 == 0 or i == train_num:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], refine_loss: {:0.4f}, total_loss: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, train_num, loss1.data, loss.data))

    model_dir = "./saved_models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), model_dir + 'mode_B' + '_%d' % epoch + '.pth')


print('Start Training!')


if __name__ == '__main__':
    for epoch in range(1, opt.epoch+1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(salobj_dataloader, model, optimizer, epoch)
