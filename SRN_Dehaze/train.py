from datas.dataset import MyDataset
import torch
import torch.nn as nn
from model.srn_dil import SRN
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms as T
from torchvision.utils import save_image
from math import log10
import os
import math
from model.vgg16 import Vgg16
import torchvision.utils as vutils
import glob
from natsort import natsorted
from PIL import Image
from options import *
from loss import get_loss


print(opt)
cuda = opt.cuda

def make_settings():
    if cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.id)

    os.makedirs(weight_dir,exist_ok=True)
    os.makedirs(optimizer_dir, exist_ok=True)

    os.makedirs(exp_dir,exist_ok=True)
    os.makedirs(exp_dir_image,exist_ok=True)

    os.makedirs(exp_dir_image_number,exist_ok=True)

    for i in range(opt.times):
        os.makedirs(os.path.join(exp_dir_image_number,'j{}'.format(i+1)),exist_ok=True)

    for i in range(opt.times):
            os.makedirs(os.path.join(exp_dir_image_number,'j{}'.format(i+1)),exist_ok=True)


#perceptual loss
def perceptualloss(a, b, trans=False):
    f = vgg(a)
    f_c = vgg(b)

    loss_1_1 = criterionCAE(f[0],f_c[0].detach())
    lossR_1_1 = criterionCAE(f[1], f_c[1].detach())
    loss_2_1 = criterionCAE(f[3], f_c[3].detach())
    lossR_2_2 = criterionCAE(f[4], f_c[4].detach())
    lossR_3_3 = criterionCAE(f[5], f_c[5].detach())
    
    if trans:
        loss_c_t = loss_1_1 + loss_2_1
        return loss_c_t
    
    else:
        loss_c = lossR_1_1 + lossR_2_2 + lossR_3_3
        return loss_c

# dark channel loss
def darkloss(a,b):
    maxpool = nn.MaxPool3d(kernel_size=(3, 9, 9), stride=1, padding=(0, 4, 4))
    a_dark = -maxpool(-a)
    b_dark = -maxpool(-b)

    loss_dark = criterionCAE(a_dark, b_dark.detach())

    return loss_dark

# learning rate scheduling
def lr_schedule_cosdecay(t,T,init_lr=1e-4):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr



def train(opt,model):
    if opt.lr_sche:
        print("\nlernig rate scheduling\n")

    Total = len(train_loader)*opt.epoch

    if opt.load:
        model_path = os.path.join(weight_dir, "netD_{}_{}.pth".format(opt.weight,opt.model_load))
        model = torch.load(model_path)

    step = 0
    for epoch in range(100):
        if opt.load:
            if epoch <= int(opt.model_load):
                continue
        epoch_loss_j1 = 0
        epoch_loss_j2 = 0
        epoch_loss_j3 = 0
        epoch_psnr = 0
        count = 1
        pro_size = len(train_loader)
        bar_size = 30

        for i,data in enumerate(train_loader):
            step+=1
            count+= 1
            if cuda:
                data['hazy'] = data['hazy'].cuda()
                data['clean'] = data['clean'].cuda()
            loss_D = 0
            gt = data['clean']

            optimizer_D.zero_grad()
            
            # Generator
            j1, j2, J= model(data['hazy'])
            down = nn.AvgPool2d(3,2,1)
            gthh = down(down(gt))
            gth = down(gt)

            outputs = [j1,j2,J]
            targets = [gthh,gth,gt]

            loss_D,losses = get_loss(outputs=outputs, targets=targets, loss_funcs=loss_funcs, times=opt.times ,weights=[1.0,1.0,2.0])
            loss_j1, loss_j2, loss_J = losses

            epoch_loss_j1 += loss_j1 / len(train_loader)
            epoch_loss_j2 += loss_j2 / len(train_loader)
            epoch_loss_j3 += loss_J / len(train_loader)

            loss_D.backward()

            mse = criterion(J, gt)
            psnr = 10 * log10(1 / mse.item())
            epoch_psnr += psnr / len(train_loader)

            optimizer_D.step()
            pro_bar = ('=' * int(((i+1)/pro_size)*bar_size)) + (' ' * (bar_size - int(((i+1)/pro_size)*bar_size)))
            print(f'\rtrain epoch:{epoch} [{pro_bar}] {(i+1)}/{pro_size}', end='', flush=True)

            if epoch == 0 and count == 30:
                print("\n{}_{}_{}\tloss_j1: {:.5f}\tloss_j2: {:.5f}\tloss_J: {:.5f}\tpsnr: {:.5f}".format(opt.weight, epoch, i,loss_j1,loss_j2, loss_J, psnr))

            if (epoch % 3 == 2 or epoch == 0) and count % len(dataset):
                test(epoch)

            if opt.lr_sche:
                lr=lr_schedule_cosdecay(step,Total)
                for param_group in optimizer_D.param_groups:
                    param_group["lr"] = lr               
        
        print("\n{} | loss_j1:{:.4f} | loss_j2:{:.4f} | loss_J:{:.4f} | psnr:{:.4f}\n".format(opt.weight, epoch_loss_j1,epoch_loss_j2, epoch_loss_j3, epoch_psnr))

        # model save
        if opt.dataset != 'Train_ITS_comp':
            if (((epoch+1) > 2 and (epoch+1) % 3 ==0) or epoch == 0):
                weight_path = os.path.join(weight_dir, 'netD_{}_{:02}.pth'.format(opt.weight,epoch))
                torch.save(model, weight_path)

        else:
            if (epoch+1) % 2 == 0 or epoch == 0:
                weight_path = os.path.join(weight_dir, 'netD_{}_{:02}.pth'.format(opt.weight,epoch))
                torch.save(model, weight_path)

def test(epoch):
    image_list = glob.glob(os.path.join('./testset', '*'))
    train_images = []
    for i_test,image_name in enumerate(natsorted(image_list)):
        if i_test == 15:
            continue
        img = Image.open(image_name)
        img = T.Resize((256,256))(img)
        img = T.ToTensor()(img)
        img = torch.unsqueeze(img, 0)
        
        with torch.no_grad():
            if cuda:
                model = model.cuda()
                model.train(False)
                img = img.cuda()
            else:
                img = img.cpu()
                model = model.cpu()
            model.train(False)
            model.eval()
            # feeding forward
            output = model(img)
        
        for k in range(len(output)):
            im_output = output[k].cpu()
            if i_test == 0:
                train_images.append(im_output)
            else:
                train_images[k] = torch.cat((train_images[k],im_output),dim=0)
        
        for k in range(len(output)):
            grids = vutils.make_grid(train_images[k],nrow=7)
            save_image(grids, os.path.join(exp_dir_image_number, 'j{}'.format(k+1), 'j{}_{}.jpg'.format(k+1,epoch)))

if __name__=='__main__':
    
    if opt.dataset!="Train_ITS_comp":
            dataset = MyDataset(opt.dataset, crop_size=(opt.size,opt.size),length=opt.length)
    else:
        dataset = MyDataset(opt.dataset, crop_size=(opt.size,opt.size))

    vgg = Vgg16()
    vgg = vgg.cuda()
    vgg = vgg.eval()

    # model
    model = SRN()
    model = model.cuda()
    model.train(True)

    # dataloader
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    #loss function
    criterion = nn.MSELoss()
    criterionCAE = nn.L1Loss()
    criterionBCE = nn.BCELoss()

    loss_funcs = [criterionCAE,darkloss,perceptualloss]

    # optimizer
    optimizer_D = optim.Adam(model.parameters(),lr=1e-4)

    make_settings()
    train(opt,model)