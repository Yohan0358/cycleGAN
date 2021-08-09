import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from PIL import Image

def unNormalize(img, mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], plot_show = True, title = None, figsize = (10, 4)):
    img = make_grid(img)
    img = img.cpu().detach()
    np_img = np.transpose(img.numpy(), (1, 2, 0))
    np_img = np_img * std + mean
    np_img = (np_img * 255).astype('uint8')
    if plot_show:
        plt.figure(figsize = figsize)
        plt.imshow(np_img)
        if title:
            plt.title(title)
        plt.show()  
    else:
        return np_img

def save_plot(A, A2B, epoch, fig_name):
    img = torch.cat([A, A2B], dim = 0)
    img = unNormalize(img, plot_show=False)
    img = Image.fromarray(img)
    if not os.path.exists('./save_images'):
        os.makedirs('./save_images')
    img.save('./save_images/' + fig_name + '_' +str(epoch + 1) + '.jpg')

def load_model(dir_save_model, netG_A, netG_B, netD_A, netD_B, netG_optim, netD_optim, device):
    if not os.path.exists(dir_save_model):
        start_epoch = 0
        return start_epoch, netG_A, netG_B, netD_A, netD_B, netG_optim, netD_optim
    
    model_list = glob.glob('./save_model/' + '*.pth')
    extract_num = [int(''.join(list(filter(str.isdigit, m)))) for m in model_list]
    start_epoch = max(extract_num)
    recent_model_idx = extract_num.index(start_epoch)

    load_dic = torch.load(model_list[recent_model_idx], map_location = device)

    netG_A.load_state_dict(load_dic['netG_A'])
    netG_B.load_state_dict(load_dic['netG_B'])
    netD_A.load_state_dict(load_dic['netD_A'])
    netD_B.load_state_dict(load_dic['netD_B'])
    netG_optim.load_state_dict(load_dic['netG_optim'])
    netD_optim.load_state_dict(load_dic['netD_optim'])

    print(f'Load model at {model_list[recent_model_idx]}')
    return start_epoch, netG_A, netG_B, netD_A, netD_B, netG_optim, netD_optim


def plot_loss(train_hist):
    plt.figure()
    plt.plot(train_hist['G_losses'], label = 'G_loss')
    plt.plot(train_hist['D_losses'], label = 'D_loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    A = torch.randn(1, 3, 256, 256)
    A2B = torch.randn(1, 3, 256, 256)
    save_plot(A, A2B, 5, 'A2B')