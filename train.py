import argparse

import datetime
import time
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from dataset import *
from model import *
from utils import *

# setup parse
parser = argparse.ArgumentParser(description = 'hyperparameter for cycleGAN')
parser.add_argument('--lr', type = float, default = 2e-4)
parser.add_argument('--total_epochs', type = int, default = 150)
parser.add_argument('--batch_size', type = int, default = 1)
parser.add_argument('--save_epoch', type = int, default = 10)
parser.add_argument('--continue_train', type = str, default= 'on', choices=['on', 'off'])
parser.add_argument('--save_dir', default = './save_model')
parser.add_argument('--weight_cycle', type = int, default = 10)
parser.add_argument('--weight_identity', type = int, default = 5)
parser.add_argument('--img_path', type = str, default = './dataset/')
args = parser.parse_args()

# Hyperparameter
epochs = args.total_epochs
lr = args.lr
b1 = 0.5
b2 = 0.999
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = args.batch_size
save_epoch = args.save_epoch
dir_save_model = args.save_dir
continue_train = args.continue_train
weight_cycle = args.weight_cycle
weight_identity = args.weight_identity

# Datatloader
img_path = args.img_path
monet_path = glob.glob(img_path + 'monet_jpg/*')
photo_path = glob.glob(img_path + 'photo_jpg/*')

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], 
                                 [0.5, 0.5, 0.5])
])

train_dataset = Custom_dataset([monet_path, photo_path], transform, mode = 'train')
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

# monet to photo
netG_A = Generator().to(device)
# photo to monet
netG_B = Generator().to(device)

netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

netG_optim = optim.Adam(itertools.chain(netG_A.parameters(), netG_B.parameters()), lr = lr, betas=(b1, b2))
netD_optim = optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr = lr, betas=(b1, b2))

# weight값 초기화
weight_init(netG_A)
weight_init(netG_B)
weight_init(netD_A)
weight_init(netD_B)

# Loss function 정의
Loss_GAN = nn.MSELoss()
Loss_cycle = nn.L1Loss()
Loss_identity = nn.L1Loss()

train_hist = {}
train_hist['G_losses'] = []
train_hist['D_losses'] = []

print('training is starting...')
t = time.time()

if continue_train == 'on':
    start_epoch, netG_A, netG_B, netD_A, \
        netD_B, netG_optim, netD_optim = \
            load_model(dir_save_model, netG_A, netG_B, netD_A, netD_B, netG_optim, netD_optim, device)

elif not os.path.exists(dir_save_model):
    os.makedirs(dir_save_model)
    start_epoch = 0
else:
    start_epoch = 0

for epoch in range(start_epoch, epochs):
    netG_A.train()
    netG_B.train()
    
    G_losses = 0
    D_losses = 0

    for (A, B) in train_loader:
        
        # A : monet    B : photo
        A, B = A.to(device), B.to(device)
        
        A2B = netG_A(A)
        B2A = netG_B(B)
        
        # train_discriminator
        # Forward pass
        # D_A
        pred_real_A = netD_A(A)
        pred_fake_A = netD_A(B2A.detach())
        
        D_A_loss = Loss_GAN(pred_real_A, torch.ones_like(pred_real_A)) + Loss_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
        
        # D_B
        pred_real_B = netD_B(B)
        pred_fake_B = netD_B(A2B.detach())
        D_B_loss = Loss_GAN(pred_real_B, torch.ones_like(pred_real_B)) + Loss_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
        
        D_loss = (D_A_loss + D_B_loss) / 2
        
        # Backward pass
        netD_optim.zero_grad()
        D_loss.backward()
        netD_optim.step()

        # Generator
        # Forward pass
        A2B2A = netG_B(A2B)
        B2A2B = netG_A(B2A)
        
        pred_fake_A = netD_A(B2A)
        pred_fake_B = netD_B(A2B)
        
        # GAN loss
        G_GAN_loss = Loss_GAN(pred_fake_A, torch.ones_like(pred_fake_A)) + Loss_GAN(pred_fake_B, torch.ones_like(pred_fake_B))
        
        # Cycle loss
        G_cycle_loss = Loss_cycle(A2B2A, A) + Loss_cycle(B2A2B, B)
        
        # Identity loss
        G_identity_loss = Loss_identity(netG_A(B), B) + Loss_identity(netG_B(A), A)
           
        # Backward
        G_loss = G_GAN_loss + weight_cycle * G_cycle_loss + weight_identity * G_identity_loss
                
        netG_optim.zero_grad()
        G_loss.backward()
        netG_optim.step()
                
        D_losses += D_loss / len(train_loader)
        G_losses += G_loss / len(train_loader)
    # lr_scheduler_G.step()
    # lr_scheduler_D.step()

    print(f'[{epoch + 1}/{epochs}] | D_loss : {D_losses:.6f} | G_loss : {G_losses:.6f}')
    train_hist['G_losses'].append(G_losses.item())
    train_hist['D_losses'].append(D_losses.item())

    # 모델 저장
    if (epoch + 1) % save_epoch == 0:
        # save model
        if not os.path.exists(dir_save_model):
            os.makedirs(dir_save_model)
        torch.save({'netG_A' : netG_A.state_dict(),
                    'netG_B' : netG_B.state_dict(),
                    'netD_A' : netD_A.state_dict(),
                    'netD_B' : netD_B.state_dict(),
                    'netG_optim' : netG_optim.state_dict(),
                    'netD_optim' : netD_optim.state_dict()},
                    dir_save_model + '/' + f'resNet_{epoch + 1}.pth')

        print(f'Model is saved at {epoch + 1}epochs')

        save_plot(A, A2B, epoch, 'A2B')
        save_plot(B, B2A, epoch, 'B2A')

sec = time.time() - t
print('learning time :', str(datetime.timedelta(seconds = sec)))
plot_loss(train_hist)