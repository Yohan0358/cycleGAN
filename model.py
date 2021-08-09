import torch
import torch.nn as nn
from layers import *

# resnet network
class Generator(nn.Module):
    def __init__(self, n_feature = 64, n_residu = 9):
        super(Generator, self).__init__()
        '''
        kernel   D64 - D128 - D256 - R256 * n - U128 - U64 - U3
        filter   7x7 - 3x3  - 3x3  -          -  3x3 - 3x3 - 7x7
        stride    1     2      2       1          2     2     1
        '''
        self.main = nn.Sequential(
            nn.ReflectionPad2d(3),
            Conv_Down(3, n_feature, 7, 1, 0, False),
            Conv_Down(n_feature * 1, n_feature * 2, 3, 2),
            Conv_Down(n_feature * 2, n_feature * 4, 3, 2),
            *[residual_block(n_feature * 4, n_feature * 4) for _ in range(n_residu)],
            Conv_Up(n_feature * 4, n_feature * 2, 3, 2, 1),
            Conv_Up(n_feature * 2, n_feature * 1, 3, 2, 1),
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_feature, 3, 7, 1, 0, bias = False),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.main(x)
        return x

class Discriminator(nn.Module):
    '''
    C64-C128-C256-C512-C512-1
    '''
    def __init__(self, n_features = 64):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            Conv_Down(3, n_features, batch_Norm = False),
            Conv_Down(n_features * 1, n_features * 2),
            Conv_Down(n_features * 2, n_features * 4),
            Conv_Down(n_features * 4, n_features * 8, stride = 1),
            nn.Conv2d(n_features * 8, 1, 4, 1, 1, bias = False),
        )
        
    def forward(self, x):
        x = self.main(x)
        return x

if __name__ == '__main__':
    # def test():
    #     G = Generator()
    #     D = Discriminator()
    #     x = torch.randn(1, 3, 256, 256)
    #     out_G = G(x)
    #     out_D = D(out_G)
    #     print(out_G.shape)
    #     print(out_D.shape)

    # test()
    model = torch.load('./save_model/resNet_50.pth', map_location='cpu')
    print(model.keys())