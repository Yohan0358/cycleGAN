import torch
import torch.nn as nn

# 초기화 함수
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Instance') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# residual block
class residual_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1):
        super(residual_block, self).__init__()
        
        self.res = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch,
                              kernel_size = kernel_size, 
                              stride = stride, 
                              padding = padding, 
                              padding_mode = 'reflect',
                              bias = False),
                    nn.InstanceNorm2d(out_ch),
                    nn.ReLU(inplace = True),
                    nn.Conv2d(out_ch, out_ch,
                              kernel_size = kernel_size, 
                              stride = stride, 
                              padding = padding, 
                              padding_mode = 'reflect',
                              bias = False),
                    nn.InstanceNorm2d(out_ch)
        )
        
    def forward(self, x):
        return x + self.res(x)
    
# Upsampling block
class Conv_Up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 4, stride = 2, padding = 1, 
    output_padding = 1, drop_out = True):
        super(Conv_Up, self).__init__()
        
        self.convT = nn.ConvTranspose2d(in_ch, out_ch, 
                                        kernel_size = kernel_size, 
                                        stride = stride, 
                                        padding = padding, 
                                        output_padding = output_padding,
                                        bias= False)
        self.Instance_Norm = nn.InstanceNorm2d(out_ch)
        self.relu = nn.ReLU(inplace =  True)
        self.drop_out = drop_out
        
    def forward(self, x):
        x = self.convT(x)
        x = self.Instance_Norm(x)
        x = self.relu(x)
        if self.drop_out:
            x = nn.Dropout2d(0.5)(x)
            
        return x

# Down sampling block
class Conv_Down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 4, stride = 2, padding = 1, batch_Norm = True):
        super(Conv_Down, self).__init__()
        
        self.conv = nn.Conv2d(in_ch, out_ch, 
                              kernel_size = kernel_size, 
                              stride = stride, 
                              padding = padding,
                              bias = False)
        self.Instance_Norm = nn.InstanceNorm2d(out_ch)
        self.relu = nn.ReLU(inplace = True)
        self.batch = batch_Norm
        
    def forward(self, x):
        x = self.conv(x)
        if self.batch:
            x = self.Instance_Norm(x)        
        x = self.relu(x)
        return x