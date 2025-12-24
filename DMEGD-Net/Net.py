import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
from thop import profile
from AKconV import AKConv


def custom_sigmoid(x):
    return 4 * torch.sigmoid(x) - 2

class DownConv(nn.Module):
    def __init__(self, in_channels):
        super(DownConv, self).__init__()
        self.DownConv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.DownConv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.UpConv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.UpConv(x)


class EdgeExpert(nn.Module):
    def __init__(self, in_channels):
        super(EdgeExpert, self).__init__()
        self.EdgeConv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 3, 1, 1, bias=True),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.EdgeConv(x)


class ColorExpert(nn.Module):
    def __init__(self, in_channels):
        super(ColorExpert, self).__init__()
        self.ColorConv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=1, bias=True),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ColorConv(x)


class GatingNetwork(nn.Module):
    def __init__(self, in_channels, num_experts=2, size=512):
        super(GatingNetwork, self).__init__()
        self.Gating = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, padding=1, bias=True),
        )
        self.fc = nn.Linear(2 * size//16 * size//16, num_experts)

    def forward(self, x):
        x = self.Gating(x)
        x = torch.flatten(x, start_dim=1)
        gates = self.fc(x)
        gates = F.softmax(gates, dim=-1)
        return gates


class MixtureOfExperts(nn.Module):
    def __init__(self, in_channels=1024, num_experts=2, size=512):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            EdgeExpert(in_channels=in_channels),
            ColorExpert(in_channels=in_channels)
        ])
        self.gating_network = GatingNetwork(in_channels=in_channels, num_experts=self.num_experts, size=size)

    def forward(self, x):
        edge_output = self.experts[0](x)
        color_output = self.experts[1](x)
        gates = self.gating_network(x)
        gates = gates.unsqueeze(-1).unsqueeze(-1)
        edge_output1 = edge_output * gates[:, 0:1]
        color_output1 = color_output * gates[:, 1:2]
        output = edge_output1 + color_output1
        return output


class FUU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FUU, self).__init__()
        self.FuuConv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.FuuConv(x)


class PALayer(nn.Module):
    def __init__(self, in_channels=512):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//8, 1, 3, padding=1, bias=True),
        )

    def forward(self, x):
        y = custom_sigmoid(self.pa(x))
        return x * y


class CALayer(nn.Module):
    def __init__(self, in_channels=512):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//8, in_channels, 1, padding=0, bias=True),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = custom_sigmoid(self.ca(y))
        return x * y


class DMEGDNet(nn.Module):
    def __init__(self, dim=64, size=512):
        super(DMEGDNet, self).__init__()
        self.C1 = AKConv(3, dim, 9, 1, True)
        self.D1 = DownConv(dim)
        self.C2 = AKConv(dim, dim*2, 6, 1, True)
        self.D2 = DownConv(dim*2)
        self.C3 = AKConv(dim*2, dim*4, 6, 1, True)
        self.D3 = DownConv(dim*4)
        self.C4 = AKConv(dim*4, dim*8, 6, 1, True)
        self.D4 = DownConv(dim*8)

        self.C5 = AKConv(dim*8, dim*16, 6, 1, True)
        self.MOE = MixtureOfExperts(in_channels=dim*16, size=size)
        self.C51 = nn.Conv2d(dim*16, dim*4, 3, 1, 1, bias=True)

        self.CA = CALayer(in_channels=dim*8)
        self.PA = PALayer(in_channels=dim*8)
        self.C52 = nn.Conv2d(dim*8, dim*4, 3, 1, 1, bias=True)

        self.U1 = UpConv(dim*8*2)
        self.F1 = FUU(dim*8*2, dim*8)
        self.U2 = UpConv(dim*4*3)
        self.F2 = FUU(dim*4*3, dim*4)
        self.U3 = UpConv(dim*2*3)
        self.F3 = FUU(dim*2*3, dim*2)
        self.U4 = UpConv(dim*3)
        self.F4 =FUU(dim*3, dim)
        self.c = nn.Conv2d(dim, 3, 3, 1, 1, bias=True)
        self.f = nn.ReLU()

    def forward(self, x):
        d1 = self.D1(self.C1(x))
        d2 = self.D2(self.C2(d1))
        d3 = self.D3(self.C3(d2))
        d4 = self.D4(self.C4(d3))
        ld = self.C5(d4)
        mgu = self.MOE(ld)
        x1 = self.C51(ld)
        ca = self.CA(torch.concat([x1,mgu],dim=1))
        pa =self.PA(ca)
        x2 = self.C52(pa)

        f1 = self.F1(self.U1(torch.concat([x2,x1,d4],dim=1)))
        f2 = self.F2(self.U2(torch.concat([f1, d3],dim=1)))
        f3 = self.F3(self.U3(torch.concat([f2, d2], dim=1)))
        f4 = self.F4(self.U4(torch.concat([f3, d1], dim=1)))
        output = self.f(self.c(f4))
        return output

# 总FLOPs(G): 86.1472 G
# 总Params(M): 31.4738 M


