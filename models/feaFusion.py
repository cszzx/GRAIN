import torch
import torch.nn as nn
import torch.nn.functional as F
from models.helpers import SEModule


class Fusion(nn.Module):
	def __init__(self, opts):
		super(Fusion, self).__init__()
		self.opts = opts
		self.se1 = SEModule(192, 16)
		self.conv1 = nn.Sequential(*[
								nn.Conv2d(192, 128, 3, 1, 1, bias=True),
								nn.LeakyReLU(negative_slope=0.2, inplace=True),
								nn.Conv2d(128, 128, 3, 1, 1, bias=True),
								nn.LeakyReLU(negative_slope=0.2, inplace=True)
								])
		self.se2 = SEModule(128, 16)
		self.conv2 = nn.Sequential(*[
								nn.Conv2d(128, 64, 3, 1, 1, bias=True),
								nn.LeakyReLU(negative_slope=0.2, inplace=True),
								nn.Conv2d(64, 64, 3, 1, 1, bias=True),
								nn.LeakyReLU(negative_slope=0.2, inplace=True)
								])
		self.se3 = SEModule(96, 16)
		self.conv3 = nn.Sequential(*[
								nn.Conv2d(96, 32, 3, 1, 1, bias=True),
								nn.LeakyReLU(negative_slope=0.2, inplace=True),
								nn.Conv2d(32, 32, 3, 1, 1, bias=True),
								nn.LeakyReLU(negative_slope=0.2, inplace=True)
								])
		self.se4 = SEModule(224, 16)
		self.out = nn.Sequential(*[
								nn.Conv2d(224, 128, 3, 1, 1, bias=True),
								nn.LeakyReLU(negative_slope=0.2, inplace=True),
								nn.Conv2d(128, 128, 3, 1, 1, bias=True),
								nn.LeakyReLU(negative_slope=0.2, inplace=True),
								nn.Conv2d(128, 64, 3, 1, 1, bias=True),
								nn.LeakyReLU(negative_slope=0.2, inplace=True),
								nn.Conv2d(64, 64, 3, 1, 1, bias=True),
								nn.LeakyReLU(negative_slope=0.2, inplace=True),
								nn.Conv2d(64, 3, 3, 1, 1, bias=True)
								])

	def forward(self, x, y):
		z1 = torch.cat([x[-3], y[-3]], dim=1)
		z2 = torch.cat([x[-2], y[-2]], dim=1)
		z3 = torch.cat([x[-1], y[-1]], dim=1)
		z1 = self.se1(z1)
		z1 = self.conv1(z1)
		z1 = F.interpolate(z1, scale_factor=4, mode='nearest')
		z2 = self.se2(z2)
		z2 = self.conv2(z2)
		z2 = F.interpolate(z2, scale_factor=2, mode='nearest')
		z3 = self.se3(z3)
		z3 = self.conv3(z3)
		z = torch.cat([z1, z2, z3], dim=1)
		z = self.se4(z)
		z = self.out(z)
		return z


