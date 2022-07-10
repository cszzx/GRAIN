import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, Module
from models.stylegan2.model import Generator


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class GradualStyleBlock(Module):
	def __init__(self, in_c=64, out_c=512, spatial=1024):
		super(GradualStyleBlock, self).__init__()
		self.out_c = out_c
		self.spatial = spatial
		num_pools = int(np.log2(spatial))
		modules = []
		modules1 = []
		modules2 = []
		if num_pools > 4:  # need downsample and cat
			self.conv_first = nn.Sequential(*[
				Conv2d(in_c, in_c, kernel_size=3, stride=2, padding=1),
				nn.LeakyReLU(negative_slope=0.2, inplace=True)])
			for i in range(num_pools - 5):
				modules += [
					Conv2d(in_c, in_c, kernel_size=3, stride=2, padding=1),
					nn.LeakyReLU(negative_slope=0.2, inplace=True)
				]
			self.convs = nn.Sequential(*modules)
		else:
			self.convs = None

		modules1 += [
			Conv2d(in_c, in_c, kernel_size=3, stride=2, padding=1),  # 16*64 -> 8*64
			nn.LeakyReLU(negative_slope=0.2, inplace=True),  
			Conv2d(in_c, in_c*2, kernel_size=3, stride=2, padding=1),  # 8*64 -> 4*128
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			Conv2d(in_c*2, in_c*4, kernel_size=3, stride=2, padding=1),  # 4*128 -> 2*256
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			Conv2d(in_c*4, in_c*8, kernel_size=3, stride=2, padding=1),  # 2*256 -> 1*512
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
		]
		modules2 += [
			Conv2d(in_c, in_c, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			Conv2d(in_c, in_c*2, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			Conv2d(in_c*2, in_c*4, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			Conv2d(in_c*4, in_c*8, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
		]
		self.convs1 = nn.Sequential(*modules1)
		self.convs2 = nn.Sequential(*modules2)
		self.linear1 = nn.Linear(out_c, out_c)
		self.linear2 = nn.Linear(out_c, out_c)

	def forward(self, x):
		if self.convs is not None:
			fea = self.conv_first(x)
			x = self.convs(fea)
		else:
			fea = None
		x1 = self.convs1(x)
		x2 = self.convs2(x)
		x1 = x1.view(-1, self.out_c)
		x2 = x2.view(-1, self.out_c)
		x1 = self.linear1(x1)
		x2 = self.linear2(x2)
		return fea, x1, x2


class Fea2Code(Module):
	def __init__(self):
		super(Fea2Code, self).__init__()
		self.styles = nn.ModuleList()
		for i in range(9):
			if i < 3:
				style = GradualStyleBlock(64, 512, 16)
			else:
				style = GradualStyleBlock(64, 512, 2**(i + 2))
			self.styles.append(style)

	def forward(self, fea):  # len(fea) = 7
		latents = []
		feat = None
		for i in range(8, -1, -1):
			if i == 8:
				feat, c1, c2 = self.styles[i](fea[i-2])
				latents.append(c1)
				latents.append(c2)
			elif i > 2:
				feat, c1, c2 = self.styles[i](fea[i-2] + feat)
				latents.append(c1)
				latents.append(c2)
			else:
				_, c1, c2 = self.styles[i](fea[0] + feat)
				latents.append(c1)
				latents.append(c2)
		latents.reverse()
		out = torch.stack(latents, dim=1)
		return out


class codeStyleGAN(nn.Module):
	def __init__(self, opts):
		super(codeStyleGAN, self).__init__()
		self.opts = opts
		# Define architecture
		self.encoder = Fea2Code()
		self.decoder = Generator(self.opts.output_size, 512, 8)
		ckpt = torch.load(self.opts.stylegan_weights)
		self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
		self.__load_latent_avg(ckpt, repeat=18)
		# Load weights if needed
		self.load_weights() 

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
				inject_latent=None, return_latents=False, alpha=None, return_features=False):
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		input_is_latent = not input_code

		if return_features:
			images, result_latent, feature = self.decoder([codes],
											 input_is_latent=input_is_latent,
											 randomize_noise=randomize_noise,
											 return_latents=return_latents,
											 return_features=True)
			return images, feature
		else:
			images, result_latent = self.decoder([codes],
												input_is_latent=input_is_latent,
												randomize_noise=randomize_noise,
												return_latents=return_latents)

		if return_latents:
			return images, result_latent
		else:
			return images

	def load_weights(self):
		if self.opts.CodeStyleGAN_checkpoint_path is not None:
			print('Loading CodeStyleGAN from checkpoint: {}'.format(self.opts.CodeStyleGAN_checkpoint_path))
			ckpt = torch.load(self.opts.CodeStyleGAN_checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
		else:
			print('No pretrained model. Training from begining!')

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
		if repeat is not None:
			self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
