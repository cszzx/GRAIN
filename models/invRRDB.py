import functools
import torch
import torch.nn as nn


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt

def make_layer(block, n_layers):
	layers = []
	for _ in range(n_layers):
		layers.append(block())
	return nn.Sequential(*layers)


class Noise(nn.Module):
	def __init__(self):
		super(Noise, self).__init__()

	def forward(self, input, train=True):
		input = (input + 1. ) / 2.
		input = input * 255.0
		if train:
			noise = torch.nn.init.uniform_(torch.zeros_like(input), -0.5, 0.5).cuda()
			output = input + noise
			output = torch.clamp(output, 0, 255.)
		else:
			output = input.round() * 1.0
			output = torch.clamp(output, 0, 255.)
		return (output / 255.0 - 0.5) / 0.5


class ResidualDenseBlock_5C(nn.Module):
	def __init__(self, nf=64, gc=32, bias=True):
		super(ResidualDenseBlock_5C, self).__init__()
		# gc: growth channel, i.e. intermediate channels
		self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
		self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
		self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
		self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
		self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
		self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x):
		x1 = self.lrelu(self.conv1(x))
		x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
		x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
		x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
		x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
		return x5 * 0.2 + x


class ResidualDenseBlock_4C(nn.Module):
	def __init__(self, nf=64, gc=32, bias=True):
		super(ResidualDenseBlock_4C, self).__init__()
		# gc: growth channel, i.e. intermediate channels
		self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
		self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
		self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
		self.conv4 = nn.Conv2d(nf + 3 * gc, nf, 3, 1, 1, bias=bias)
		self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x):
		x1 = self.lrelu(self.conv1(x))
		x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
		x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
		x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
		return x4 * 0.2 + x


class RRDB(nn.Module):
	'''Residual in Residual Dense Block'''
	def __init__(self, nf, gc=32):
		super(RRDB, self).__init__()
		self.RDB1 = ResidualDenseBlock_5C(nf, gc)
		self.RDB2 = ResidualDenseBlock_5C(nf, gc)
		self.RDB3 = ResidualDenseBlock_5C(nf, gc)

	def forward(self, x):
		out = self.RDB1(x)
		out = self.RDB2(out)
		out = self.RDB3(out)
		return out * 0.2 + x


class RRDB2(nn.Module):
	'''Residual in Residual Dense Block with fewer blocks'''
	def __init__(self, nf, gc=32):
		super(RRDB2, self).__init__()
		self.RDB1 = ResidualDenseBlock_4C(nf, gc)

	def forward(self, x):
		out = self.RDB1(x)
		return out * 0.2 + x


class RRDBEncNet(nn.Module):
	def __init__(self, in_nc=3, out_nc=3, nf=64, gc=32, nb=10, scale=6):
		super(RRDBEncNet, self).__init__()
		self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

		modules = []
		for i in range(scale):
			modules.append(nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1, bias=True),
										 nn.LeakyReLU(negative_slope=0.2, inplace=True)))
		self.body = nn.Sequential(*modules)

		RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
		self.RRDB_trunk = make_layer(RRDB_block_f, nb)
		self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

		self.out = nn.Sequential(*[
								nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
								nn.LeakyReLU(negative_slope=0.2, inplace=True),
								nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
								])

	def forward(self, x):
		x = self.conv_first(x)
		fea = self.body(x)
		trunk = self.trunk_conv(self.RRDB_trunk(fea))
		fea = fea + trunk
		out = self.out(fea)
		return out


class RRDBDecNet(nn.Module):
	def __init__(self, in_nc=3, out_nc=3, nf=64, gc=32, nb=23, scale=6, feature=False):
		super(RRDBDecNet, self).__init__()
		self.feature = feature
		self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

		RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

		self.RRDB_trunk16 = make_layer(RRDB_block_f, 3)
		self.trunk_conv16 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
		self.upsample16 = nn.Upsample(scale_factor=2, mode='nearest')

		self.RRDB_trunk32 = make_layer(RRDB_block_f, 3)
		self.trunk_conv32 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
		self.upsample32 = nn.Upsample(scale_factor=2, mode='nearest')

		self.RRDB_trunk64 = make_layer(RRDB_block_f, 3)
		self.trunk_conv64 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
		self.upsample64 = nn.Upsample(scale_factor=2, mode='nearest')

		self.RRDB_trunk128 = make_layer(RRDB_block_f, 2)
		self.trunk_conv128 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
		self.upsample128 = nn.Upsample(scale_factor=2, mode='nearest')

		self.RRDB_trunk256 = make_layer(RRDB_block_f, 1)
		self.trunk_conv256 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
		self.upsample256 = nn.Upsample(scale_factor=2, mode='nearest')

		RRDB_block_f4 = functools.partial(RRDB2, nf=nf, gc=gc)
		self.RRDB_trunk512 = make_layer(RRDB_block_f4, 1)
		self.trunk_conv512 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
		self.upsample512 = nn.Upsample(scale_factor=2, mode='nearest')
		
		self.RRDB_trunk1024 = make_layer(RRDB_block_f4, 1)
		self.trunk_conv1024 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

		self.out = nn.Sequential(*[
								nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
								nn.LeakyReLU(negative_slope=0.2, inplace=True),
								nn.Conv2d(nf, int(nf/2), 3, 1, 1, bias=True),
								nn.LeakyReLU(negative_slope=0.2, inplace=True),
								nn.Conv2d(int(nf/2), out_nc, 3, 1, 1, bias=True)
								])

	def forward(self, x, return_feature=False):
		x = self.conv_first(x)  # 16×16×64
		features = []
		
		trunk = self.trunk_conv16(self.RRDB_trunk16(x))
		fea = x + trunk
		features.append(fea)
		fea = self.upsample16(fea)

		trunk = self.trunk_conv32(self.RRDB_trunk32(fea))
		fea = fea + trunk
		features.append(fea)
		fea = self.upsample32(fea)

		trunk = self.trunk_conv64(self.RRDB_trunk64(fea))
		fea = fea + trunk
		features.append(fea)
		fea = self.upsample64(fea)

		trunk = self.trunk_conv128(self.RRDB_trunk128(fea))
		fea = fea + trunk
		features.append(fea)
		fea = self.upsample128(fea)
		
		trunk = self.trunk_conv256(self.RRDB_trunk256(fea))
		fea = fea + trunk
		features.append(fea)
		fea = self.upsample256(fea)

		trunk = self.trunk_conv512(self.RRDB_trunk512(fea))
		fea = fea + trunk
		features.append(fea)
		fea = self.upsample512(fea)

		trunk = self.trunk_conv1024(self.RRDB_trunk1024(fea))
		fea = fea + trunk
		features.append(fea)

		out = self.out(fea)

		if return_feature:
			return out, features

		return out


class InvRRDB(nn.Module):
	def __init__(self, opts):
		super(InvRRDB, self).__init__()
		self.opts = opts
		self.enc = RRDBEncNet()
		self.dec = RRDBDecNet()
		self.quantization = Noise()
		self.load_weights()

	def forward(self, x, y=None):
		img1 = self.enc(x)
		img1 = self.quantization(img1)
		img2, features = self.dec(img1, return_feature=True)
		return x, img1, img2, features

	def load_weights(self):
		if self.opts.InvRRDB_checkpoint_path is not None:
			print('Loading InvRRDB from checkpoint: {}'.format(self.opts.InvRRDB_checkpoint_path))
			ckpt = torch.load(self.opts.InvRRDB_checkpoint_path, map_location='cpu')
			self.enc.load_state_dict(get_keys(ckpt, 'enc'), strict=True)
			self.dec.load_state_dict(get_keys(ckpt, 'dec'), strict=True)
		else:
			print('No pretrained model. Training from begining!')


class Discriminator(nn.Module):
	def __init__(self, input_shape, opts):
		super(Discriminator, self).__init__()
		self.opts = opts
		self.input_shape = input_shape
		in_channels, in_height, in_width = self.input_shape
		patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
		self.output_shape = (1, patch_h, patch_w)

		def discriminator_block(in_filters, out_filters, first_block=False):
			layers = []
			layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
			if not first_block:
				layers.append(nn.BatchNorm2d(out_filters))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
			layers.append(nn.BatchNorm2d(out_filters))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		layers = []
		in_filters = in_channels
		for i, out_filters in enumerate([64, 128, 256, 512]):
			layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
			in_filters = out_filters

		layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

		self.model = nn.Sequential(*layers)
		self.load_weights()

	def forward(self, img):
		return self.model(img)

	def load_weights(self):
		if self.opts.InvRRDBdis_checkpoint_path is not None:
			d_ckpt = torch.load(self.opts.InvRRDBdis_checkpoint_path, map_location='cpu')
			print('Loading InvRRDB dis from checkpoint: {}'.format(self.opts.InvRRDBdis_checkpoint_path))
			self.model.load_state_dict(d_ckpt["state_dict"])
		else:
			print('No pretrained discriminator model. Training from begining!')

		



