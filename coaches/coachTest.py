import os
import matplotlib

matplotlib.use('Agg')

import torch
from torch.utils.data import DataLoader
from utils import common
from configs import data_configs
from datasets.images_dataset import ImagesDataset
import time
from models.invRRDB import InvRRDB
from models.feaFusion import Fusion
from models.codeStyleGAN import codeStyleGAN
import numpy as np
from PIL import Image


class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0
		self.device = 'cuda:0'
		self.opts.device = self.device

		# Initialize network
		self.net1 = InvRRDB(self.opts).to(self.device)
		self.net1.eval()
		self.net2 = codeStyleGAN(self.opts).to(self.device)
		self.net2.eval()
		self.net3 = Fusion(self.opts).to(self.device)
		self.net3.eval()
		ckpt = torch.load(self.opts.feaFusion_checkpoint_path,  map_location='cpu')
		print('Loading feaFusion from checkpoint: {}'.format(self.opts.feaFusion_checkpoint_path))
		self.net3.load_state_dict(ckpt["state_dict"])

		# Initialize dataset
		self.test_dataset = self.configure_datasets()
		self.n_images = len(self.test_dataset)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)
		self.starttime = time.time()

	def test(self):
		# dir: lr  saves the extreme low-resolution images (16*16)
		# dir: invRRDB  saves the Invertible Extreme Rescaling Module HR results (1024*1024)
		# dir: styleGAN  saves the Scale-Specific Generative Prior Module HR results (1024*1024)
		# dir: fusion  saves the Upscaling Priors Decoding Module HR results (1024*1024) i.e. the final results
		# dir: comp  saves the invRRDB-styleGAN-fusion-gt for convenient comparison
		out_path_lr = os.path.join(self.opts.exp_dir, 'lr')
		os.makedirs(out_path_lr, exist_ok=True)
		out_path_invRRDB = os.path.join(self.opts.exp_dir, 'invRRDB')
		os.makedirs(out_path_invRRDB, exist_ok=True)
		out_path_styleGAN = os.path.join(self.opts.exp_dir, 'styleGAN')
		os.makedirs(out_path_styleGAN, exist_ok=True)
		out_path_fusion = os.path.join(self.opts.exp_dir, 'fusion')
		os.makedirs(out_path_fusion, exist_ok=True)
		out_path_comp = os.path.join(self.opts.exp_dir, 'comp')
		os.makedirs(out_path_comp, exist_ok=True)

		for batch_idx, batch in enumerate(self.test_dataloader):
			if batch_idx >= self.n_images:
				break
			x, y = batch
			with torch.no_grad():
				x, y = x.to(self.device).float(), y.to(self.device).float()
				_, img1, img2, feature1 = self.net1.forward(x)
				img2s, feature2 = self.net2.forward(feature1, return_features=True)
				img3 = self.net3.forward(feature1, feature2)

			for i in range(self.opts.batch_size):
				im_path = self.test_dataset.source_paths[batch_idx]
				x_, img1_, img2_, img2s_, img3_ = common.tensor2im(y[i]), common.tensor2im(img1[i]), common.tensor2im(img2[i]), common.tensor2im(img2s[i]), common.tensor2im(img3[i])
				comp = np.concatenate([np.array(img2_), np.array(img2s_), np.array(img3_), np.array(x_)], axis=1)
				Image.fromarray(comp).save(os.path.join(out_path_comp, os.path.basename(im_path)))
				Image.fromarray(np.array(img1_)).save(os.path.join(out_path_lr, os.path.basename(im_path)))
				Image.fromarray(np.array(img2_)).save(os.path.join(out_path_invRRDB, os.path.basename(im_path)))
				Image.fromarray(np.array(img2s_)).save(os.path.join(out_path_styleGAN, os.path.basename(im_path)))
				Image.fromarray(np.array(img3_)).save(os.path.join(out_path_fusion, os.path.basename(im_path)))

			if batch_idx % 500 == 0:
				endtime = time.time()
				deltime = endtime - self.starttime
				self.starttime = time.time()
				print(f'count {batch_idx}, time {deltime}')

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {self.opts.dataset_type}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
									 target_root=dataset_args['test_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts)
		print(f"Number of test samples: {len(test_dataset)}")
		return test_dataset

