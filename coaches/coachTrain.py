import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from coaches.ranger import Ranger
import time
from models.invRRDB import InvRRDB, Discriminator
from models.stylegan2.model import Discriminator as StyleGANDis
from models.feaFusion import Fusion
from models.codeStyleGAN import codeStyleGAN
from configs.paths_config import model_paths
from torch.autograd import Variable
import numpy as np


class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0
		self.device = 'cuda:0'
		self.opts.device = self.device

		# Initialize network
		if self.opts.invRRDB:
			self.net1 = InvRRDB(self.opts).to(self.device)
			self.opts.lr_l2_lambda = 0.1
		elif self.opts.codeStyleGAN:
			self.net1 = InvRRDB(self.opts).to(self.device)
			self.net1.eval()
			self.net2 = codeStyleGAN(self.opts).to(self.device)
			self.opts.lr_l2_lambda = 0
		elif self.opts.Fusion:
			self.net1 = InvRRDB(self.opts).to(self.device)
			self.net1.eval()
			self.net2 = codeStyleGAN(self.opts).to(self.device)
			self.net2.eval()
			self.net3 = Fusion(self.opts).to(self.device)
			self.opts.lr_l2_lambda = 0
		elif self.opts.finetune:
			self.net1 = InvRRDB(self.opts).to(self.device)
			self.net2 = codeStyleGAN(self.opts).to(self.device)
			self.net3 = Fusion(self.opts).to(self.device)
			self.opts.lr_l2_lambda = 0.1
			self.opts.learning_rate = 0.00005
		else:
			raise Exception("Need to choose a network to train!!!")

		# Discriminator
		if self.opts.invRRDB:
			self.dis = Discriminator((3, 1024, 1024), self.opts).to(self.device)
			self.d_optim = Ranger(self.dis.parameters(), self.opts.learning_rate)
			Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
			self.GANLoss = nn.BCEWithLogitsLoss().to(self.device)
			self.valid = Variable(Tensor(np.ones((1, 1, 16, 16))), requires_grad=False)
			self.fake = Variable(Tensor(np.zeros((1, 1, 16, 16))), requires_grad=False)
		else:
			self.dis = StyleGANDis(self.opts.output_size).to(self.device)
			d_ckpt = torch.load(self.opts.stylegan_dis_weights, map_location='cpu')
			self.dis.load_state_dict(d_ckpt["d"])
			self.d_optim = Ranger(self.dis.parameters(), self.opts.learning_rate)
			print("Loading the pretrained StyleGAN discriminator.")

		if (self.opts.Fusion or self.opts.finetune) and (self.opts.feaFusion_checkpoint_path is not None):
			ckpt = torch.load(self.opts.feaFusion_checkpoint_path,  map_location='cpu')
			print('Loading feaFusion from checkpoint: {}'.format(self.opts.feaFusion_checkpoint_path))
			self.net3.load_state_dict(ckpt["state_dict"])

		# Initialize loss
		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps
		self.starttime = time.time()

	def train(self):
		d_loss = 0
		d_iter = 1
		if self.opts.invRRDB:
			self.net1.train()
			d_iter = 6
		elif self.opts.codeStyleGAN:
			self.net2.train()
			d_iter = 5
		elif self.opts.Fusion:
			self.net3.train()
			d_iter = 3
		elif self.opts.finetune:
			self.net1.train()
			self.net2.train()
			self.net3.train()
			d_iter = 3

		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				self.optimizer.zero_grad()
				x, y = batch
				x, y = x.to(self.device).float(), y.to(self.device).float()

				if self.opts.invRRDB:
					if self.global_step % d_iter == 0:
						self.requires_grad(self.dis, True)
						self.requires_grad(self.net1, False)
						_, img1, img2, _ = self.net1.forward(x) 
						y_ = F.interpolate(y, size=[256, 256], mode="bilinear")
						img2_ = F.interpolate(img2, size=[256, 256], mode="bilinear")
						real_dis = self.dis(y_)
						fake_dis = self.dis(img2_)
						loss_real = self.GANLoss(real_dis - fake_dis.mean(0, keepdim=True), self.valid)
						loss_fake = self.GANLoss(fake_dis - real_dis.mean(0, keepdim=True), self.fake)
						d_loss = (loss_real + loss_fake) / 2
						self.d_optim.zero_grad()
						d_loss.backward()
						self.d_optim.step()
						del img1, img2, img2_, real_dis, fake_dis
						self.requires_grad(self.dis, False)
						self.requires_grad(self.net1, True)
				else:
					if self.global_step % d_iter == 0:
						self.requires_grad(self.dis, True)
						if self.opts.codeStyleGAN:
							self.requires_grad(self.net2, False)
						elif self.opts.Fusion:
							self.requires_grad(self.net3, False)
						elif self.opts.finetune:
							self.requires_grad(self.net1, False)
							self.requires_grad(self.net2, False)
							self.requires_grad(self.net3, False)
						if self.opts.codeStyleGAN:
							_, img1, img2, feature1 = self.net1(x)
							img2s = self.net2.forward(feature1) 
							real_dis = self.dis(y)
							fake_dis = self.dis(img2s)
						else:
							_, img1, img2, feature1 = self.net1.forward(y)
							img2s, feature2 = self.net2.forward(feature1, return_features=True)
							img3 = self.net3.forward(feature1, feature2)
							real_dis = self.dis(y)
							fake_dis = self.dis(img3)
						d_loss = self.d_logistic_loss(real_dis, fake_dis)
						self.d_optim.zero_grad()
						d_loss.backward()
						self.d_optim.step()
						d_regularize = self.global_step % 16 == 0
						if d_regularize:
							y.requires_grad = True
							real_dis = self.dis(y)
							r1_loss = self.d_r1_loss(real_dis, y)
							(10 / 2 * r1_loss * 16 + 0 * real_dis[0]).backward()
							self.d_optim.step()
						del img1, img2, img2s, feature1, real_dis, fake_dis
						if not self.opts.codeStyleGAN:
							del img3, feature2
						self.requires_grad(self.dis, False)
						if self.opts.codeStyleGAN:
							self.requires_grad(self.net2, True)
						elif self.opts.Fusion:
							self.requires_grad(self.net3, True)
						elif self.opts.finetune:
							self.requires_grad(self.net1, True)
							self.requires_grad(self.net2, True)
							self.requires_grad(self.net3, True)
				
				if self.opts.invRRDB:
					_, img1, img2, _ = self.net1.forward(x)
					loss, loss_dict, id_logs = self.calc_loss(y, img2, img1)
					img2s = torch.zeros_like(img2).to(self.device)  # just for logging images
					img3 = torch.zeros_like(img2).to(self.device)   # just for logging images
				elif self.opts.codeStyleGAN:
					with torch.no_grad():
						_, img1, img2, feature1 = self.net1(x)
					img2s = self.net2.forward(feature1)
					loss, loss_dict, id_logs = self.calc_loss(y, img2s)
					img3 = torch.zeros_like(img2).to(self.device)   # just for logging images
				elif self.opts.Fusion:
					with torch.no_grad():
						_, img1, img2, feature1 = self.net1.forward(x)
						img2s, feature2 = self.net2.forward(feature1, return_features=True)
					img3 = self.net3.forward(feature1, feature2)
					loss, loss_dict, id_logs = self.calc_loss(y, img3)
				elif self.opts.finetune:
					_, img1, img2, feature1 = self.net1.forward(x)
					img2s, feature2 = self.net2.forward(feature1, return_features=True)
					img3 = self.net3.forward(feature1, feature2)
					loss, loss_dict, id_logs = self.calc_loss(y, img3, img1)
				loss.backward()
				self.optimizer.step()

				# Logging related
				if self.global_step % self.opts.image_interval == 0:
					self.parse_and_log_images(y, img1, img2, img2s, img3, title='images/train/faces')
				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')
					if self.opts.adv_lambda > 0:
						print("dis_loss: ", float(d_loss))

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)
				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1

	def validate(self, num=50):  # validate the fist 50 images of test dataset
		if self.opts.invRRDB:
			self.net1.eval()
		elif self.opts.codeStyleGAN:
			self.net2.eval()
		elif self.opts.Fusion:
			self.net3.eval()
		elif self.opts.finetune:
			self.net1.eval()
			self.net2.eval()
			self.net3.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			x, y = batch
			x, y = x.to(self.device).float(), y.to(self.device).float()
			with torch.no_grad():
				if self.opts.invRRDB:
					_, img1, img2, _ = self.net1.forward(x)
					loss, cur_loss_dict, id_logs = self.calc_loss(y, img2, img1)
					img2s = torch.zeros_like(img2).to(self.device)  # just for logging images
					img3 = torch.zeros_like(img2).to(self.device)   # just for logging images
				elif self.opts.codeStyleGAN:
					_, img1, img2, feature1 = self.net1(x)
					img2s = self.net2.forward(feature1)
					loss, cur_loss_dict, id_logs = self.calc_loss(y, img2s)
					img3 = torch.zeros_like(img2).to(self.device)   # just for logging images
				elif self.opts.Fusion:
					_, img1, img2, feature1 = self.net1.forward(x)
					img2s, feature2 = self.net2.forward(feature1, return_features=True)
					img3 = self.net3.forward(feature1, feature2)
					loss, cur_loss_dict, id_logs = self.calc_loss(y, img3)
				elif self.opts.finetune:
					_, img1, img2, feature1 = self.net1.forward(x)
					img2s, feature2 = self.net2.forward(feature1, return_features=True)
					img3 = self.net3.forward(feature1, feature2)
					loss, cur_loss_dict, id_logs = self.calc_loss(y, img3, img1)
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(y, img1, img2, img2s, img3, title='images/test/faces', subscript='{:04d}'.format(batch_idx))
			if batch_idx >= num:
				break
		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		if self.opts.invRRDB:
			self.net1.train()
		elif self.opts.codeStyleGAN:
			self.net2.train()
		elif self.opts.Fusion:
			self.net3.train()
		elif self.opts.finetune:
			self.net1.train()
			self.net2.train()
			self.net3.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		if not self.opts.finetune:
			save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
			save_name_dis = 'best_model_dis.pt' if is_best else f'iteration_{self.global_step}_dis.pt'
			checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
			checkpoint_path_dis = os.path.join(self.checkpoint_dir, save_name_dis)

			save_dict, save_dict_dis = self.__get_save_dict()

			torch.save(save_dict, checkpoint_path)
			torch.save(save_dict_dis, checkpoint_path_dis)
		else:
			save_name1 = 'best_invRRDB_model.pt' if is_best else f'iteration_{self.global_step}_invRRDB.pt'
			save_name2 = 'best_codeStyleGAN_model.pt' if is_best else f'iteration_{self.global_step}_codeStyleGAN.pt'
			save_name3 = 'best_Fusion_model.pt' if is_best else f'iteration_{self.global_step}_Fusion.pt'
			save_name_dis = 'best_model_dis.pt' if is_best else f'iteration_{self.global_step}_dis.pt'
			checkpoint_path1 = os.path.join(self.checkpoint_dir, save_name1)
			checkpoint_path2 = os.path.join(self.checkpoint_dir, save_name2)
			checkpoint_path3 = os.path.join(self.checkpoint_dir, save_name3)
			checkpoint_path_dis = os.path.join(self.checkpoint_dir, save_name_dis)

			save_dict1, save_dict2, save_dict3, save_dict_dis = self.__get_save_dict()

			torch.save(save_dict1, checkpoint_path1)
			torch.save(save_dict2, checkpoint_path2)
			torch.save(save_dict3, checkpoint_path3)
			torch.save(save_dict_dis, checkpoint_path_dis)

		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def configure_optimizers(self):
		params = []
		if self.opts.invRRDB:
			params = list(self.net1.parameters())
		elif self.opts.codeStyleGAN:
			params = list(self.net2.parameters())
		elif self.opts.Fusion:
			params = list(self.net3.parameters())
		elif self.opts.finetune:
			params = list(self.net1.parameters()) + list(self.net2.parameters()) + list(self.net3.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {self.opts.dataset_type}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
									  target_root=dataset_args['train_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
		test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
									 target_root=dataset_args['test_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts)

		print(f"Number of training samples: {len(train_dataset)}")
		print(f"Number of test samples: {len(test_dataset)}")
		return train_dataset, test_dataset

	def calc_loss(self, y, hr, lr=None):
		loss_dict = {}
		loss = 0.0
		id_logs = None
		if self.opts.lr_l2_lambda > 0:
			lr_gt = F.interpolate(y, size=[16, 16], mode="bilinear")
			loss_lr_l2 = F.mse_loss(lr, lr_gt)
			loss_dict['loss_lr_l2'] = float(loss_lr_l2)
			loss += loss_lr_l2 * self.opts.lr_l2_lambda
		if self.opts.l2_lambda > 0:
			loss_hr_l2 = F.mse_loss(hr, y)
			loss_dict['loss_hr_l2'] = float(loss_hr_l2)
			loss += loss_hr_l2 * self.opts.l2_lambda
		if self.opts.id_lambda > 0:
			y_ = F.interpolate(y, size=[256, 256], mode="bilinear")
			hr_ = F.interpolate(hr, size=[256, 256], mode="bilinear")
			loss_id, sim_improvement, id_logs = self.id_loss(hr_, y_, y_)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_id * self.opts.id_lambda
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(hr, y)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda
		if self.opts.adv_lambda > 0:
			if self.opts.invRRDB:
				y_ = F.interpolate(y, size=[256, 256], mode="bilinear")
				hr_ = F.interpolate(hr, size=[256, 256], mode="bilinear")
				fake_dis = self.dis(hr_)
				real_dis = self.dis(y_)
				adv_loss = self.GANLoss(fake_dis - real_dis.mean(0, keepdim=True), self.valid)
				loss_dict['adv_loss'] = float(adv_loss)
				loss += adv_loss * self.opts.adv_lambda
			else:
				y_hat_dis = self.dis(hr)
				adv_loss = self.g_nonsaturating_loss(y_hat_dis)
				loss_dict['adv_loss'] = float(adv_loss)
				loss += adv_loss * self.opts.adv_lambda
		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs

	def requires_grad(self, model, flag=True):
		for p in model.parameters():
			p.requires_grad = flag 
	def d_logistic_loss(self, real_pred, fake_pred):
		real_loss = F.softplus(-real_pred)
		fake_loss = F.softplus(fake_pred)
		return real_loss.mean() + fake_loss.mean()
	def d_r1_loss(self, real_pred, real_img):
		grad_real, = autograd.grad(
			outputs=real_pred.sum(), inputs=real_img, create_graph=True
			)
		grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
		return grad_penalty
	def g_nonsaturating_loss(self, fake_pred):
		loss = F.softplus(-fake_pred).mean()
		return loss


	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		endtime = time.time()
		deltime = endtime - self.starttime
		self.starttime = time.time()
		print(f'Metrics for {prefix}, step {self.global_step}, time {deltime}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	def parse_and_log_images(self, y, img1, img2, img2s, img3, title, subscript=None, display_count=1):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'gt_face': common.tensor2im(y[i]),
				'styleGAN_face': common.tensor2im(img2s[i]),
				'invRRDB_face': common.tensor2im(img2[i]),
				'final_face': common.tensor2im(img3[i]),
				'lr_face': common.tensor2im(img1[i]),
				'diff_styleGAN_face': common.tensor2im(y[i]-img2s[i]),
				'diff_invRRDB_face': common.tensor2im(y[i]-img2[i]),
				'diff_final_face': common.tensor2im(y[i]-img3[i]),
			}
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = self.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
		else:
			path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def vis_faces(self, log_hooks):
		display_count = len(log_hooks)
		fig = plt.figure(figsize=(50, 24 * display_count))
		gs = fig.add_gridspec(2, 4)
		for i in range(display_count):
			hooks_dict = log_hooks[i]
			fig.add_subplot(gs[0, 0])
			plt.imshow(hooks_dict['gt_face'])
			plt.title('gt_face')
			fig.add_subplot(gs[0, 1])
			plt.imshow(hooks_dict['invRRDB_face'])
			plt.title('invRRDB_face')
			fig.add_subplot(gs[0, 2])
			plt.imshow(hooks_dict['styleGAN_face'])
			plt.title('styleGAN_face')
			fig.add_subplot(gs[0, 3])
			plt.imshow(hooks_dict['final_face'])
			plt.title('final_face')

			fig.add_subplot(gs[1, 0])
			plt.imshow(hooks_dict['lr_face'])
			plt.title('lr_face')
			fig.add_subplot(gs[1, 1])
			plt.imshow(hooks_dict['diff_invRRDB_face'])
			plt.title('diff_invRRDB_face')
			fig.add_subplot(gs[1, 2])
			plt.imshow(hooks_dict['diff_styleGAN_face'])
			plt.title('diff_style_face')
			fig.add_subplot(gs[1, 3])
			plt.imshow(hooks_dict['diff_final_face'])
			plt.title('diff_final_face')
		plt.tight_layout()
		return fig

	def __get_save_dict(self):
		if not self.opts.finetune:
			if self.opts.invRRDB:
				save_dict = {
					'state_dict': self.net1.state_dict(),
					'opts': vars(self.opts)
				}
			elif self.opts.codeStyleGAN:
				save_dict = {
					'state_dict': self.net2.state_dict(),
					'opts': vars(self.opts)
				}
			elif self.opts.Fusion:
				save_dict = {
					'state_dict': self.net3.state_dict(),
					'opts': vars(self.opts)
				}
			save_dict_dis = {
				'state_dict': self.dis.state_dict(),
			}
			return save_dict, save_dict_dis
		else:
			save_dict1 = {
				'state_dict1': self.net1.state_dict(),
				'opts': vars(self.opts)
			}
			save_dict2 = {
				'state_dict2': self.net2.state_dict(),
				'opts': vars(self.opts)
			}
			save_dict3 = {
				'state_dict3': self.net3.state_dict(),
				'opts': vars(self.opts)
			}
			save_dict_dis = {
				'state_dict': self.dis.state_dict(),
			}
			return save_dict1, save_dict2, save_dict3, save_dict_dis
