from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', default='experiments/test', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--dataset_type', default='celebs_invertible_super_resolution', type=str, help='Type of dataset/experiment to run')
		self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels')
		self.parser.add_argument('--output_size', default=1024, type=int, help='Output size')

		self.parser.add_argument('--invRRDB', action='store_true', help='Train Invertible Extreme Rescaling Module')
		self.parser.add_argument('--codeStyleGAN', action='store_true', help='Train Scale-Specific Latent Code Prediction Layers')
		self.parser.add_argument('--Fusion', action='store_true', help='Train Upscaling Priors Decoding Module')
		self.parser.add_argument('--finetune', action='store_true', help='Finetune the whole model')

		self.parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing')
		self.parser.add_argument('--workers', default=1, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=1, type=int, help='Number of test dataloader workers')

		self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use') 
		self.parser.add_argument('--start_from_latent_avg', default=True, type=bool, help='Whether to add average latent vector to generate codes from encoder.')

		self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
		self.parser.add_argument('--id_lambda', default=0.1, type=float, help='ID loss multiplier factor')
		self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
		self.parser.add_argument('--lr_l2_lambda', default=0, type=float, help='L2 loss multiplier factor for LR images')
		self.parser.add_argument('--adv_lambda', default=0.01, type=float, help='Adversarial loss multiplier factor')

		self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--stylegan_dis_weights', default=model_paths['stylegan_ffhq_dis'], type=str, help='Path to StyleGAN discriminator model weights')
		self.parser.add_argument('--InvRRDB_checkpoint_path', default='pretrained_models/invRRDB_model.pt', type=str, help='Path to InvRRDB model checkpoint')
		self.parser.add_argument('--InvRRDBdis_checkpoint_path', default=None, type=str, help='Path to InvRRDB discriminator model checkpoint')
		self.parser.add_argument('--CodeStyleGAN_checkpoint_path', default='pretrained_models/codeStyleGAN_model.pt', type=str, help='Path to codeStyleGAN model checkpoint')
		self.parser.add_argument('--feaFusion_checkpoint_path', default='pretrained_models/Fusion_model.pt', type=str, help='Path to feaFusion model checkpoint')

		self.parser.add_argument('--max_steps', default=300000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=5000, type=int, help='Interval for logging train images during training') # 1500
		self.parser.add_argument('--board_interval', default=5000, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=20000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=20000, type=int, help='Model checkpoint interval')

	def parse(self):
		opts = self.parser.parse_args()
		return opts
