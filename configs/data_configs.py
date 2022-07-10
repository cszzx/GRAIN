from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'celebs_invertible_super_resolution': {
		'transforms': transforms_config.InvSuperResTransforms,
		'train_source_root': dataset_paths['celeba_train'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'constant': {
		'transforms': transforms_config.ConstantTransforms,
		'train_source_root': '',
		'train_target_root': '',
		'test_source_root': '',
		'test_target_root': '',
	},
}
