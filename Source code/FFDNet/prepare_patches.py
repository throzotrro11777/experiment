
import argparse
from dataset import prepare_data2

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=\
								  "Building the training patch database")
	parser.add_argument("--gray", action='store_true',default=False,\
						help='prepare grayscale database instead of RGB')
	# Preprocessing parameters patch的长和宽
	parser.add_argument("--patch_size", "--p", type=int, default=70, \
					 help="Patch size")
	parser.add_argument("--stride", "--s", type=int, default=40, \
					 help="Size of stride")
	parser.add_argument("--max_number_patches", "--m", type=int, default=None, \
						help="Maximum number of patches")
	parser.add_argument("--aug_times", "--a", type=int, default=1, \
						help="How many times to perform data augmentation")
	# Dirs
	parser.add_argument("--trainset_dir", type=str, default=None, \
					 help='path of trainset')
	parser.add_argument("--valset_dir", type=str, default=None, \
						 help='path of validation set')
	args = parser.parse_args()

	if args.gray:
		if args.trainset_dir is None:
			args.trainset_dir = 'data/gray/train'
		if args.valset_dir is None:
			args.valset_dir = 'data/gray/Set12'
	else:
		if args.trainset_dir is None:
			# args.trainset_dir = 'data/rgb/CImageNet_expl'
			args.trainset_dir = '/home/liyu/data/denoiseing/exploration_database_and_code/'
		if args.valset_dir is None:
			# args.valset_dir = 'data/rgb/Kodak24'
			args.valset_dir = '/home/liyu/data/denoiseing/Kodak24'

	print("\n### Building databases ###")
	print("> Parameters:")
	for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	prepare_data2(args.trainset_dir,\
					args.valset_dir,\
					args.patch_size,\
					args.stride,\
					args.max_number_patches,\
					aug_times=args.aug_times,\
					gray_mode=args.gray)
