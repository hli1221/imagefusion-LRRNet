
class Args():
	# For training
	path_ir = '/your/training/datasets/path/to/lwir/'
	vgg_model_dir = './model/vgg'
	cuda = 1
	lr = 0.00001
	epochs = 4
	batch_size = 8

	# Network Parameters
	Height = 128
	Width = 128

	n = 128  # number of filters
	channel = 1  # 1 - gray, 3 - RGB
	s = 3  # filter size
	stride = 1
	num_block = 4  # number of LRR blocks, 2, 4, 6, 8
	#num_bblock = 5  # number of backbone blocks

	resume_model = None
	save_fusion_model = "./model"
	save_loss_dir = "./model/loss_v1"







