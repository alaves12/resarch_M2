import argparse
import os

parser = argparse.ArgumentParser(description="Pytorch AtJ_model Evaluation")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda? Default is True")
parser.add_argument("--load",  default=False)
parser.add_argument("--model_load", type=str, default="23", help="model path")
parser.add_argument("--test", type=str, default="testset", help="testset path")
parser.add_argument("--size", type=int, default=256, help="testset path")
parser.add_argument("--exp", type=str, default="exp-1", help="model path")
parser.add_argument("--weight", type=str, default="srn_dil", help="model path")
parser.add_argument("--dataset", type=str, default="Train_full", help="model path")
parser.add_argument("--times", type=int, default=3, help="model path")
parser.add_argument("--epoch", type=int, default=100, help="the number of epochs")
parser.add_argument("--length", type=str, default=1000)
parser.add_argument("--id", type=str, default=2)
parser.add_argument("--lr_sche",default=True)

opt = parser.parse_args()
cuda = opt.cuda
device_label = 'GPU' if opt.cuda else 'CPU'


weight_dir = '/data/runs/save_dir/weight_epoch_D/{}'.format(opt.weight)
optimizer_dir = '/data/runs/save_dir/weight_epoch_D/optimizer/{}'.format(opt.weight)
exp_dir = '/data/runs/save_dir/exp/{}'.format(opt.exp)
exp_dir_image = '/data/runs/save_dir/exp/{}/image'.format(opt.exp)
exp_dir_image_number = os.path.join(exp_dir_image, '{}'.format(opt.weight))