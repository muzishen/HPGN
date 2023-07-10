# encoding: utf-8
"""
@author:  muzishen
@contact: shenfei140721@126.com
"""


import argparse
import yaml
import os
from  shutil import copyfile

parser = argparse.ArgumentParser(description='Vehcile Reid')
parser.add_argument('--mode', default='train', choices=['train', 'evaluate', 'vis'], help='train or evaluate ')
parser.add_argument('--data_path', default="../resize256_pytorch", help='path of Vehicle image')
parser.add_argument('--num_classes', default=13164,type=int, help='classes number. VeRi776: 576;  ')
parser.add_argument('--seed', default=2020,type=int, help='random seed;  ')
########################################  Dataload    ###################################################
parser.add_argument("--name", default='HPGN',type=str, help='the name of log')
parser.add_argument('--h', default=256,type=int, help='height of input image ')
parser.add_argument('--w', default=256,type=int, help='weight of input image ')
parser.add_argument('--mean', default=[0.391, 0.411, 0.411],type=float, help='mean of veri776 image =[0.418,0.323,0.340],normal: [0.485, 0.456, 0.406],veID:[0.391,0.411,0.411]')
parser.add_argument('--std', default=[0.246, 0.243, 0.243],type=float, help='std of veri776 image =[0.183,0.160,0.164], normal:[0.229, 0.224, 0.225],veID:[0.246, 0.243, 0.243]')
parser.add_argument('--sampler', default=1, help='0:normal sampler; 0.5: MGN_sampler; 1 :randomID' )
########################################  trick    ###################################################
parser.add_argument('--cutmix_beta', default=0, type=float, help='hyperparameter cutmix ')
parser.add_argument('--cutmix_prob', default=0, type=float, help='cut mixup probability')
parser.add_argument('--pad', default=10, type=float, help='pading')
parser.add_argument('--cj', default=0, type=float, help='0: not use color jitter')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')

#########################################  Basic Config ###################################################
# parser.add_argument('--optimizer', default=1, help='0:Adam,  1:SGD' )
parser.add_argument("--batchid",    default=32,   help='the batch for id')
parser.add_argument("--batchimage", default=4,    help='the batch of per id')
parser.add_argument('--epoch',      default=151,  help='number of epoch to train')

########################################  Network trainning Config  ##########################################
parser.add_argument('--resume',          default=61, help='0: do not use resume trainning' )
parser.add_argument('--warm_epoch',      default=10,    type=int, help='the first k epoch ')
parser.add_argument('--warm_base_lr',    default=3e-4,   help='warm up initial learning_rate')
parser.add_argument('--lr',              default=3e-2,   help='initial learning_rate')
parser.add_argument('--lr_scheduler',    default=[50, 85, 120], help='MultiStepLR,decay the learning rate')
parser.add_argument('--margin',          default=1.2,  type=float, help='margin of triplet loss')
parser.add_argument('--triplet_lambda',  default=1,  type=float, help='weight of triplet loss in total loss')
parser.add_argument('--softmax_lambda',  default=2,  type=float, help='weight of softmax loss in total loss')
parser.add_argument('--AdaBound',  default=0,  type=float, help='0:not use adaBound')

###################################### Network testing Config   ###################################################
parser.add_argument('--model_weight',  default='60',  help='load choice weights number for test ')
parser.add_argument("--batchtest",  default=64,  help='the batch size for test')

######################################  Network visdom Config   ###################################################
parser.add_argument('--query_image',  default='../pytorch/probe/0005/0005_c003_00077670_0.jpg', help='path to the image you want to query')


opt = parser.parse_args()
print(opt)


save_dir = './log/'+ opt.name
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
copyfile('./network.py', save_dir + '/network.py')
weights_dir = save_dir +'/weights'

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
if not os.path.isdir(weights_dir):
    os.mkdir(weights_dir)

with open('./log/%s/opts.yaml' % opt.name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)