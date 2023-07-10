"""
@author:  muzishen
@contact: shenfei140721@126.com
"""

import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from  optimizer import build_optimizer
import torch
from torch.optim import lr_scheduler
import time
from opt import opt
from data import Data
from network import HPGN
from loss import Loss
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking
from utils.lr_scheduler import LRScheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


class Main():
    def __init__(self, model, loss, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.model = model.to('cuda')
        self.loss = loss
        self.scheduler = LRScheduler(base_lr=opt.lr, step=opt.lr_scheduler,
                                    factor=0.1, warmup_epoch=opt.warm_epoch,
                                    warmup_begin_lr=opt.warm_base_lr, warmup_mode='linear')

    def train(self,epoch):

        lr = self.scheduler.update(epoch-1)
        self.optimizer = build_optimizer(self.model, optim='sgd',lr=lr, weight_decay=5e-4, momentum=0.9)
        print("lr:%lf" % lr)

        self.model.train()
        running_loss = 0.0
        running_triplet_loss = 0.0
        running_crossentropy_loss =0.0
        batchsize_num = 0

        for step, (inputs, labels) in enumerate(self.train_loader):
            batchsize_num += 1
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            now_batch_size, _, _, _ = inputs.shape

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss,Triplet_Loss, CrossEntropy_Loss = self.loss(outputs, labels)
            running_loss += loss
            running_triplet_loss += Triplet_Loss
            running_crossentropy_loss += CrossEntropy_Loss

            loss.backward()
            self.optimizer.step()
#############################record batch_loss ###############################
        print()
        print('Batch size:', (opt.batchimage * opt.batchid), '  Batchsize number have:', batchsize_num, )
        epoch_loss = running_loss/(batchsize_num)
        epoch_triplet_loss = running_triplet_loss/(batchsize_num)
        epoch_crossentropy_loss = running_crossentropy_loss/(batchsize_num)
        ###log###
        with open('./log/%s/loss.txt' % opt.name, 'a') as loss_file:
            loss_file.write(' Training Epoch: %2d, lr: %.8f, Ce_Loss: %.8f, Tri_Loss: %.8f, Epoch Total Loss: %.8f\n'
                            % ( epoch, lr, epoch_crossentropy_loss,epoch_triplet_loss,epoch_loss))
            print('Training  Epoch Ce_loss: {:.4f},  Epoch Tri_loss: {:.4f}, Epoch Total Loss: {:.4f} '
                  .format(epoch_crossentropy_loss, epoch_triplet_loss, epoch_loss))
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    def evaluate(self):
        self.model.eval()

        print('Model are extracting features now, this may take a few minutes ! Please Waiting !')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids,
                    query_cams=None,
                    gallery_cams=None,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, query_cams=None, gallery_cams=None)

            return r, m_ap

        #########################   re rank  ##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        r, m_ap = rank(dist)

        with open('./log/%s/test_acc.txt' % opt.name, 'a') as test_file:
            test_file.write('[With Re-Ranking] Epoch: {} mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}\n'
              .format(epoch, m_ap, r[0], r[2], r[4], r[9]))
            print('[With Re-Ranking] Epoch: {}  mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(epoch, m_ap, r[0], r[2], r[4], r[9]))

        #########################no re rank##########################
        dist = cdist(qf, gf)
        r, m_ap = rank(dist)
        with open('./log/%s/test_acc.txt' % opt.name, 'a') as test_file:
            test_file.write('[Without Re-Ranking] Epoch: {} mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}\n'
              .format(epoch, m_ap, r[0], r[2], r[4], r[9]))
            print('[Without Re-Ranking] Epoch: {} mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(epoch, m_ap, r[0], r[2], r[4], r[9]))
        for ii  in range(0,30):
            print (r[ii])
            ii += 1
        print (m_ap)


    def vis(self):

        self.model.eval()

        gallery_path = data.testset.imgs
        gallery_label = data.testset.ids

        # Extract feature
        print('Model are extracting features now, this may take a few minutes ! Please Waiting !')

        query_feature = extract_feature(model, tqdm([(torch.unsqueeze(data.query_image, 0), 1)]))
        gallery_feature = extract_feature(model, tqdm(data.test_loader))

        # sort images
        query_feature = query_feature.view(-1, 1)
        score = torch.mm(gallery_feature, query_feature)
        score = score.squeeze(1).cpu()
        score = score.numpy()

        index = np.argsort(score)  # from small to large
        index = index[::-1]  # from large to small

        fig = plt.figure(figsize=(16, 4))

        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        plt.imshow(plt.imread(opt.query_image))
        ax.set_title('query')

        print('Top 10 images are as follow:')

        for i in range(10):
            img_path = gallery_path[index[i]]
            print(img_path)

            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            plt.imshow(plt.imread(img_path))
            ax.set_title(img_path.split('/')[-1][:9])

        fig.savefig('./log/'+ opt.name+"/show.png")
        print('result saved to show.png')


if __name__ == '__main__':

    data = Data()
    model = HPGN(opt.num_classes)
    model = torch.nn.DataParallel(model)
    loss = Loss()
    main = Main(model, loss, data)


    if opt.mode == 'train':
######################################### resume model ###################################
        if opt.resume == 0:
            start_epoch = 1
        else:
            start_epoch = opt.resume
            resume_path = ('./log/{}/weights/model_{}.pt'.format(opt.name, opt.resume))
            print('Model Resume Path:',resume_path)
            model.load_state_dict(torch.load(resume_path))
            print('loading  finish !!!')
########################################## resume finish ###################################
        since = time.time()
        for epoch in range(start_epoch, (opt.epoch + 1) - start_epoch):
            print()
            print('Epoch {}/{}'.format(epoch, opt.epoch))
            print('-' * 10)

            main.train(epoch=epoch)
            if epoch > 30:
                if epoch % 15 == 0:
                    print('\nstart evaluate')
                    main.evaluate()
                torch.save(model.state_dict(), ('./log/{}/weights/model_{}.pt'.format(opt.name,epoch)))

    if opt.mode == 'evaluate':
        print('start evaluate')
        weight_num = ('./log/{}/weights/model_{}.pt'.format(opt.name,opt.model_weight))
        print('Model Path:', weight_num)
        epoch = weight_num
        model.load_state_dict(torch.load(weight_num))
        main.evaluate()


    if opt.mode == 'vis':
        print('visualize')
        weight_num = ('./log/{}/weights/model_{}.pt'.format(opt.name, opt.mdoel_weight))
        print('Model Path:', weight_num)
        epoch = weight_num
        model.load_state_dict(torch.load(weight_num))
        main.vis()
