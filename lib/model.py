"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from lib.networks import NetG, NetD, weights_init
from lib.visualizer import Visualizer
from lib.loss import l2_loss, l1_loss
from lib.evaluate import evaluate

import cv2
import time
import copy
import csv
import parameters as params


import matplotlib.pyplot as plt
import lib.heatMap as heatMap




class BaseModel():
    """ Base Model for ganomaly
    """
    def __init__(self, opt, dataloader):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
        self.epoch4Test = 0

    ##
    def set_input(self, input:torch.Tensor):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            # print(input[0].size())
            # print(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])

    ##
    def seed(self, seed_value):
        """ Seed 
        
        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_enc', self.err_g_enc.item())])
        # errors = OrderedDict([
        #     ('err_d', self.err_d.item()),
        #     ('err_g', self.err_g.item()),
        #     ('err_g_adv', self.err_g_adv.item()),
        #     ('err_g_con', self.err_g_con.item()),
        #     ('err_g_enc', self.err_g_enc.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG_%s.pth' % (weight_dir, epoch))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD_%s.pth' % (weight_dir, epoch))

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """
        self.epoch4Test += 1

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            # self.optimize()
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            # if self.total_steps % self.opt.save_image_freq == 0:
            if self.epoch % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                #print(f'img Saved to')
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)

        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()
            res = self.test()
            # if res[self.opt.metric] > best_auc:
            if (self.epoch % self.opt.save_weight_freq == 0):
                best_auc = res[self.opt.metric]
                self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name)

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']
                directory = f'./output/{self.name.lower()}/{self.opt.dataset}/test/images/fake'
                try:
                    self.netg.load_state_dict(pretrained_dict)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            self.batchNum = 0
            ab_RGB = []
            is_abList = []
            stT = time.time()
            allFiles, _ = map(list, zip(*self.dataloader['test'].dataset.samples))
            total_test_size = len(allFiles)
            # h = HoughCircleDetection('.bmp', self.opt.isize)
            ab_scores_dataSet = []
            for i, data in tqdm(enumerate(self.dataloader['test'], 0)):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                
                self.set_input(data)
                self.input
                self.fake, latent_i, latent_o = self.netg(self.input)

                error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
                time_o = time.time()
                
                #print(f'processing time: {time.time() - startTime}')########################################################################################
                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    
                    real_img = real.cpu().data.numpy().squeeze()
                    generated_img = fake.cpu().data.numpy().squeeze()
                    diff_img, ch3_diff_img = heatMap.calc_diff(real_img, generated_img, self.opt.batchsize, params.filter_thres)

                    anomaly_img = np.zeros(shape=(self.opt.isize, self.opt.isize, self.opt.nc))
                    
                    
                    anomaly_img = heatMap.Draw_Anomaly_image(real_img, diff_img, ch3_diff_img, self.opt.batchsize)
                    
                    rawPATH = './RAW' # 1280x720 원본 이미지 경로.
                    rawPATH = params.raw_PATH
                    print(f'rawPATH: {rawPATH}')
                    allFiles, _ = map(list, zip(*self.dataloader['test'].dataset.samples))
                    sav_fName = allFiles[i]
                    sav_fName = sav_fName.replace("\\", '/')
                    sav_fName = sav_fName[sav_fName.rfind('/')+1:]

                    print(f'{i+1: 5d} / {total_test_size} : {i / total_test_size * 100: .2f}%')
                    raw_img, new_diff_img, is_abnormal = heatMap.DrawResult(diff_img, sav_fName, rawPATH, params=None)

                    ab_RGB.append(np.sum(new_diff_img) / (self.opt.isize*self.opt.isize*255)) ## ab_RGB를 Draw_Result 하고 계산해야함!
                    is_abList.append(is_abnormal)
                    
                    if raw_img is None: # 총 3523 개의 RAW Image 중 8개의 사진에서 원 검출을 실패했을 경우.
                        continue
                    
                    real_img = cv2.normalize(real_img, real_img, 0, 255, cv2.NORM_MINMAX)
                    
                    #make_pannel 이전에 적용시킬것
                    real_img = self.BGR2RGB(real_img)
                    generated_img = cv2.normalize(generated_img, generated_img, 0, 255, cv2.NORM_MINMAX)
                    generated_img = self.BGR2RGB(generated_img)
                    
                    anomaly_img = self.BGR2RGB(anomaly_img)
                    newImg = self.make_result_panel(raw_img, real_img, generated_img, anomaly_img, ab_RGB, params.ab_thres, is_abList, params.use_abscore)

                    # cv2.imwrite(f'./output/{self.name.lower()}/{self.opt.dataset}/test/images/anomaly/{sav_fName[:-4]}_anomaly{params.PREFIX_SAV}', np.transpose(anomaly_img, (1,2,0)))
                    # cv2.imwrite(f'./output/{self.name.lower()}/{self.opt.dataset}/test/images/fake/{sav_fName[:-4]}_fake{params.PREFIX_SAV}', np.transpose(generated_img, (1,2,0)))
                    cv2.imwrite(f'./output/{self.name.lower()}/{self.opt.dataset}/test/images/{sav_fName[:-4]}_result{params.PREFIX_SAV}', newImg)

            print(f'{time.time() - stT: .4f}sec')
            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            auc = evaluate(self.gt_labels, self.an_scores, ab_RGB, self.epoch4Test, params.ab_thres, is_abList, params.use_abscore, self.opt, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), (self.opt.metric, auc)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance
    
    def BGR2RGB(self, arr):
        arr = np.transpose(arr, (1,2,0)) #(128,128,3)
        arr = np.transpose(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), (2,0,1))
        return arr
    
    def make_result_panel(self, raw_img, real_img, generated_img, anomaly_img, ab_RGB, ab_thres, is_abList, use_abscore=True,predict = False):
        houseParty = []
        tmp0 = []
        tmp1 = []
        tmp2 = []
        if(self.opt.batchsize == 1):
            tmp0.append(np.transpose(real_img, (1,2,0)))
            tmp1.append(np.transpose(generated_img, (1,2,0)))
            tmp2.append(np.transpose(anomaly_img, (1,2,0)))
            houseParty.append(np.hstack(tuple(tmp0)))
            houseParty.append(np.hstack(tuple(tmp1)))
            houseParty.append(np.hstack(tuple(tmp2)))

        else:
            for bts in range(self.opt.batchsize):
                tmp0.append(np.transpose(real_img[bts], (1,2,0)))
                tmp1.append(np.transpose(generated_img[bts], (1,2,0)))
                tmp2.append(np.transpose(anomaly_img[bts], (1,2,0)))
            houseParty.append(np.hstack(tuple(tmp0)))
            houseParty.append(np.hstack(tuple(tmp1)))
            houseParty.append(np.hstack(tuple(tmp2)))

        houseParty = tuple(houseParty)
        addImg = np.vstack(houseParty)

        scorePanel = np.zeros(shape=(self.opt.batchsize, int(self.opt.isize/2), self.opt.isize, self.opt.nc))
        for bts in range(self.opt.batchsize):
            tmp = np.transpose(scorePanel[bts], (2,0,1))
            # print(is_abList)
            # exit(0)
            if predict:
                if( (use_abscore and ab_RGB >= ab_thres) or ((not use_abscore) and is_abList)): #abnormal
                    #tp-RED
                    self.diagosis_result = 'Abormal'
                    tmp[2] = 234
                    tmp[1] = 67
                    tmp[0] = 53

                else:
                    #tn-GREEN
                    self.diagnois_result = 'Normal'
                    tmp[2] = 52
                    tmp[1] = 168
                    tmp[0] = 83

            else:
                if( (use_abscore and ab_RGB[bts + self.batchNum*self.opt.batchsize] >= ab_thres) or ((not use_abscore) and is_abList[bts + self.batchNum*self.opt.batchsize])): #abnormal
                    #tp-RED
                    if (self.gt_labels[bts + self.batchNum*self.opt.batchsize] == 0):
                        tmp[2] = 234
                        tmp[1] = 67
                        tmp[0] = 53
                    #fp-CYAN
                    else:
                        tmp[2] = 92
                        tmp[1] = 222
                        tmp[0] = 226
                    
                else:
                    #tn-GREEN
                    if (self.gt_labels[bts + self.batchNum*self.opt.batchsize] == 1):
                        tmp[2] = 52
                        tmp[1] = 168
                        tmp[0] = 83
                    #fn-ORANGE
                    else:
                        tmp[2] = 229
                        tmp[1] = 156
                        tmp[0] = 30
            scorePanel[bts] = np.transpose(tmp, (1,2,0))

        hList = []
        for bts in range(self.opt.batchsize):
            position = (3, 20)
            cv2.putText(
                scorePanel[bts],
                f'{ab_RGB[bts + self.batchNum*self.opt.batchsize]: .4f}',
                position, #position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                0.8, #font size
                (255 ,255, 255, 0), #font color BGR
                1)
            # cv2.putText(
            #     scorePanel[bts],
            #     f'{ab_RGB[bts + self.batchNum*self.opt.batchsize]: .4f}',
            #     position,
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.8,
            #     (0 ,0, 0, 0),
            #     1)

            position = (15, 45)
            if predict:
                cv2.putText(
                    scorePanel[bts],
                    f'{self.diagnois_result}',
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255 ,255, 255, 255),
                    1)
                # cv2.putText(
                #     scorePanel[bts],
                #     f'check2',
                #     position,
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.8,
                #     (0 ,0, 0, 0),
                #     1)
            
            else:
                cv2.putText(
                    scorePanel[bts],
                    f'{self.gt_labels[bts + self.batchNum*self.opt.batchsize]}',
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255 ,255, 255, 255),
                    3)
                cv2.putText(
                    scorePanel[bts],
                    f'{self.gt_labels[bts + self.batchNum*self.opt.batchsize]}',
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0 ,0, 0, 0),
                    1)
            hList.append(scorePanel[bts])
        self.batchNum += 1


        scsc = tuple(hList)
        nnewImg = np.hstack(scsc)
        newImg = np.vstack((addImg, nnewImg)) #(448,128,3)
        # pad = np.zeros(shape=(720, 128, 3))
        pad = np.zeros(shape=(3, 128, 720))
        pad[:,:128,:448] = np.transpose(newImg, (2,1,0))
        pad = np.transpose(pad, (2,1,0))
        
        newImg = np.hstack((raw_img, pad))
        return newImg
        
    def load_weight(self):
        path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
        pretrained_dict = torch.load(path)['state_dict']
        directory = f'./output/{self.name.lower()}/{self.opt.dataset}/test/images/fake'
        try:
            self.netg.load_state_dict(pretrained_dict)
            if not os.path.exists(directory):
                os.makedirs(directory)

        except IOError:
            raise IOError("netG weights not found")
        print('   Loaded weights.')
##
class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, opt, dataloader):
        super(Ganomaly, self).__init__(opt, dataloader)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        self.l_adv = l2_loss
        #self.l_adv = l1_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        # self.l_enc = l1_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)

    ##
    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

    ##
    def backward_g(self):
        """ Backpropagate through netG
        """
        self.err_g_adv = self.l_adv(self.netd(self.input)[1], self.netd(self.fake)[1]) # l1_loss
        self.err_g_con = self.l_con(self.fake, self.input)  # nn.L1Loss()
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)   # l1_loss
        self.err_g = self.err_g_adv * self.opt.w_adv + \
                     self.err_g_con * self.opt.w_con + \
                     self.err_g_enc * self.opt.w_enc
        self.err_g.backward(retain_graph=True)

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()

    ##
    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        print('   Reloading net d')
        self.netd.apply(weights_init)

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        self.forward_d()

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # netd
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d.item() < 1e-5: self.reinit_d()
    
    def predict(self,input,_path):
        """[summary]
        Args:
            input ([type]): input image
        """
        self.load_weight = True
      

        self.opt.phase = 'test'
        self.times = []
        self.total_steps = 0
        epoch_iter = 0
        self.batchNum = 0
        ab_RGB = []
        is_abList = []
        stT = time.time()
        ab_scores_dataSet = []
        time_i = time.time()


        self.set_input((input,torch.ones(1)))
        self.fake, latent_i, latent_o = self.netg(self.input)  ## 여기서 self.fake  갱신
        time_o = time.time()

        save_path = os.path.dirname(_path)
        save_path = os.path.join(save_path,'result')
        save_path = os.path.join(save_path,os.path.basename(_path))

        real, fake, _ = self.get_current_images() # return real, fake, fixed
        self.visualizer.save_current_images(self.epoch, real, fake, _)

        real_img = real.cpu().data.numpy().squeeze()
        generate_img= fake.cpu().data.numpy().squeeze()
        diff_img, ch3_diff_img = heatMap.calc_diff(real_img, generate_img, 1, params.filter_thres)
        anomaly_img = np.zeros(shape=(self.opt.isize, self.opt.isize, self.opt.nc))
        anomaly_img = heatMap.Draw_Anomaly_image(real_img, diff_img, ch3_diff_img, self.opt.batchsize)
        raw_img, new_diff_img, is_abnormal = heatMap.DrawResult(diff_img, save_path, _path, params=None)
        ab_RGB.append(np.sum(new_diff_img) / (self.opt.isize*self.opt.isize*255)) ## ab_RGB를 Draw_Result 하고 계산해야함!
        is_abList.append(is_abnormal)
        

        if raw_img  is None: # 원 검출을 실패했을 경우.
            return None,  2

        real = cv2.normalize(real_img, real_img, 0, 255, cv2.NORM_MINMAX)
                
        #make_pannel 이전에 적용시킬것
        real_img = self.BGR2RGB(real_img)
        generate_img = cv2.normalize(generate_img, generate_img, 0, 255, cv2.NORM_MINMAX)
        generate_img = self.BGR2RGB(generate_img)
        
        anomaly_img = self.BGR2RGB(anomaly_img)
        newImg = self.make_result_panel(raw_img, real_img, generate_img, anomaly_img, ab_RGB, params.ab_thres, is_abnormal, params.use_abscore,True)
        cv2.imwrite(save_path, newImg)
        print('save_ result')
        result = 0 if self.diagnois_result == 'Normal' else  1
        return save_path, result
