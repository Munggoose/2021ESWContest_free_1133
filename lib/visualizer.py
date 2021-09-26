""" This file contains Visualizer class based on Facebook's visdom.

Returns:
    Visualizer(): Visualizer class to display plots and images
"""

##
import os
import time
import numpy as np
import torchvision.utils as vutils

import cv2
from PIL import Image
import matplotlib.pyplot as plt

##
class Visualizer():
    """ Visualizer wrapper based on Visdom.

    Returns:
        Visualizer: Class file.
    """
    # pylint: disable=too-many-instance-attributes
    # Reasonable.

    ##
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.win_size = 256
        self.name = opt.name
        self.opt = opt
        if self.opt.display:
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port)

        # --
        # Dictionaries for plotting data and results.
        self.plot_data = None
        self.plot_res = None

        # --
        # Path to train and test directories.
        self.img_dir = os.path.join(opt.outf, opt.name, 'train', 'images')
        self.tst_img_dir = os.path.join(opt.outf, opt.name, 'test', 'images')
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.tst_img_dir):
            os.makedirs(self.tst_img_dir)
        # --
        # Log file.
        self.log_name = os.path.join(opt.outf, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    ##
    @staticmethod
    def normalize(inp):
        """Normalize the tensor

        Args:
            inp ([FloatTensor]): Input tensor

        Returns:
            [FloatTensor]: Normalized tensor.
        """
        return (inp - inp.min()) / (inp.max() - inp.min() + 1e-5)

    ##
    def plot_current_errors(self, epoch, counter_ratio, errors):
        """Plot current errros.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            errors (OrderedDict): Error for the current epoch.
        """

        if not hasattr(self, 'plot_data') or self.plot_data is None:
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss'
            },
            win=4
        )

    ##
    def plot_performance(self, epoch, counter_ratio, performance):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            performance (OrderedDict): Performance for the current epoch.
        """
        if not hasattr(self, 'plot_res') or self.plot_res is None:
            self.plot_res = {'X': [], 'Y': [], 'legend': list(performance.keys())}
        self.plot_res['X'].append(epoch + counter_ratio)
        self.plot_res['Y'].append([performance[k] for k in self.plot_res['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_res['X'])] * len(self.plot_res['legend']), 1),
            Y=np.array(self.plot_res['Y']),
            opts={
                'title': self.name + 'Performance Metrics',
                'legend': self.plot_res['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Stats'
            },
            win=5
        )

    ##
    def print_current_errors(self, epoch, errors):
        """ Print current errors.

        Args:
            epoch (int): Current epoch.
            errors (OrderedDict): Error for the current epoch.
            batch_i (int): Current batch
            batch_n (int): Total Number of batches.
        """
        # message = '   [%d/%d] ' % (epoch, self.opt.niter)
        message = '   Loss: [%d/%d] ' % (epoch, self.opt.niter)
        for key, val in errors.items():
            message += '%s: %.5f ' % (key, val)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    ##
    def print_current_performance(self, performance, best):
        """ Print current performance results.

        Args:
            performance ([OrderedDict]): Performance of the model
            best ([int]): Best performance.
        """
        message = '   '
        for key, val in performance.items():
            message += '%s: %.3f ' % (key, val)
        message += 'max ' + self.opt.metric + ': %.3f' % best

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def display_current_images(self, reals, fakes, fixed):
        """ Display current images.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        # reals = self.normalize(reals.cpu().numpy())
        # fakes = self.normalize(fakes.cpu().numpy())
        # fixed = self.normalize(fixed.cpu().numpy())

        # self.vis.images(reals, win=1, opts={'title': 'Reals'})
        # self.vis.images(fakes, win=2, opts={'title': 'Fakes'})
        # self.vis.images(fixed, win=3, opts={'title': 'Fixed'})

    def save_current_images(self, epoch, reals, fakes, fixed):
        """ Save images for epoch i.

        Args:
            epoch ([int])        : Current epoch
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        #print(f'saved images to {self.img_dir}')
        vutils.save_image(reals, '%s/reals.png' % self.img_dir, normalize=True)
        vutils.save_image(fakes, '%s/fakes.png' % self.img_dir, normalize=True)
        vutils.save_image(fixed, '%s/fixed_fakes_%03d.png' %(self.img_dir, epoch+1), normalize=True)


        
        # real_img = np.transpose(reals.cpu().data.numpy().squeeze(), (1, 2, 0))
        # real_img = real_img.reshape(self.opt.isize, self.opt.isize, self.opt.nc)
        # generated_img = np.transpose(fakes.cpu().data.numpy().squeeze(), (1, 2, 0))
        # generated_img = generated_img.reshape(self.opt.isize, self.opt.isize, self.opt.nc)
        real_img = reals.cpu().data.numpy().squeeze()
        generated_img = fakes.cpu().data.numpy().squeeze()
        fixed_img = fixed.cpu().data.numpy().squeeze()

        # real_img = reals.cpu().data.numpy()
        # generated_img = fakes.cpu().data.numpy()
        # fixed_img = fixed.cpu().data.numpy()

        '''
        negative = np.zeros_like(real_img)

        # if not reverse:
        diff_img = real_img - generated_img
        # else:
        # diff_img = generated_img - real_img
        thres = 0.01
        diff_img[diff_img <= thres] = 0
        diff_img[diff_img > thres] = 1

        anomaly_img = [np.zeros(shape=(self.opt.isize, self.opt.isize, self.opt.nc)), np.zeros(shape=(self.opt.isize, self.opt.isize, self.opt.nc)), np.zeros(shape=(self.opt.isize, self.opt.isize, self.opt.nc))]
        
        # anomaly_img[0] = (real_img - diff_img) * 255
        anomaly_img[0] = (real_img -diff_img + 1) / 2 * 255
        anomaly_img[1] = (real_img - diff_img) * 255
        anomaly_img[2] = (real_img - diff_img) * 255
        #anomaly_img[0] = anomaly_img[0] + diff_img

        anomaly_img = [anomaly_img[0].astype(np.uint8), anomaly_img[1].astype(np.uint8), anomaly_img[2].astype(np.uint8)]
        # plt.imshow(np.transpose(anomaly_img[0], (1, 2, 0)) )
        # plt.imshow(np.transpose(diff_img, (1, 2, 0)) )
        # plt.show()
        cv2.imwrite(f'output\\ganomaly\\casting\\train\\images\\diff_img_{epoch+1}.png', np.transpose(anomaly_img[0], (1, 2, 0)))
        '''
        #########################
        negative = np.zeros_like(real_img)

        # if not reverse:
        diff_img = real_img - fixed_img
        # else:
            # diff_img = generated_img - real_img
        
        thres = 0.05
        diff_img[diff_img <= thres] = 0
        diff_img[diff_img > thres] = 1

        aanomaly_img = [np.zeros(shape=(self.opt.isize, self.opt.isize, self.opt.nc)), np.zeros(shape=(self.opt.isize, self.opt.isize, self.opt.nc)), np.zeros(shape=(self.opt.isize, self.opt.isize, self.opt.nc))]
        
        aanomaly_img[0] = (real_img -diff_img + 1) / 2 * 255
        aanomaly_img[1] = (real_img - diff_img) * 255
        aanomaly_img[2] = (real_img - diff_img) * 255
        aanomaly_img[0] = aanomaly_img[0] + diff_img

        aanomaly_img = [aanomaly_img[0].astype(np.uint8), aanomaly_img[1].astype(np.uint8), aanomaly_img[2].astype(np.uint8)]
        # plt.imshow(np.transpose(anomaly_img[0], (1, 2, 0)) )
        # plt.imshow(np.transpose(diff_img, (1, 2, 0)) )
        # plt.show()
        #cv2.imwrite(f'output\\ganomaly\\casting\\train\\images\\fixed_diff_img_{epoch+1}.png', np.transpose(aanomaly_img[0], (1, 2, 0)))

        
        # anomaly_img = np.array(anomaly_img)

        # vutils.save_image(anomaly_img[0], '%s/fake_%03d_abnormality.png' % (self.img_dir, epoch+1), normalize=True)
        # cv2.imshow('hi', anomaly_img[0])
        # cv2.waitKey(0)

        # myImg = Image.fromarray(anomaly_img)
        # cv2.imwrite("output\\ganomaly\\mnist\\train\\images\\" + str(epoch+1) +'.png', anomaly_img.reshape())
        # print(f'\t{myImg.shape()}')
        # myImg.save('my.png')
        # myImg.show()

        # fig, plots = plt.subplots(1, 4)

        # fig.suptitle(f'Anomaly - (anomaly score:)')

        # fig.set_figwidth(20)
        # fig.set_tight_layout(True)
        # plots = plots.reshape(-1)
        # plots[0].imshow(np.transpose(real_img, (1, 2, 0)), cmap='bone', label='real')
        # plots[1].imshow(np.transpose(generated_img, (1, 2, 0)), cmap='bone')
        # plots[2].imshow(np.transpose(diff_img, (1, 2, 0)), cmap='bone')
        # plots[3].imshow(np.transpose(anomaly_img[0], (1, 2, 0)), cmap='bone')


        # plots[0].set_title('real')
        # plots[1].set_title('generated')
        # plots[2].set_title('difference')
        # plots[3].set_title('Anomaly Detection')
        # plt.show()
