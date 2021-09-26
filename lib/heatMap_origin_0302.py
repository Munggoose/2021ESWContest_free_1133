import cv2
import numpy as np
import argparse
from lib.Hough import img_Contrast

import seaborn as sns
import matplotlib.pylab as plt

from skimage import transform

def add(image, heat_map, alpha=0, display=True, save=None, cmap='OrRd', axis='on', verbose=False):

    # normalize heat map
    max_value = np.max(heat_map)
    min_value = np.min(heat_map)
    normalized_heat_map = (heat_map - min_value) / (max_value - min_value)

    # display
    plt.imshow(image)
    plt.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
    plt.axis(axis)

    if display:
        plt.show()

    if save is not None:
        if verbose:
            print('save image: ' + save)
        plt.savefig(save, bbox_inches='tight', pad_inches=0)

# def create_heatmap(im_map, im_cloud, kernel_size=(5,5),colormap=cv2.COLORMAP_JET,a1=0.5,a2=0.5):
def create_heatmap(im_map, im_cloud, colormap=cv2.COLORMAP_HOT, a1=0.5, a2=0.5):
    im_cloud_clr = cv2.applyColorMap(im_cloud, colormap)
    im_map = im_map + im_cloud_clr
    
    return im_map

def Fig2Arr(fig):
    """[summary] plt.figure를 RGBA로 변환 (4-CH), shape=(h, w, l)

    Args:
        fig ([type]): [description]
    """
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

def create_heatmap2(im_map, im_cloud):
    plt.clf()
    ax = sns.heatmap(im_cloud, cmap="OrRd")
    ax = Fig2Arr(ax)
    cv2.imshow('hi',ax)
    cv2.waitKey(0)
    
def create_heatmap3(im_map, im_cloud):
    im_map = np.transpose(im_map, (1,2,0))
    im_map = cv2.normalize(im_map, im_map, 0, 255, cv2.NORM_MINMAX)
    im_cloud = np.transpose(im_cloud, (1,2,0))
    im_cloud = cv2.normalize(im_cloud, im_cloud, 0, 255, cv2.NORM_MINMAX)
    # print(f'1Avg: {np.average(im_map)}')
    # print(f'1Min: {np.min(im_map)}')
    # print(f'1Max: {np.max(im_map)}\n')
    add(im_map, im_cloud)


def calc_diff(real_img, generated_img, batchsize, thres=44.02): # 0.86 for 0221 8777777777777 (0.79 for 0.9699) (0.86 for 0.967)
    """[summary]

    Args:
        real_img ([type]): [description]        shape = (3, 128, 128)
        generated_img ([type]): [description]   shape = (3, 128, 128)
        batchsize ([type]): [description]
        thres (float, optional): [description]  차영상의 한 픽셀의 차이가 thres보다 작을 때 0으로 만듦.

    Returns:
        [type]: [description]
    """
    
    diff_img = real_img - generated_img

    ch3_diff_img = diff_img
    
    # if np.max(diff_img) < 253 or True:
    #     print(f'avg: {np.average(diff_img)}')
    #     print(f'min: {np.min(diff_img)}')
    #     print(f'max: {np.max(diff_img)}\n')
    
    # np.sum을 하여 R,G,B의 종합적인 차이를 구함.
    if batchsize == 1:
        diff_img = np.sum(diff_img, axis=0)
    else:
        diff_img = np.sum(diff_img, axis=1)
    diff_img = np.abs(diff_img)
        
    diff_img *= 51
    
    # hist = plt.hist(diff_img)
    # plt.show()
    if batchsize == 1:
        diff_img[diff_img < thres] = 0.0
    else:
        for bts in diff_img:
            bts[bts <= thres] = 0.0
    
    
    return diff_img, ch3_diff_img


def Draw_Anomaly_image(real_img, diff_img, ch3_diff_img, batchsize):
    """[summary] calc_diff 로부터 구한 diff_img를 128x128에서 1280x720의 RAW Image에 적용.

    Args:
        real_img ([type]): [description]
        diff_img ([type]): [description]
        ch3_diff_img ([type]): [description]
        batchsize ([type]): [description]

    Returns:
        [type]: [description]
    """
    anomaly_img = real_img - ch3_diff_img 
    anomaly_img = cv2.normalize(anomaly_img, anomaly_img, 0, 255, cv2.NORM_MINMAX)

    # diff_img = cv2.normalize(diff_img, diff_img, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)#########################
    diff_img = diff_img.astype(np.uint8)

    diff_img_expanded = [diff_img, diff_img, diff_img]
    
    if batchsize == 1:
        anomaly_img = np.transpose(create_heatmap(np.transpose(anomaly_img, (1,2,0)), np.transpose(diff_img_expanded, (1,2,0)), a1=.2, a2=.8), (2,0,1))
    else:
        diff_img_expanded = np.transpose(diff_img_expanded, (1,0,3,2))
        for bts in range(batchsize):
            anomaly_img[bts] = np.transpose(create_heatmap(np.transpose(anomaly_img[bts], (1,2,0)), np.transpose(diff_img_expanded[bts], (1,2,0)), a1=.2, a2=.8), (2,1,0))
    
    return anomaly_img


def DrawResult(raw_img, diff_img, sav_fName, rawPATH, param1=30, param2=100, weight1=1, weight2=1):
        """[summary]: find_Center()를 활용해 raw Image의 중심점 찾고, 얻은 좌표 기반으로 diff_img 덧붙임.
        
        Related Functions:
            img_Contrast():raw_img의 이미지 대비 증가
            find_Center(): 입력받은 이미지에서 작은 원의 x,y좌표, r(반지름 반환)
        Args:
            raw_img ([type]): [description] shape=(720, 1280, 3)
            diff_img ([type]): [description] maybe shape=(3, w, h)
            param1: Threshold for average of brightness
            param2: Threshold for area
        """
        if sav_fName[0] == '0':
            rawPATH += 'normal\\' + sav_fName
        else:
            rawPATH += 'abnormal\\' + sav_fName

        raw_img = cv2.imread(rawPATH, cv2.IMREAD_COLOR)
        
        xBias = 510
        yBias = 220

        raw_img = img_Contrast(raw_img)
        imSize = 270
        img = raw_img[yBias:yBias+imSize, xBias:xBias+imSize]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        gray_blurred = cv2.blur(gray, (3, 3)) 
        detected_circles = cv2.HoughCircles(gray_blurred,  
                           cv2.HOUGH_GRADIENT, 1.5, 1270, param1 = 95, 
                       param2 = 30, minRadius = 98, maxRadius = 102) 
        #Draw circles that are detected. 
        if detected_circles is not None: 
            raw_img = np.transpose(raw_img, (2,0,1))
            # Convert the circle parameters a, b and r to integers. 
            detected_circles = np.uint16(np.around(detected_circles)) 
            for pt in detected_circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2] 

                a = a + xBias
                b = b + yBias
                r += 204

                #검은 배경의 raw_diff 생성 후 위에서 얻은 a,b 좌표를 기준으로 하여 raw_img에 적용할 diff_img를 upsampling 하여 더함.
                #순서: raw_diff Labeling -> Filtering -> diff 갱신. (param1, param2 에 따라 ab_score 달라지기 때문에 diff_img에도 반영해야함.)
                raw_diff_1ch = np.zeros(shape=(720,1280))
                diff_img = cv2.resize(diff_img, dsize=(2*r, 2*r), interpolation=cv2.INTER_LINEAR)
                raw_diff_1ch[b-r:b+r, a-r:a+r] = diff_img
                
                raw_diff = np.array([raw_diff_1ch, raw_diff_1ch, raw_diff_1ch]).astype(np.uint8)
                
                raw_img = np.where(raw_diff == 0, raw_img, 0)
                raw_img = create_heatmap(np.transpose(raw_img, (1,2,0)), np.transpose(raw_diff, (1,2,0)), cv2.COLORMAP_HOT) #-> (720, 1280, 3)
                
                #cv2 Labeling
                cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_diff[0], connectivity=8)
                print(centroids)
                exit()
                for i, (x, y, w, h, area) in enumerate(stats):
                    brightness = np.sum(np.where(labels==i, raw_diff[0], 0)) / area
                    if brightness > param1 and area > param2:
                        cv2.rectangle(raw_img, (x, y, w, h), (0,255,255))
                        cv2.putText(raw_img, f'{brightness: .0f}({area: .0f})', (x+w, y+h), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
                    else:
                        raw_diff_1ch = np.where(labels!=i, raw_diff_1ch, 0)
                
                # 1, 2구간 가중치 차등화
                draw_paper = np.zeros(shape=(720,1280), dtype="uint8")
                draw_paper2 = np.zeros(shape=(720,1280), dtype="uint8")
                SECTION1_mask = cv2.circle(draw_paper, (a,b), r-204+150, 255, -1)
                SECTION2_mask = cv2.circle(draw_paper2, (a,b), r, 255, -1)
                SECTION2_mask = cv2.bitwise_xor(SECTION1_mask, SECTION2_mask)
                
                raw_diff_1ch = np.where(SECTION1_mask!=0, raw_diff_1ch, raw_diff_1ch*weight1)
                raw_diff_1ch = np.where(SECTION2_mask!=0, raw_diff_1ch, raw_diff_1ch*weight2)
                
                
                raw_img = raw_img[:, int(a-720/2):int(a+720/2), :]
                return raw_img, raw_diff_1ch
        else:
            return None