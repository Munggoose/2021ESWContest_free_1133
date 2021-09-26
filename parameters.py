from numpy import array

imsize = 128
PREFIX_RAW = '.bmp'
PREFIX_SAV = '.bmp'
raw_PATH = './RAW'
raw_PATH = 'C:/knv/Projs/2021-1/KnV/0.MAIN/GANomaly_Anomaly_Detection/RAW'


#True:  abnormal score으로 정상/비정상 판별
#False: cv2.labeling 을 활용한 바운딩 조건으로 정상/비정상 판별
use_abscore = False
ab_thres = 0.00156
filter_thres = 48.02
'''
secN은 N-section에 대한 기준.
cond_secN = array([
    (조건1):[ (하한):[area, brightness], (상한):[area, brightness] ],
    (조건2):[ (하한):[area, brightness], (상한):[area, brightness] ],
    ...
])
'''
cond_sec1 = array([
    [[10, 30],[700, 100]],
    [[10, 30],[700, 100]]
])
cond_sec2 = array([
    [[10, 30],[100, 100]],
    [[10, 30],[100, 100]]
])