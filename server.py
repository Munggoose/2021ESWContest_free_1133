import socket
from os import walk
import os
import time
from PIL import Image
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
from preprocessing import get_preprocess_img #여기 수정해야하는 부분
import cv2
from lib.model import Ganomaly
from options import Options


class Server:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.serv_addr = (self.host, self.port)
        self.conn_sock = None
        self.model = None
        self.dataloader = None
        self.sock = None

    #sever 시작
    def establish(self):

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(self.serv_addr) # ip:self.host , port: self.port 
        self.sock.listen(5)
        print('[server]server start')
        self.conn_sock, _ = self.sock.accept()
        print('[Server]server & client Connect')

    #모델의 가중치 및 하이퍼파라미터 값 로드
    def load_model(self):
        """[summary] 
        model initialize
        setting model pth 
        """
        self.opt = Options().parse()
        self.opt.batchsize = 1
        self.opt.load_weight = True
        self.model = Ganomaly(self.opt, None)
        self.model.load_weight()
        print('[server]default model ' + 'ganomaly' +' is ready')

    def preprocess_image(self, path):
        org_img = cv2.imread(path)
        target_img, preprocess_check = get_preprocess_img(org_img)
        target_img = Image.fromarray(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
        img_t = transforms.Resize(size=(self.opt.isize, self.opt.isize))(target_img) #opt.isize
        img_t = transforms.ToTensor()(img_t)
        norm = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 
        img_t = norm(img_t)
        img_t = torch.unsqueeze(img_t,0)
        img_t = img_t.cuda()
        return img_t,preprocess_check

    def diagnoisis_volt(self, path):
        input_img, no_err = self.preprocess_image(path)
        if not no_err:
            return None, 2  # 2는 전처리 과정에서 원 못찾는 것

        save_path, result = self.model.predict(input_img,path)
        return save_path, result

    #연결된 소켓으로 클라이언트에 메세지 전달
    def send_msg(self, msg):
        msg = msg.encode()
        self.conn_sock.sendall(msg)

    #클라이언트로 부터 메세지 수령
    def recv_msg(self, size = 1024):
        msg = self.conn_sock.recv(size)
        if not msg:
            self.conn_sock.close()
            exit()
        return msg.decode()

    def disconnet(self):
        self.conn_sock.close()
        self.sock.close()

    def server_activate(self):
        self.load_model()
        test_sample_path = './init_sample.bmp'
        _,_ = self.diagnoisis_volt(test_sample_path)
        self.establish()
        self.send_msg('server is ready')
        
        while True:
            print('Waiting for transmission....')
            img_path = self.recv_msg()

            print(f'from client: {img_path}\n')
            # msg_data = img_path.split('?')

            # img_path = msg_data[1]
            print('[server]img path(or quit) is : ' + img_path)
            if not img_path:
                print('NO Path')
                continue

            if img_path == 'finish':
                self.disconnet()
                break
            
            start = time.process_time()
            save_path, result = self.diagnoisis_volt(img_path)
            if save_path is None:
                continue

            print(f'result_code {result}\n result_path {save_path}')
            end = time.process_time()
            result_msg = f'{result}?{save_path}'
            self.send_msg(result_msg)
            print(result_msg)
        
        self.disconnect()



if __name__ =='__main__':

    host = '127.0.0.1'
    port = 3070
    S = Server(host, port)
    S.server_activate()