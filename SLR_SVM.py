#! python2
#coding=gbk
from __future__ import division
'''
Created on 2017.9.26

@author: Zhao
'''



import warnings
warnings.filterwarnings("ignore")
from numpy import *
from sys import exit
from myo import init, Hub, Feed, StreamEmg
from myo.lowlevel import stream_emg
import time
from sklearn import preprocessing
from sklearn.externals import joblib


import serial
from time import sleep
import sys
 
warnings.filterwarnings("ignore")
'''*****************'''


online_EMG_feat = array([0])
online_ACC_feat = array([0])
online_GYR_feat = array([0])
scale = loadtxt("scale.txt")


Gesture_Table = ['< You >', '< Me >', '< Think >', '< Like >', '< Eat >' ,'< What >','< Have >',\
'< No >', '< Good >', '< Meat >', '< Egg >', '  ']



class Myos(object):

    def __init__(self, myo_device):
        self._myo_device = myo_device
        self._time = 0
        self._t_s = 0
        self.sign = False
        self.Emg = [];
        self.Acc = [];
        self.Gyr = [];
        
    def start(self, t_s, isSign = False):
        global online_EMG_feat
        global online_ACC_feat
        global online_GYR_feat
        self.sign = isSign
        self._t_s = 1 / t_s
        startTime = time.time()
        while(1):
            currentTime = time.time()
            if (currentTime - startTime) > self._t_s:
                startTime = time.time()
                self.Emg = self._myo_device[0].emg + self._myo_device[1].emg
                self.Acc = [it for it in self._myo_device[0].acceleration] +[it for it in self._myo_device[1].acceleration]
                self.Gyr = [it for it in self._myo_device[0].gyroscope] +[it for it in self._myo_device[1].gyroscope]
                online_EMG_feat, online_ACC_feat,online_GYR_feat= show_output('acceleration',self.Acc, online_EMG_feat,online_ACC_feat,online_GYR_feat)
                online_EMG_feat, online_ACC_feat,online_GYR_feat= show_output('gyroscope',self.Gyr, online_EMG_feat,online_ACC_feat,online_GYR_feat)
                online_EMG_feat, online_ACC_feat,online_GYR_feat = show_output( 'emg',self.Emg, online_EMG_feat,online_ACC_feat,online_GYR_feat)

def ARC3ord(Orin_Array):
    t_value = len(Orin_Array)
    AR_coeffs = polyfit(range(t_value),Orin_Array,3)
    return AR_coeffs


def online_fea_extraction(online_feat,window_size,gesture_size,width_data):
    split_size = gesture_size / window_size
    window_samples = vsplit(online_feat, split_size)
    window_index = 0           
    for window_piece in window_samples:                
        RMS_feat = mean(sqrt(square(window_piece)), axis=0)
        piece_move1 = vstack((window_piece[1::,:], zeros((1,width_data))))
        ZC_feat = sum((-sign(window_piece) * sign(piece_move1)+1)/2,axis = 0)
        ARC_feat = apply_along_axis(ARC3ord, 0, window_piece)
        if window_index == 0:
            RMS_feat_level = RMS_feat
            ZC_feat_level = ZC_feat
            ARC_feat_level = ARC_feat
        if window_index > 0:
            RMS_feat_level = vstack((RMS_feat_level,RMS_feat))
            ZC_feat_level = vstack((ZC_feat_level,ZC_feat))
            ARC_feat_level = vstack((ARC_feat_level,ARC_feat))
        window_index += 1
    temp_gest_sample_feat = vstack((RMS_feat_level, ZC_feat_level, ARC_feat_level))
    get_gest_feat = (temp_gest_sample_feat.T).ravel()
    return get_gest_feat   


def show_output( flag ,data, online_feat_level1,online_feat_level2,online_feat_level3):
    gesture_size = 160
    window_size = 16
    global scale
    global Gesture_Table
    
    stepdata_online = array(data)
    if flag=='emg':
        if online_feat_level1.any() == 0:
            online_feat_level1 = stepdata_online
        else:
            online_feat_level1 = vstack((online_feat_level1, stepdata_online))
    if flag=='acceleration':
        if online_feat_level2.any() == 0:
            online_feat_level2 = stepdata_online  
        else:
            online_feat_level2 = vstack((online_feat_level2, stepdata_online))
    if flag=='gyroscope':
        if online_feat_level3.any() == 0:
            online_feat_level3 = stepdata_online  
        else:
            online_feat_level3 = vstack((online_feat_level3, stepdata_online))
    if len(online_feat_level1)>=gesture_size:
        if len(online_feat_level2)>=gesture_size: 
            if len(online_feat_level3)>=gesture_size:
                EMG_ges_fea = online_fea_extraction(online_feat_level1[0:gesture_size,8:16],window_size, gesture_size,8)
                ACC_ges_fea = online_fea_extraction(online_feat_level2[0:gesture_size,3:6],window_size, gesture_size,3)
                GYR_ges_fea = online_fea_extraction(online_feat_level3[0:gesture_size,3:6],window_size, gesture_size,3)
                Combined_fea_list = hstack((EMG_ges_fea,ACC_ges_fea,GYR_ges_fea))
                max_abs_scaler = preprocessing.MaxAbsScaler()
                max_abs_scaler.scale_ = scale
                norm_online_feat = max_abs_scaler.transform(Combined_fea_list)
                clf = joblib.load("train_model.m")
                Table_Index = int(clf.predict([norm_online_feat]))
                print(Gesture_Table[Table_Index],'\n')
                
                online_feat_level1 = array([0])
                online_feat_level2 = array([0])
                online_feat_level3 = array([0])             
    return online_feat_level1,online_feat_level2,online_feat_level3

       
def main():
    init()
    feed = Feed()
    hub = Hub()
    times = 0
    hub.run(1000, feed)
    try:
        myo_device = feed.get_devices()
        print(myo_device)
        time.sleep(1)
        myo_device[0].set_stream_emg(StreamEmg.enabled)
        myo_device[1].set_stream_emg(StreamEmg.enabled)
        time.sleep(0.5)
        twoMyos = Myos(myo_device)
        twoMyos.start(100,70)
        
    except KeyboardInterrupt:
        print("Quitting ...")
        
    finally:
        hub.shutdown()

if __name__ == '__main__':
    main()
    