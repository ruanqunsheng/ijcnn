#!/usr/bin/env python
#-*- coding: utf-8 -*-
#@file: pre_process.py
#@author: Yilong Yang
#@time: 2017/12/21 15:58
########################################################
import scipy.io as sio
import argparse
import os
import numpy as np
import pandas as pd
import time
import pickle

np.random.seed(0)

def get_args():
    parser = argparse.ArgumentParser()
    hpstr = "set dataset directory"
    parser.add_argument('-d', '--directory', default="data_preprocessed_matlab/", nargs='*', type=str, help=hpstr)
    hpstr = "set window size"
    parser.add_argument('-w', '--window', default=128, nargs='*', type=int, help=hpstr)
    hpstr = "set output directory"
    parser.add_argument('-o', '--output_dir', default="./deap_shuffled_data/origin_arousal/", nargs='*', help=hpstr)
    args = parser.parse_args()
    return (args)

def print_top(dataset_dir, window_size, convert, parallel, segment, begin_subject, end_subject, output_dir, set_store):
    print("######################## SEED EEG data preprocess ########################	\
		   \n### Author: Yilong Yang   XMU, Xiamen	email: elonyang.xmu@gmail.com ###	\
		   \n# input directory:	%s \
		   \n# window size:		%d 	\
		   \n# output directory:	%s	\
		   \n##############################################################################" % \
          (dataset_dir,window_size,output_dir))
    return None

def data_1Dto2D(data, Y=9, X=9):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0,  	   	0, 	        0,          data[0],    0,          data[16], 	0,  	    0, 	        0       )
    data_2D[1] = (0,  	   	0,          0,          data[1],    0,          data[17],   0,          0,          0       )
    data_2D[2] = (data[3],  0,          data[2],    0,          data[18],   0,          data[19],   0,          data[20])
    data_2D[3] = (0,        data[4],    0,          data[5],    0,          data[22],   0,          data[21],   0       )
    data_2D[4] = (data[7],  0,          data[6],    0,          data[23],   0,          data[24],   0,          data[25])
    data_2D[5] = (0,        data[8],    0,          data[9],    0,          data[27],   0,          data[26],   0       )
    data_2D[6] = (data[11], 0,          data[10],   0,          data[15],   0,          data[28],   0,          data[29])
    data_2D[7] = (0,        0,          0,          data[12],   0,          data[30],   0,          0,          0       )
    data_2D[8] = (0,        0,          0,          data[13],   data[14],   data[31],   0,          0,          0       )
    # return shape:9*9
    return data_2D

def norm_dataset(dataset_1D):
    print("---------------------------------------------------------------------------------------------------------------")
    print("dataset-1D:",dataset_1D.shape)
    norm_dataset_1D = np.zeros([dataset_1D.shape[0], 32])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
    # return shape: m*32
    return norm_dataset_1D

def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data. nonzero ()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
    # return shape: 9*9
    return data_normalized

def dataset_1Dto2D(dataset_1D):
    dataset_2D = np.zeros([dataset_1D.shape[0],9,9])
    for i in range(dataset_1D.shape[0]):
        dataset_2D[i] = data_1Dto2D(dataset_1D[i])
    # return shape: m*9*9
    return dataset_2D

def norm_dataset_1Dto2D(dataset_1D):
    norm_dataset_2D = np.zeros([dataset_1D.shape[0], 9, 9])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_2D[i] = feature_normalize( data_1Dto2D(dataset_1D[i]))
    # return shape: m*9*9
    return norm_dataset_2D

def windows(data, size):
    start = 0
    while ((start+size) < data.shape[0]):
        yield int(start), int(start + size)
        start += size

def segment_signal_without_transition(data,label,label_index,window_size):
    print(" label index is:", label_index)  # get data file name and label file name
    print("data shape:", data.shape)
    for (start, end) in windows(data, window_size):
        # print(data.shape)
        if((len(data[start:end]) == window_size)):
            if(start == 0):
                segments = data[start:end]
                segments = np.vstack([segments, data[start:end]])

                labels = np.array(label[label_index])
                labels = np.append(labels, np.array(label[label_index]))
                print("labels:",labels)
            else:
                segments = np.vstack([segments, data[start:end]])
                labels = np.append(labels, np.array(label[label_index])) # labels = np.append(labels, stats.mode(label[start:end])[0][0])
    print(set(labels))
    return segments, labels

def apply_mixup(dataset_file,window_size): # initial empty label arrays
    data_file_in = sio.loadmat(dataset_file)
    data_in = data_file_in["data"].transpose(0,2,1)
    #0 valence, 1 arousal, 2 dominance, 3 liking
    label_in= data_file_in["labels"][:,1]>5
    label_inter	= np.empty([0]) # initial empty data arrays
    data_inter_cnn	= np.empty([0,window_size, 9, 9])
    data_inter_rnn	= np.empty([0, window_size, 32])
    trials = data_in.shape[0]

    # Data pre-processing
    for trial in range(0,trials):
        base_signal = (data_in[trial,0:128,0:32]+data_in[trial,128:256,0:32]+data_in[trial,256:384,0:32])/3
        data = data_in[trial,384:8064,0:32]
        # compute the deviation between baseline signals and experimental signals
        for i in range(0,60):
            data[i*128:(i+1)*128,0:32]=data[i*128:(i+1)*128,0:32]-base_signal
        label_index = trial
        print("trial:",trial)
        #read data and label
        print("data shape: ",data.shape)
        data = norm_dataset(data)
        data, label = segment_signal_without_transition(data, label_in,label_index,window_size)
        # cnn data process
        data_cnn    = dataset_1Dto2D(data)
        data_cnn    = data_cnn.reshape ( int(data_cnn.shape[0]/window_size), window_size, 9, 9)
        # rnn data process
        data_rnn    = data. reshape(int(data.shape[0]/window_size), window_size, 32)
        # append new data and label
        data_inter_cnn  = np.vstack([data_inter_cnn, data_cnn])
        data_inter_rnn  = np.vstack([data_inter_rnn, data_rnn])
        label_inter = np.append(label_inter, label)

    print("total cnn size:", data_inter_cnn.shape)
    print("total rnn size:", data_inter_rnn.shape)
    print("total label size:", label_inter.shape)
    # shuffle data
    index = np.array(range(0, len(label_inter)))
    np.random.shuffle( index)
    shuffled_data_cnn	= data_inter_cnn[index]
    shuffled_data_rnn	= data_inter_rnn[index]
    shuffled_label 	= label_inter[index]
    return shuffled_data_cnn ,shuffled_data_rnn,shuffled_label,record

if __name__ == '__main__' :
    begin = time.time()
    print("time begin:",time.localtime())
    dataset_dir		=   get_args(). directory
    window_size		=	get_args ().window
    output_dir		=   get_args().output_dir
    # get directory name for one subject
    for j in range(1,33):
        record_list = [task for task in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir,task))]
        print(record_list)
        for record in record_list:
            file = os.path.join(dataset_dir,record)
            print_top(file, window_size, output_dir)
            shuffled_cnn_data,shuffled_rnn_data,shuffled_label,record = apply_mixup(file, window_size)
            output_data_cnn = output_dir+record+"_win_"+str(window_size)+"_cnn_dataset.pkl"
            output_data_rnn = output_dir+record+"_win_"+str(window_size)+"_rnn_dataset.pkl"
            output_label= output_dir+record+"_win_"+str(window_size)+"_labels.pkl"

            with open(output_data_cnn, "wb") as fp:
                pickle.dump( shuffled_cnn_data,fp, protocol=4)
            with open( output_data_rnn, "wb") as fp:
                pickle.dump(shuffled_rnn_data, fp, protocol=4)
            with open(output_label, "wb") as fp:
                pickle.dump(shuffled_label, fp)
            end = time.time()
            print("end time:",time.localtime())
            print("time consuming:",(end-begin))