import random
import tensorflow as  tf
import rnn.Configurations
import numpy as np
from util.dataLoader import loadData

import time

STEP_SIZE = 48
BATCH_SIZE = 30
EPOCH_ITER = 4
config = None

def next_batch(xData, yData, batchIndices, epoch_idx):
    xList = []
    nxList = []
    yList = []
    for i in range(BATCH_SIZE):
        idx1 = batchIndices[i] + epoch_idx*STEP_SIZE
        idx2 = idx1 + STEP_SIZE
        xList.append(xData[idx1-1:idx2-1])
        nxList.append(xData[idx1:idx2])
        yList.append(yData[idx1:idx2])
    return xList, nxList, yList


def batch_indices(yData, xData):
    size = len(yData)
    indices = []
    start_y = []
    start_x = []
    for _ in range(BATCH_SIZE):
        idx = random.randrange(1, size - STEP_SIZE*(EPOCH_ITER+1));
        indices.append(idx)
        start_y.append(yData[idx-1])
        start_x.append(xData[idx-1])
    return indices, start_y, start_x


def run(model, sess, values, feedDict):
    if (config.IS_STATE_MODEL == False):
        if ((model.initial_state in feedDict) and feedDict[model.initial_state] == None):
            del feedDict[model.initial_state]
    return sess.run(values, feedDict)

def run_training(folder, load=False, test=False, lr=0.0001):
    np.set_printoptions(formatter={'float_kind':lambda x:"%.6f"%x})
    
    xData = loadData("%s/data/xData.dat"%(folder))
    yData = loadData("%s/data/yData.dat"%(folder))
    
    global config
    config = rnn.Configurations.get_config(folder)
    config.load_normal_data(folder)
    global STEP_SIZE
    STEP_SIZE = config.TRAIN_STEP_SIZE
    global BATCH_SIZE
    BATCH_SIZE = config.TRAIN_BATCH_SIZE
    global EPOCH_ITER
    EPOCH_ITER = config.TRAIN_EPOCH_ITER
    
    if (test):
        BATCH_SIZE = 1
    
    tStart = time.time()
    model = config.model(BATCH_SIZE, STEP_SIZE, lr)
    
    with tf.Session() as sess:
#     with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=38)) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        if load == True:
            if (test):
                saver.restore(sess, "%s/train/ckpt"%(folder))
            else:
                saver.restore(sess, "%s/train2/ckpt"%(folder))
        tTrain1 = 0
        tSave = 0
        print("initialize time %d"%(time.time() - tStart))
        for idx in range(1000000):
            batchIndices, start_y, start_x = batch_indices(yData, xData)
            state = None
            lossList = []
            if config.IS_PREDICTION_MODEL:
                current_y = np.concatenate([start_y, start_x], axis=1)
            else:
                current_y = start_y
            if (config.IS_STATE_MODEL):
                state = [[0]*config.STATE_DIMENSION]*BATCH_SIZE
            
            for i in range(EPOCH_ITER):
                xList, nxList, yList = next_batch(xData, yData, batchIndices, i)
                
                if (test):
                    loss_g, loss_detail, generated, state, current_y = run(model, sess, \
                                [model.loss_g, model.loss_detail, model.generated, model.final_state, model.final_y], \
                                { model.x:xList, model.initial_state:state, model.y:yList, model.prev_y:current_y })
                    #print(len(generated))
                    motion = generated[0]
                    for fIdx in range(len(motion)):
                        frame = motion[fIdx] 
                        frame = config.y_normal.de_normalize_l(frame)
                        print("%d :: %s, %s, %s"%(fIdx, frame[0:3], frame[6:8], loss_detail[3:7]))
                    reg_loss_g = 0
                else:
                    t0 = time.time()
                    if config.IS_PREDICTION_MODEL:
                        yList = np.concatenate([yList, nxList], axis=2)
                    loss_g, loss_detail, reg_loss_g, state, current_y, _ = run(model, sess, \
                                [model.loss_g, model.loss_detail, model.reg_loss_g, model.final_state, model.final_y, model.train_list ], \
                                { model.x:xList, model.initial_state:state, model.y:yList, model.prev_y:current_y })
                    tTrain1 += (time.time() - t0)
                
                
                
                loss_detail.extend([loss_g, reg_loss_g])
                lossList.append(loss_detail)
            
            if (test):
                break
            
            if ((idx % 50) == 0):
                t0 = time.time()
                saver.save(sess, "%s/train/ckpt"%(folder))
                tSave += (time.time() - t0)
            
            log_mean = np.mean(lossList, 0)
            print('step %s %d : %s'%(folder, idx, [log_mean, tTrain1, tSave, time.time() - tStart]))            
