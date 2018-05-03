#coding=utf-8
import numpy as np
import caffe
import sklearn.metrics as metrics
import time, sys
"""
I was going to speed up the predicting, but it doesn't helps.
I think it's probably bacause loading data by python is slow.
Other setting is like that in test_net.py
"""

def get_jpg_label(test_file, test_path):
    lines = open(test_file, 'r').readlines()
    images, labels = [], []
    for line in lines:
        new_line = line.strip().split('\t')
        images.append(test_path + new_line[0])
        labels.append(int(new_line[1]))
    return images, labels

net_file='../deploy_2.prototxt'
caffe_model='../resnet50_iter_16000.caffemodel'
mean_file='mean.npy'
test_path = './test_jpgs/'
test_file = 'test_data.txt'
#if you have no GPUs,set mode cpu
caffe.set_mode_gpu()
net = caffe.Net(net_file, caffe_model, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255) 
transformer.set_channel_swap('data', (2, 1, 0))
images, labels = get_jpg_label(test_file, test_path)

# the data holder of every batch is needed
batch_labels, batch_imgs, batch_size =[], [], 64
true_labels, pred_logits, pred_labels = [], [], []
pre_time, x_time = time.time(), 0.0
for index, image_path in enumerate(images):
    try:
        img = caffe.io.load_image(image_path)
        img = transformer.preprocess('data', img)
        batch_imgs.append(img)
        batch_labels.append(labels[index])
    except:
        continue
    if len(batch_labels) == batch_size:
        net.blobs['data'].data[...] = batch_imgs
        x_start = time.time()
        pred = net.forward()
        x_gap = time.time() - x_start
        logit = pred["prob"]
        pred_logits.extend(logit[:, 1])
        pred_label = np.argmax(logit, axis=1)
        pred_labels.extend(pred_label)

        true_labels.extend(batch_labels)
        batch_imgs = []
        batch_labels = []
        x_time += x_gap

    if (index+1) % 2048 == 0:
        gap = (time.time() - pre_time) / 60
        print("TIME= {} min, forward time = {}s, Have processed {} images".format(gap, x_time, index+1))
        sys.stdout.flush()
        pre_time = time.time()
        x_time = 0.0
    if (index+1) % 4096 == 0:
        break

np.save('true_labels.npy', true_labels)
np.save('pred_logits.npy', pred_logits)
np.save('pred_labels.npy', pred_labels)
# metrics.roc_auc_score need a list of y_true and  a list of logits_pred
auc = metrics.roc_auc_score(true_labels, pred_logits)
print('the AUC is {}'.format(auc))
acc = metrics.accuracy_score(true_labels, pred_labels)
print('the accuracy is {}'.format(acc))

""" output time: 
TIME= 1.34273308118 min, forward time = 5.41079306602,  Have processed 1024 images
TIME= 1.23873046637 min, forward time = 5.48528146744,  Have processed 2048 images
TIME= 1.1734894673 min, forward time = 5.52893972397,  Have processed 3072 images
TIME= 1.30460258325 min, forward time = 5.39959812164,  Have processed 4096 images
Using nvidia-smi to see GPU utility percent, most time is very low!
""""
