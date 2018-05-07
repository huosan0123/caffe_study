#coding=utf-8
import numpy as np
import caffe
import sklearn.metrics as metrics
import time, sys
from multiprocessing import Pool

# parallel predicting of caffe deploy proto.
# About ten times faster than single version
# I should enclose it into a class

def get_jpg_label(test_file, test_path):
    lines = open(test_file, 'r').readlines()
    images, images_path, labels = [], [], []
    for line in lines:
        new_line = line.strip().split('\t')
        images.append(new_line[0])
        labels.append(int(new_line[1]))
        images_path.append(test_path + new_line[0])
    return images, labels, images_path

net_file='./deploy.prototxt'
caffe_model='./net_iter_30000.caffemodel'
mean_file='mean_dark.npy'
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
images, labels, images_path = get_jpg_label(test_file, test_path)

def read_image(image_path):
    img = caffe.io.load_image(image_path)
    img = transformer.preprocess('data', img)
    return img

batch_size, batch_labels, batch_imgs, batch_names = 32, [], [], []
true_labels, pred_logits, pred_labels, image_names = [], [], [], []
pool = Pool(processes=32)
image_async_list, i = [None] * batch_size, 0
pre_time = time.time()

for index, image_path in enumerate(images_path):
    try:
        image_async_list[i] = pool.apply_async(read_image, (image_path,))
        batch_labels.append(labels[index])
        batch_names.append(images[index])
    except:
        continue
    i += 1
    if i == batch_size:
        for i in range(batch_size):
            batch_imgs.append(image_async_list[i].get())
        i = 0
        net.blobs['data'].data[...] = batch_imgs
        pred = net.forward()
        logit = pred["prob"]
        logit = logit.reshape((-1,2))
        pred_logits.extend(logit[:, 1])
        pred_label = np.argmax(logit, axis=1)
        pred_labels.extend(pred_label)
        image_names.extend(batch_names)
        true_labels.extend(batch_labels)

        batch_imgs = []
        batch_labels = []
        batch_names = []

    if (index+1) % 1024 == 0:
        gap = (time.time() - pre_time) / 60
        print("TIME= {} min, Have processed {} images".format(gap, index+1))
        sys.stdout.flush()
pool.close()

auc = metrics.roc_auc_score(true_labels, pred_logits)
print('the AUC is {}'.format(auc))
acc = metrics.accuracy_score(true_labels, pred_labels)
print('the accuracy is {}'.format(acc))
