#coding=utf-8
import numpy as np
import caffe
import sklearn.metrics as metrics
import time, sys
from multiprocessing import Pool

# My implementation of testing in caffemodel, a more object-oriented parallel version.
# 10x faster than unparalleled version.
# if you want to use, you may change the logit according to your net softmax layer.

def read_image(inputs):
    image_path, transformer = inputs
    img = caffe.io.load_image(image_path)
    img = transformer.preprocess('data', img)
    return img

class Parallel_test(object):
    def __init__(self, net_file, caffe_model, mean_file, test_path, test_file, batch_size):
        self.net_file = net_file
        self.caffe_model = caffe_model
        self.mean_file = mean_file
        self.test_path = test_path
        self.test_file = test_file
        self.batch_size = batch_size
        #if you have no GPUs,set mode cpu
        caffe.set_mode_gpu()
        self.net = caffe.Net(net_file, caffe_model, caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_channel_swap('data', (2, 1, 0))


    def get_jpg_label(self):
        lines = open(self.test_file, 'r').readlines()
        images, labels = [], []
        for line in lines:
            new_line = line.strip().split('\t')
            images.append(self.test_path + new_line[0])
            labels.append(int(new_line[1]))
        return images, labels


    def predict(self):
        pool = Pool(processes=32)
        images, labels = self.get_jpg_label()
        batch_labels, batch_imgs, batch_size =[], [], self.batch_size
        self.true_labels, self.pred_logits, self.pred_labels = [], [], []
        image_async_list, ibatch = [None] * batch_size, 0
        pre_time = time.time()

        for index, image_path in enumerate(images):
            inputs = [image_path, self.transformer]
            image_async_list[ibatch] = pool.apply_async(read_image, (inputs,))
            batch_labels.append(labels[index])
            ibatch += 1
            if ibatch == batch_size:
                ibatch = 0
                for i in range(batch_size):
                    img = image_async_list[i].get()
                    batch_imgs.append(img)

                self.net.blobs['data'].data[...] = batch_imgs
                pred = self.net.forward()
                logit = pred["prob"]
                logit = logit.reshape((-1, 2))
                pred_label = np.argmax(logit, axis=1)

                self.pred_logits.extend(logit[:, 1])
                self.pred_labels.extend(pred_label)
                self.true_labels.extend(batch_labels)
                batch_imgs = []
                batch_labels = []

            if (index+1) % 1024 == 0:
                gap = (time.time() - pre_time) / 60
                print("TIME= {} min, Have processed {} images".format(gap, index+1))
                sys.stdout.flush()
            if (index+1) == 10240:
                break

    # metrics.roc_auc_score need a list of y_true and  a list of logits_pred
    def measure(self):
        #np.save('true_labels.npy', true_labels)
        #np.save('pred_logits.npy', pred_logits)
        #np.save('pred_labels.npy', pred_labels)
        auc = metrics.roc_auc_score(self.true_labels, self.pred_logits)
        print('the AUC is {}'.format(auc))
        acc = metrics.accuracy_score(self.true_labels, self.pred_labels)
        print('the accuracy is {}'.format(acc))

a = Parallel_test('deploy.prototxt', 'darknet_iter_30000.caffemodel',
        'mean_dark.npy', './test_jpgs/', 'test_data.txt', 32)
a.predict()
a.measure()
