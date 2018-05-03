#coding=utf-8
import numpy as np
import caffe
import sklearn.metrics as metrics

def get_jpg_label(test_file, test_path):
	"""
	every line of test file like this: 'abc.jpg 0'
	test path is the path of images saving dir
	Note: make sure that image_name in test file consistent with images in test_path
	"""
    lines = open(test_file, 'r').readlines()
    images, labels = [], []
    for line in lines:
        new_line = line.strip().split('\t')
        images.append(test_path + new_line[0])
        labels.append(int(new_line[1]))
    return images, labels

# if mean_file is mean.binaryproto, covert it to npy
net_file='../deploy.prototxt'
caffe_model='../resnet50_iter_16000.caffemodel'
mean_file='mean.npy'
# must have '/' at the end
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

true_labels, pred_logits, pred_labels = [], [], []
for index, image_path in enumerate(images):
	# if loading error occured, ignore
    try:
        img = caffe.io.load_image(image_path)
        true_labels.append(labels[index])
    except:
        continue
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    pred = net.forward()
    logit = pred["prob"][0]
    pred_label = np.argmax(logit)
    pred_logits.append(logit[1])
    pred_labels.append(pred_label)
    if (index+1) % 2000 == 0:
        print("Have processed {} images".format(index+1))
    if (index+1) % 2000 == 0:
        break

# details of sklearn.metrics.roc_auc_score and accuracy_score see its docs.
auc = metrics.roc_auc_score(true_labels_arr, pred_logits_arr)
print('the AUC is {}'.format(auc))
acc = metrics.accuracy_score(true_labels_arr, pred_logits_arr)
print('the accuracy is {}'.format(acc))
