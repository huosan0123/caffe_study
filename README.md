## caffe_study
This repo contains some implementations of my interested net, including net proto/solver proto, and some tools.

#### inception-v4
For: solver_in4.prototxt, inception_v4_train.prototxt, inception_v4_test.prototxt.  I was trying to seperate common inception_v4_train_test.prototxt into train and test file. Just change include phase!

#### resnet_train_test.prototxt
A implementation of resnet50.

#### test_net.py
After deploying the model, we need to do prediction for a image, so here is a tool for that. See notes and details in it.
