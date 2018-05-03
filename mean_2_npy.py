#coding=utf-8
import caffe
import numpy as np
# main part借鉴自https://blog.csdn.net/hyman_yx/article/details/51732656
# 

def convert_from_binary(mean_proto_path, mean_npy_path)
	blob = caffe.proto.caffe_pb2.BlobProto()           # 创建protobuf blob
	data = open(MEAN_PROTO_PATH, 'rb' ).read()         # 读入mean.binaryproto文件内容
	blob.ParseFromString(data)                         # 解析文件内容到blob
	array = np.array(caffe.io.blobproto_to_array(blob))# 将blob中的均值转换成numpy格式，array的shape （mean_number，channel, hight, width）
	mean_npy = array[0]                                # 一个array中可以有多组均值存在，故需要通过下标选择其中一组均值
	print mean_npy
	np.save(MEAN_NPY_PATH ,mean_npy)
	print('convert done')

def convert_from_list(mean_list, mean_npy_path)
	# if you only have mean value of RGB, use this
	mean = np.ones([3,256, 256], dtype=np.float)
	mean[0,:,:] = mean_list[0]
	mean[1,:,:] = mean_list[1]
	mean[2,:,:] = mean_list[2]
	np.save(mean_npy_path, mean)
	print('convert done')
# mean_value channel [0]: 90.4254
# mean_value channel [1]: 99.2691
# mean_value channel [2]: 107.671
