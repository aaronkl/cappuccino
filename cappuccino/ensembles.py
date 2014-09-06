'''
Created on Aug 31, 2014

@author: Aaron Klein
'''

import h5py
import logging
import numpy as np

import cma

from google.protobuf import text_format

from caffe.proto import caffe_pb2
from caffe.classifier import Classifier

from scipy import optimize


def average_correlation(predictions):
    corr = 0

    for i in xrange(0, predictions.shape[1]):
        #add small epsilon to avoid nan if variance is 0
        corr += np.corrcoef(predictions[:, i, :] + 1e-10)

    corr /= predictions.shape[1]
    return corr


def weighted_error(weights, *args):
    predicitons = args[0]

    true_labels = args[1]
    npoints = predicitons.shape[1]
    nclasses = predicitons.shape[2]
    comp_labels = np.zeros([npoints, nclasses])

    for i, p in enumerate(predicitons):
        labels = np.argmax(p, axis=1)
        y = np.zeros(comp_labels.shape)
        for j in xrange(0, y.shape[0]):
            y[j, labels[j]] = 1
        comp_labels += weights[i] * y

    comp_labels += 0.5
    comp_labels = np.floor(comp_labels)

    #check how many predictions are correct
    acc = float(np.count_nonzero(true_labels.T[0] == np.argmax(comp_labels, axis=1))) / npoints

    return 1 - acc


def weight_constraint(x, *args):
    return 1 - np.sum(x)


def weighted_ensemble(predictions, true_labels, method="cma"):
    predictions = np.array(predictions)
    num_nets = predictions.shape[0]
    weights = np.ones([num_nets]) / num_nets
    if method == "local":
        bounds = [(0., 1.)]
        for i in xrange(1, weights.shape[0]):
            bounds.append((0, 1))

        weights = optimize.fmin_slsqp(weighted_error, weights, args=(predictions, true_labels), eqcons=[weight_constraint, ], epsilon=0.5, bounds=bounds)
        #weights, f_min, info = optimize.fmin_l_bfgs_b(weighted_error, weights, args=(predictions, true_labels), approx_grad=True, epsilon=0.1)

    elif method == "cma":
        bounds = [0, 1]
        if not weights.shape[0] == 1:
            #CMA does not work in a 1-D space
            res = cma.fmin(weighted_error, weights, sigma0=0.25, args=(predictions, true_labels), options={'bounds': [0, 1]})
            #hacky, it sums the weights to 1
            weights = res[0] / np.sum(res[0])
        else:
            logging.error("CMA does not work in a 1D space")

    err = weighted_error(weights, predictions, true_labels)
    return err, weights


def predict(config, model, ndata, nclasses, batch_size, image_dims=(6400, 1)):
    """
        Take the data specified in the config file as inputs and compute the softmax predictions of all networks
    """
    #net = Classifier(config, model, image_dims=(1, 28, 28))

    net = Classifier(config, model, image_dims=image_dims)

    pred = np.zeros([ndata, nclasses])
    for i in xrange(0, ndata, batch_size):
        #pass one batch through the network
        output = net.forward()
        pred[i:i + batch_size] = np.asarray(output['prob'])[:, :, 0, 0]
    return pred


def create_test_config(valid_conifg, train_file, batch_size):
    """
        Create a temporary caffe-config file. This config is basically the same as the valid-config just without the accuracy layer
    """
    net = caffe_pb2.NetParameter()
    test_net = caffe_pb2.NetParameter()

    test_config = "tmp_test_config.prototxt"
    with open(valid_conifg, "rb") as fh:
        text_format.Merge(fh.read(), net)

    for i in xrange(0, len(net.layers) - 1):
        layer = test_net.layers.add()
        if net.layers[i].type == caffe_pb2.LayerParameter.HDF5_DATA:
            layer.name = net.layers[i].name
            layer.top.append("data")
            layer.top.append("label")
            layer.type = net.layers[i].type
            layer.hdf5_data_param.source = train_file
            layer.hdf5_data_param.batch_size = batch_size
            bottom = "data"

        elif net.layers[i].type == caffe_pb2.LayerParameter.CONVOLUTION:
            layer.name = net.layers[i].name
            layer.top.append(layer.name)
            layer.bottom.append(bottom)
            layer.type = net.layers[i].type
            layer.blobs_lr.append(net.layers[i].blobs_lr[0])
            layer.blobs_lr.append(net.layers[i].blobs_lr[1])
            layer.weight_decay.append(net.layers[i].weight_decay[0])
            layer.weight_decay.append(net.layers[i].weight_decay[1])
            layer.convolution_param.num_output = net.layers[i].convolution_param.num_output
            layer.convolution_param.weight_filler.type = net.layers[i].convolution_param.weight_filler.type
            layer.convolution_param.weight_filler.std = net.layers[i].convolution_param.weight_filler.std

            layer.convolution_param.bias_filler.type = net.layers[i].convolution_param.bias_filler.type
            layer.convolution_param.bias_filler.value = net.layers[i].convolution_param.bias_filler.value
            layer.convolution_param.pad = net.layers[i].convolution_param.pad
            layer.convolution_param.kernel_size = net.layers[i].convolution_param.kernel_size
            layer.convolution_param.stride = net.layers[i].convolution_param.stride
            bottom = layer.name
            top = layer.name

        elif net.layers[i].type == caffe_pb2.LayerParameter.INNER_PRODUCT:
            layer.name = net.layers[i].name
            layer.top.append(layer.name)
            layer.bottom.append(bottom)
            layer.type = net.layers[i].type
            layer.blobs_lr.append(net.layers[i].blobs_lr[0])
            layer.blobs_lr.append(net.layers[i].blobs_lr[1])
            layer.weight_decay.append(net.layers[i].weight_decay[0])
            layer.weight_decay.append(net.layers[i].weight_decay[1])
            layer.inner_product_param.num_output = net.layers[i].inner_product_param.num_output
            layer.inner_product_param.weight_filler.type = net.layers[i].inner_product_param.weight_filler.type
            #layer.inner_product_param.weight_filler.std = net.layers[i].inner_product_param.weight_filler.std
            layer.inner_product_param.bias_filler.type = net.layers[i].inner_product_param.bias_filler.type
            layer.inner_product_param.bias_filler.value = net.layers[i].inner_product_param.bias_filler.value

            bottom = layer.name
            top = layer.name

        elif net.layers[i].type == caffe_pb2.LayerParameter.DROPOUT:
            layer.name = net.layers[i].name
            layer.top.append(layer.name)
            layer.bottom.append(bottom)
            layer.type = net.layers[i].type
            layer.dropout_param.dropout_ratio = net.layers[i].dropout_param.dropout_ratio

            bottom = layer.name
            top = layer.name

        elif net.layers[i].type == caffe_pb2.LayerParameter.RELU:
            layer.type = net.layers[i].type
            layer.name = net.layers[i].name
            layer.top.append(top)
            layer.bottom.append(top)

        elif net.layers[i].type == caffe_pb2.LayerParameter.POOLING:
            layer.type = net.layers[i].type
            layer.name = net.layers[i].name
            layer.pooling_param.pool = net.layers[i].pooling_param.pool
            layer.pooling_param.kernel_size = net.layers[i].pooling_param.kernel_size
            layer.pooling_param.stride = net.layers[i].pooling_param.stride
            layer.top.append(layer.name)
            layer.bottom.append(bottom)
            bottom = layer.name
            top = layer.name

        elif net.layers[i].type == caffe_pb2.LayerParameter.LRN:
            print net.layers[i].type
            layer.type = net.layers[i].type
            layer.name = net.layers[i].name
            layer.lrn_param.local_size = net.layers[i].lrn_param.local_size
            layer.lrn_param.alpha = net.layers[i].lrn_param.alpha
            layer.lrn_param.beta = net.layers[i].lrn_param.beta
            layer.lrn_param.norm_region = net.layers[i].lrn_param.norm_region
            layer.top.append(layer.name)
            layer.bottom.append(bottom)
            bottom = layer.name
            top = layer.name

        elif net.layers[i].type == caffe_pb2.LayerParameter.SOFTMAX:
            layer.type = net.layers[i].type
            layer.name = net.layers[i].name
            layer.top.append(layer.name)
            layer.bottom.append(bottom)
        else:
            logging.error("Wrong layer type: " + str(net.layers[i].type) + " does not exist!")

    with open(test_config, "wb") as fh:
        fh.write(str(test_net))
        fh.close()

    return test_config
