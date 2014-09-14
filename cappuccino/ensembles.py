'''
Created on Aug 31, 2014

@author: Aaron Klein
'''

import h5py
import logging
import numpy as np
import copy
import cma

from google.protobuf import text_format

from caffe.proto import caffe_pb2
from caffe.classifier import Classifier


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
        logging.error("No scipy installed")
        #weights = optimize.fmin_slsqp(weighted_error, weights, args=(predictions, true_labels), eqcons=[weight_constraint, ], epsilon=0.5, bounds=bounds)
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


def entropy_measure(pred, true_labels):
    N = pred.shape[1]
    L = pred.shape[0]
    max_div = L - np.ceil(L / 2)
    pred_labels = pred.argmax(axis=2)
    E = 0
    for i in xrange(0, N):
        l = np.count_nonzero(pred_labels[:, i] == true_labels[i])
        E += (1 / max_div) * min(l, L - l)
    return E / N


def create_test_config(working_dir, net, valid_file, batch_size):
    #create a temporary caffe-config for the prediction
    test_config = working_dir + "/caffenet_test.prototxt"
    test_net = copy.deepcopy(net)
    test_net.name = "test"
    test_net.layers[0].hdf5_data_param.source = valid_file
    test_net.layers[0].hdf5_data_param.batch_size = batch_size

    last_layer_top = test_net.layers[-1].top[0]
    prob_layer = test_net.layers.add()
    prob_layer.name = "prob"
    prob_layer.type = caffe_pb2.LayerParameter.SOFTMAX
    prob_layer.bottom.append(last_layer_top)
    prob_layer.top.append("prob")

    with open(test_config, "wb") as fh:
        fh.write(str(test_net))
        fh.close()
    return test_config


def get_true_labels(valid_file):
    valid = open(valid_file, 'r').readline().strip('\n')
    #load valid data labels
    f = h5py.File(valid, "r")
    l = f['label']
    true_labels = np.array(l)
    f.close()
    return true_labels
