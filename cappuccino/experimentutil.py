import fcntl
import json
import os
import re
import sys
import cPickle
import traceback
import h5py
import copy
import logging
import numpy as np
from collections import defaultdict
from caffe.proto import caffe_pb2
from cappuccino.paramutil import hpolib_to_caffenet
from cappuccino.ensembles import predict, weighted_ensemble, average_correlation


def get_current_ybest():
    ybest_curr = None
    if os.path.exists("ybest.txt"):
        ybest_curr = float(open("ybest.txt").read())
    return ybest_curr


def update_ybest(y_candidate):
    """
         y_candidate: latest accuracy

         Set ybest in ybest.txt if y_candidate is lower than the previous ybest.

         returns the current ybest.
    """
    ybest_curr = get_current_ybest()
    if ybest_curr is None or y_candidate > ybest_curr:
        with open("ybest.txt", "w") as ybest_file:
            ybest_file.write(str(y_candidate))
        return y_candidate
    else:
        return ybest_curr


def store_result(dirname, params, loss, total_time, learning_curves,
    learning_curve_timestamps, predicted_loss=None, extra={}):
    """
        Store the results in a central file, one line of json per experiment.

    """
    result_file_name = os.path.join(dirname, "results.json")
    with open(result_file_name, "a") as result_file:
        #lock file:
        fcntl.lockf(result_file.fileno(), fcntl.LOCK_EX)
        result_line = {"loss": loss,
                       "total_time": total_time,
                       "params": params,
                       "learning_curves": learning_curves,
                       "learning_curve_timestamps": learning_curve_timestamps}
        if predicted_loss is not None:
            result_line["predicted_loss"] = predicted_loss
        result_line.update(extra)
        result_file.write(json.dumps(result_line))
        result_file.write("\n")


def log_error(dirname, error_msg):
    """
        Store the errors that occur in a central file
    """
    error_log_file_name = os.path.join(dirname, "errors.txt")
    with open(error_log_file_name, "a") as error_log_file:
        #lock file:
        fcntl.lockf(error_log_file.fileno(), fcntl.LOCK_EX)

        error_log_file.write(error_msg)
        error_log_file.write("\n")
        error_log_file.write("\n")


def read_learning_curve():
    """
        Read the learning curve from a file.
        Expects the file learning_curve.txt in the current folder.
    """
    with open("learning_curve.txt") as f:
        learning_curve = f.read().split(",")[:-1]
        return learning_curve
    return []


def learning_curve_from_log(lines):
    """
    lines: the line by line output of caffe

    returns learning curves for each network, timestamps
    """
    #example test accuracy:
    #I0512 15:43:21.701407 13354 solver.cpp:183] valid test score #0: 0.0792
    line_regex = "[^]]+]\s(\w+)\stest score\s#0:\s(\d+\.?\d*)"
    #test timestamp line
    #I0512 16:29:38.952080 13854 solver.cpp:141] Test timestamp 1399904978
    time_regex = "[^]]+] Test timestamp (\d+)"

    network_learning_curves = defaultdict(list)

    learning_curve_timestamps = []

    mday = 1
    for line in lines:
        m = re.match(time_regex, line.strip())
        if m:
            learning_curve_timestamps.append(int(m.group(1)))
        m = re.match(line_regex, line.strip())
        if m:
            network_name = m.group(1)
            accuracy = float(m.group(2))

            network_learning_curves[network_name].append(accuracy)

    return network_learning_curves, learning_curve_timestamps


def get_last_model_snapshot(lines):

    #I0901 11:45:53.896986 28026 solver.cpp:268] Snapshotting to caffenet_0_iter_132720
    regex = "[^]]+] Snapshotting to (\w+)"

    for line in lines:
        m = re.match(regex, line.strip())
        if m:
            model = m.group(1)
            return model
    return None


def get_validation_accuracy(lines):
    #example test accuracy:
    #I0512 15:43:21.701407 13354 solver.cpp:183] valid test score #0: 0.0792
    line_regex = "[^]]+] valid test score\s#0:\s(\d+\.?\d*)"

    accuracy = None
    for line in lines:
            m = re.match(line_regex, line.strip())
            if m:
                accuracy = float(m.group(1))
    return accuracy


def hpolib_experiment_ensemble_main(params, construct_caffeconvnet,
    experiment_dir, working_dir, mean_performance_on_last, **kwargs):
    """
        params: parameters coming directly from hpolib
        construct_caffeconvnet: a function that takes caffeconvnet parameters and constructs a CaffeConvNet
        mean_performance_on_last: take average of the last x values from the validation network as the reported performance.
    """
    try:
        standard = False
        corr_acc = True
        caffe_convnet_params = hpolib_to_caffenet(params)

        caffeconvnet = construct_caffeconvnet(caffe_convnet_params)
        output_log = caffeconvnet.run()

        #create a temporary caffe-config for the prediction
        test_config = working_dir + "/caffenet_test.prototxt"
        test_net = copy.deepcopy(caffeconvnet._caffe_net)
        test_net.name = "test"
        test_net.layers[0].hdf5_data_param.source = caffeconvnet._valid_file
        test_net.layers[0].hdf5_data_param.batch_size = caffeconvnet._batch_size_valid

        last_layer_top = test_net.layers[-1].top[0]
        prob_layer = test_net.layers.add()
        prob_layer.name = "prob"
        prob_layer.type = caffe_pb2.LayerParameter.SOFTMAX
        prob_layer.bottom.append(last_layer_top)
        prob_layer.top.append("prob")

        with open(test_config, "wb") as fh:
            fh.write(str(test_net))
            fh.close()

        model = get_last_model_snapshot(output_log.split("\n"))
        model = working_dir + "/" + model
        if model == None:
            log_error(experiment_dir, output_log)
            raise Exception("no valid model found")

        batch_size = caffeconvnet._batch_size_valid
        valid_file = caffeconvnet._valid_file
        valid = open(valid_file, 'r').readline().strip('\n')
        #load valid data labels
        f = h5py.File(valid, "r")
        l = f['label']
        true_labels = np.array(l)
        f.close()

        ndata = true_labels.shape[0]
        assert ndata > 0

        nclasses = np.unique(true_labels).shape[0]
        assert nclasses > 2

        #predictions of current model
        pred = predict(test_config, model.strip('\n'), ndata, nclasses, batch_size)

        pred_labels = np.argmax(pred, axis=1)
        npoints = pred.shape[0]
        acc = float(np.count_nonzero(true_labels.T[0] == pred_labels)) / npoints
        logging.debug("get valid acc from log: " + str(get_validation_accuracy(output_log.split("\n"))))

        #check if predictions.pkl already exist
        if os.path.exists("predictions.pkl"):
            #load previous predictions
            predictions = cPickle.load(open("predictions.pkl", 'rb'))
            #ensemble prediction
            predictions = np.concatenate((predictions, np.array([pred])), axis=0)
            ensemble_pred = predictions.sum(axis=0)
            #save predictions
            cPickle.dump(predictions, open("predictions.pkl", 'wb'))
            #check how many predictions are correct

            if standard == True:
                npoints = predictions.shape[1]
                pred_labels = np.argmax(ensemble_pred, axis=1)
                acc = float(np.count_nonzero(true_labels.T[0] == pred_labels)) / npoints
                error = 1 - acc
                return error
            elif corr_acc == True:
                #correlation between the last network and all others

                corr = average_correlation(predictions)[-1]

                logging.debug("corr: " + str(corr.mean()))
                logging.debug("acc: " + str(acc))
                if np.isnan(corr.mean()):
                    return 1.0
                error = 1 - acc
                return (error + corr.mean()) / 2

        else:
            cPickle.dump(np.array([pred]), open("predictions.pkl", 'wb'))
            pred_labels = np.argmax(pred, axis=1)
            npoints = pred.shape[0]
            acc = float(np.count_nonzero(true_labels.T[0] == pred_labels)) / npoints

            logging.debug("acc: " + str(acc))

            error = 1 - acc
            return error

    except Exception:
        print "Unexpected error:", sys.exc_info()[0]
        print "Trackback: ", traceback.format_exc()
        log_error(experiment_dir, str(sys.exc_info()[0]))
        log_error(experiment_dir, str(traceback.format_exc()))
        #maximum loss:
        return 1.0


def hpolib_experiment_main_ensemble_arithmetic_mean(params, construct_caffeconvnet,
    experiment_dir, working_dir, mean_performance_on_last, **kwargs):
    """
        Compute the error based on the arithmetic mean softmax layer probabilities

        params: parameters coming directly from hpolib
        construct_caffeconvnet: a function that takes caffeconvnet parameters and constructs a CaffeConvNet
        mean_performance_on_last: take average of the last x values from the validation network as the reported performance.
    """
    try:

        caffe_convnet_params = hpolib_to_caffenet(params)

        caffeconvnet = construct_caffeconvnet(caffe_convnet_params)
        output_log = caffeconvnet.run()

        #create a temporary caffe-config for the prediction
        test_config = working_dir + "/caffenet_test.prototxt"
        test_net = copy.deepcopy(caffeconvnet._caffe_net)
        test_net.name = "test"
        test_net.layers[0].hdf5_data_param.source = caffeconvnet._valid_file
        test_net.layers[0].hdf5_data_param.batch_size = caffeconvnet._batch_size_valid

        last_layer_top = test_net.layers[-1].top[0]
        prob_layer = test_net.layers.add()
        prob_layer.name = "prob"
        prob_layer.type = caffe_pb2.LayerParameter.SOFTMAX
        prob_layer.bottom.append(last_layer_top)
        prob_layer.top.append("prob")

        with open(test_config, "wb") as fh:
            fh.write(str(test_net))
            fh.close()

        model = get_last_model_snapshot(output_log.split("\n"))
        model = working_dir + "/" + model
        if model == None:
            log_error(experiment_dir, output_log)
            raise Exception("no valid model found")

        batch_size = caffeconvnet._batch_size_valid
        valid_file = caffeconvnet._valid_file
        valid = open(valid_file, 'r').readline().strip('\n')
        #load valid data labels
        f = h5py.File(valid, "r")
        l = f['label']
        true_labels = np.array(l)
        f.close()

        ndata = true_labels.shape[0]
        assert ndata > 0

        nclasses = np.unique(true_labels).shape[0]
        assert nclasses > 2

        #predictions of current model
        pred = predict(test_config, model.strip('\n'), ndata, nclasses, batch_size)

        pred_labels = np.argmax(pred, axis=1)
        npoints = pred.shape[0]
        acc = float(np.count_nonzero(true_labels.T[0] == pred_labels)) / npoints
        logging.debug("get valid acc from log: " + str(get_validation_accuracy(output_log.split("\n"))))

        #check if predictions.pkl already exist
        if os.path.exists("predictions.pkl"):
            #load previous predictions
            predictions = cPickle.load(open("predictions.pkl", 'rb'))
            #ensemble prediction
            predictions = np.concatenate((predictions, np.array([pred])), axis=0)

            #arithmetic mean of the predictions
            ensemble_pred = predictions.mean(axis=0)

            #save predictions
            cPickle.dump(predictions, open("predictions.pkl", 'wb'))
            #check how many predictions are correct

            npoints = predictions.shape[1]
            pred_labels = np.argmax(ensemble_pred, axis=1)
            acc = float(np.count_nonzero(true_labels.T[0] == pred_labels)) / npoints
            error = 1 - acc
            return error

        else:
            cPickle.dump(np.array([pred]), open("predictions.pkl", 'wb'))
            pred_labels = np.argmax(pred, axis=1)
            npoints = pred.shape[0]
            acc = float(np.count_nonzero(true_labels.T[0] == pred_labels)) / npoints

            logging.debug("acc: " + str(acc))

            error = 1 - acc
            return error

    except Exception:
        print "Unexpected error:", sys.exc_info()[0]
        print "Trackback: ", traceback.format_exc()
        log_error(experiment_dir, str(sys.exc_info()[0]))
        log_error(experiment_dir, str(traceback.format_exc()))
        #maximum loss:
        return 1.0


def hpolib_experiment_main_ensemble_entropy(params, construct_caffeconvnet,
    experiment_dir, working_dir, mean_performance_on_last, **kwargs):
    """
        Search for networks that maximizes the entropy measure

        params: parameters coming directly from hpolib
        construct_caffeconvnet: a function that takes caffeconvnet parameters and constructs a CaffeConvNet
        mean_performance_on_last: take average of the last x values from the validation network as the reported performance.
    """
    try:

        caffe_convnet_params = hpolib_to_caffenet(params)

        caffeconvnet = construct_caffeconvnet(caffe_convnet_params)
        output_log = caffeconvnet.run()

        #create a temporary caffe-config for the prediction
        test_config = working_dir + "/caffenet_test.prototxt"
        test_net = copy.deepcopy(caffeconvnet._caffe_net)
        test_net.name = "test"
        test_net.layers[0].hdf5_data_param.source = caffeconvnet._valid_file
        test_net.layers[0].hdf5_data_param.batch_size = caffeconvnet._batch_size_valid

        last_layer_top = test_net.layers[-1].top[0]
        prob_layer = test_net.layers.add()
        prob_layer.name = "prob"
        prob_layer.type = caffe_pb2.LayerParameter.SOFTMAX
        prob_layer.bottom.append(last_layer_top)
        prob_layer.top.append("prob")

        with open(test_config, "wb") as fh:
            fh.write(str(test_net))
            fh.close()

        model = get_last_model_snapshot(output_log.split("\n"))
        model = working_dir + "/" + model
        if model == None:
            log_error(experiment_dir, output_log)
            raise Exception("no valid model found")

        batch_size = caffeconvnet._batch_size_valid
        valid_file = caffeconvnet._valid_file
        valid = open(valid_file, 'r').readline().strip('\n')
        #load valid data labels
        f = h5py.File(valid, "r")
        l = f['label']
        true_labels = np.array(l)
        f.close()

        ndata = true_labels.shape[0]
        assert ndata > 0

        nclasses = np.unique(true_labels).shape[0]
        assert nclasses > 2

        #predictions of current model
        pred = predict(test_config, model.strip('\n'), ndata, nclasses, batch_size)

        pred_labels = np.argmax(pred, axis=1)
        npoints = pred.shape[0]
        acc = float(np.count_nonzero(true_labels.T[0] == pred_labels)) / npoints
        logging.debug("get valid acc from log: " + str(get_validation_accuracy(output_log.split("\n"))))

        #check if predictions.pkl already exist
        if os.path.exists("predictions.pkl"):
            #load previous predictions
            predictions = cPickle.load(open("predictions.pkl", 'rb'))
            #ensemble prediction
            predictions = np.concatenate((predictions, np.array([pred])), axis=0)

            #save predictions
            cPickle.dump(predictions, open("predictions.pkl", 'wb'))

            return 1 - entropy_measure(predictions, true_labels)

        else:
            return 1 - (1 / ndata)

    except Exception:
        print "Unexpected error:", sys.exc_info()[0]
        print "Trackback: ", traceback.format_exc()
        log_error(experiment_dir, str(sys.exc_info()[0]))
        log_error(experiment_dir, str(traceback.format_exc()))
        #maximum loss:
        return 1.0


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


def hpolib_experiment_main_ensemble_geometric_mean(params, construct_caffeconvnet,
    experiment_dir, working_dir, mean_performance_on_last, **kwargs):
    """
        Compute the error based on the geometric mean of the different softmax layer probabilities

        params: parameters coming directly from hpolib
        construct_caffeconvnet: a function that takes caffeconvnet parameters and constructs a CaffeConvNet
        mean_performance_on_last: take average of the last x values from the validation network as the reported performance.
    """
    try:

        caffe_convnet_params = hpolib_to_caffenet(params)

        caffeconvnet = construct_caffeconvnet(caffe_convnet_params)
        output_log = caffeconvnet.run()

        #create a temporary caffe-config for the prediction
        #TODO: remove code duplicates
        test_config = working_dir + "/caffenet_test.prototxt"
        test_net = copy.deepcopy(caffeconvnet._caffe_net)
        test_net.name = "test"
        test_net.layers[0].hdf5_data_param.source = caffeconvnet._valid_file
        test_net.layers[0].hdf5_data_param.batch_size = caffeconvnet._batch_size_valid

        last_layer_top = test_net.layers[-1].top[0]
        prob_layer = test_net.layers.add()
        prob_layer.name = "prob"
        prob_layer.type = caffe_pb2.LayerParameter.SOFTMAX
        prob_layer.bottom.append(last_layer_top)
        prob_layer.top.append("prob")

        with open(test_config, "wb") as fh:
            fh.write(str(test_net))
            fh.close()

        model = get_last_model_snapshot(output_log.split("\n"))
        model = working_dir + "/" + model
        if model == None:
            log_error(experiment_dir, output_log)
            raise Exception("no valid model found")

        batch_size = caffeconvnet._batch_size_valid
        valid_file = caffeconvnet._valid_file
        valid = open(valid_file, 'r').readline().strip('\n')
        #load valid data labels
        f = h5py.File(valid, "r")
        l = f['label']
        true_labels = np.array(l)
        f.close()

        ndata = true_labels.shape[0]
        assert ndata > 0

        nclasses = np.unique(true_labels).shape[0]
        assert nclasses > 2

        #predictions of current model
        pred = predict(test_config, model.strip('\n'), ndata, nclasses, batch_size)

        pred_labels = np.argmax(pred, axis=1)
        npoints = pred.shape[0]
        acc = float(np.count_nonzero(true_labels.T[0] == pred_labels)) / npoints
        logging.debug("get valid acc from log: " + str(get_validation_accuracy(output_log.split("\n"))))

        #check if predictions.pkl already exist
        if os.path.exists("predictions.pkl"):
            #load previous predictions
            predictions = cPickle.load(open("predictions.pkl", 'rb'))
            #ensemble prediction
            predictions = np.concatenate((predictions, np.array([pred])), axis=0)

            #geometric mean of the predictions
            from scipy import stats
            ensemble_pred = stats.gmean(predictions)

            #save predictions
            cPickle.dump(predictions, open("predictions.pkl", 'wb'))
            #check how many predictions are correct

            npoints = predictions.shape[1]
            pred_labels = np.argmax(ensemble_pred, axis=1)
            acc = float(np.count_nonzero(true_labels.T[0] == pred_labels)) / npoints
            error = 1 - acc
            return error

        else:
            cPickle.dump(np.array([pred]), open("predictions.pkl", 'wb'))
            pred_labels = np.argmax(pred, axis=1)
            npoints = pred.shape[0]
            acc = float(np.count_nonzero(true_labels.T[0] == pred_labels)) / npoints

            logging.debug("acc: " + str(acc))

            error = 1 - acc
            return error

    except Exception:
        print "Unexpected error:", sys.exc_info()[0]
        print "Trackback: ", traceback.format_exc()
        log_error(experiment_dir, str(sys.exc_info()[0]))
        log_error(experiment_dir, str(traceback.format_exc()))
        #maximum loss:
        return 1.0


def hpolib_experiment_weighted_ensemble_main(params, construct_caffeconvnet,
    experiment_dir, working_dir, mean_performance_on_last, **kwargs):
    """
        params: parameters coming directly from hpolib
        construct_caffeconvnet: a function that takes caffeconvnet parameters and constructs a CaffeConvNet
        mean_performance_on_last: take average of the last x values from the validation network as the reported performance.
    """
    try:
        caffe_convnet_params = hpolib_to_caffenet(params)

        caffeconvnet = construct_caffeconvnet(caffe_convnet_params)
        output_log = caffeconvnet.run()

        #create a temporary caffe-config for the prediction
        test_config = working_dir + "/caffenet_test.prototxt"
        test_net = copy.deepcopy(caffeconvnet._caffe_net)
        test_net.name = "test"
        test_net.layers[0].hdf5_data_param.source = caffeconvnet._valid_file
        test_net.layers[0].hdf5_data_param.batch_size = caffeconvnet._batch_size_valid

        last_layer_top = test_net.layers[-1].top[0]
        prob_layer = test_net.layers.add()
        prob_layer.name = "prob"
        prob_layer.type = caffe_pb2.LayerParameter.SOFTMAX
        prob_layer.bottom.append(last_layer_top)
        prob_layer.top.append("prob")

        with open(test_config, "wb") as fh:
            fh.write(str(test_net))
            fh.close()

        model = get_last_model_snapshot(output_log.split("\n"))
        model = working_dir + "/" + model
        if model == None:
            log_error(experiment_dir, output_log)
            raise Exception("no valid model found")

        batch_size = caffeconvnet._batch_size_valid
        valid_file = caffeconvnet._valid_file
        valid = open(valid_file, 'r').readline().strip('\n')
        #load valid data labels
        f = h5py.File(valid, "r")
        l = f['label']
        true_labels = np.array(l)
        f.close()

        ndata = true_labels.shape[0]
        assert ndata > 0

        nclasses = np.unique(true_labels).shape[0]
        assert nclasses > 2

        #predictions of current model
        pred = predict(test_config, model.strip('\n'), ndata, nclasses, batch_size)
        #check if predictions.pkl already exist
        if os.path.exists("predictions.pkl"):
            #load previous predictions
            predictions = cPickle.load(open("predictions.pkl", 'rb'))
            #ensemble prediction
            predictions = np.concatenate((predictions, np.array([pred])), axis=0)
            #save predictions
            cPickle.dump(predictions, open("predictions.pkl", 'wb'))
            #check how many predictions are correct
            error, weights = weighted_ensemble(predictions, true_labels, method="cma")
            logging.debug("weights: " + str(weights))
            cPickle.dump(weights, open("weights.pkl", 'wb'))
        else:
            cPickle.dump(np.array([pred]), open("predictions.pkl", 'wb'))
            pred_labels = np.argmax(pred, axis=1)
            npoints = pred.shape[0]
            acc = float(np.count_nonzero(true_labels.T[0] == pred_labels)) / npoints
            error = 1 - acc
        return error

    except Exception:
        print "Unexpected error:", sys.exc_info()[0]
        print "Trackback: ", traceback.format_exc()
        log_error(experiment_dir, str(sys.exc_info()[0]))
        log_error(experiment_dir, str(traceback.format_exc()))
        #maximum loss:
        return 1.0


def hpolib_experiment_main(params, construct_caffeconvnet,
    experiment_dir, working_dir, mean_performance_on_last, **kwargs):
    """
        params: parameters coming directly from hpolib
        construct_caffeconvnet: a function that takes caffeconvnet parameters and constructs a CaffeConvNet
        mean_performance_on_last: take average of the last x values from the validation network as the reported performance.
    """
    try:
        caffe_convnet_params = hpolib_to_caffenet(params)

        output_log = construct_caffeconvnet(caffe_convnet_params).run()

        learning_curves, learning_curve_timestamps = learning_curve_from_log(output_log.split("\n"))

        if "valid" not in learning_curves:
            log_error(experiment_dir, output_log)
            raise Exception("no learning curve found")

        valid_learning_curve = learning_curves["valid"]
        best_accuracy = np.mean(valid_learning_curve[-mean_performance_on_last:])
        if not np.isfinite(best_accuracy):
            best_accuracy = .0
        lowest_error = 1.0 - best_accuracy
        total_time = learning_curve_timestamps[-1] - learning_curve_timestamps[0]

        #read output from termination_criterion
        best_predicted_accuracy = None
        lowest_predicted_error = None
        if os.path.exists("y_predict.txt"):
            best_predicted_accuracy = float(open("y_predict.txt").read())
            lowest_predicted_error = 1.0 - best_predicted_accuracy
            #make sure we don't use it in the next run as well..
            os.remove("y_predict.txt")

        try:
            current_ybest = get_current_ybest()
            store_result(experiment_dir, caffe_convnet_params, lowest_error, total_time,
                         learning_curves, learning_curve_timestamps, predicted_loss=lowest_predicted_error,
                         extra={"current_ybest": current_ybest})
            store_result(working_dir, caffe_convnet_params, lowest_error, total_time,
                         learning_curves, learning_curve_timestamps, predicted_loss=lowest_predicted_error,
                         extra={"current_ybest": current_ybest})
            update_ybest(best_accuracy)
        except Exception as e:
            print "Unexpected error:", sys.exc_info()[0]
            print "Trackback: ", traceback.format_exc()
            log_error(experiment_dir, str(sys.exc_info()[0]))
            log_error(experiment_dir, str(traceback.format_exc()))
        finally:
            if lowest_predicted_error is not None and np.isfinite(lowest_predicted_error):
                return lowest_predicted_error
            elif np.isfinite(lowest_error):
                return lowest_error
            else:
                raise Exception("RESULT NOT FINITE!")
    except Exception:
        print "Unexpected error:", sys.exc_info()[0]
        print "Trackback: ", traceback.format_exc()
        log_error(experiment_dir, str(sys.exc_info()[0]))
        log_error(experiment_dir, str(traceback.format_exc()))
        #raise e
        #maximum loss:
        return 1.0
