from sklearn.metrics import zero_one_loss
from scipy.optimize import linear_sum_assignment
import numpy as np
import time
import logging


class RelativeValidation:
    """
    This class is initialized with the supervised algorithm to test cluster stability and
    the clustering algorithm whose labels are used as true labels.

    Methods:
        train() -- it compares the clustering labels on training set (i.e., A(X)) against
        the labels obtained through the classification algorithm (i.e., f(X)).
        It returns the misclassification error and the supervised model fitted to the data.

        test() -- it compares the clustering labels on the test set (i.e., A(X')) against
        the (permuted) labels obtained through the classification algorithm fitted to the training set
        (i.e., f(X')).
        It return the misclassification error.

    """

    def __init__(self, S, C):
        self.class_method = S
        self.clust_method = C

    def train(self, train_data):
        """
        Parameters
        ----------
        train_data: array with the training data
                    (n_samples, n_features)
        Returns
        -------
        misclass: float
        fit_model: fit object
        labels: dictionary of clustering and classification labels
                (array of integers (n_samples,))
        """

        clustlab_tr = self.clust_method.fit_predict(train_data)  # A_k(X)
        fitclass_tr = self.class_method.fit(train_data, clustlab_tr)
        classlab_tr = fitclass_tr.predict(train_data)

        misclass = zero_one_loss(clustlab_tr, classlab_tr)

        labels = {'classification': classlab_tr,
                  'clustering': clustlab_tr}

        return misclass, fitclass_tr, labels

    def test(self, test_data, fit_model):
        """
        Parameters
        ----------
        test_data: array with test data
                   (n_samples, n_features)
        fit_model: object, fitted model

        Returns
        -------
        misclass: float
        labels: dictionary of clustering and classification labels
                (array of integers (n_samples, ))
        """
        clustlab_ts = self.clust_method.fit_predict(test_data)  # A_k(X')
        classlab_ts = fit_model.predict(test_data)
        bestperm = _kuhn_munkres_algorithm(clustlab_ts, classlab_ts)  # array of integers
        # print(f'Clust ts:{clustlab_ts}, Classific ts:{classlab_ts}, Best perm:{bestperm}')
        misclass = zero_one_loss(clustlab_ts, bestperm)
        # print(f'Misclassification ts:{misclass}')
        labels = {'classification': classlab_ts,
                  'clustering': clustlab_ts}

        return misclass, labels

    def train_eval_rnd(self, train_data, test_data, rand_iter, train_labels, test_labels):
        """"
        Method performing random labeling training and validation
        rand_iter times
        Parameters
        ----------
        train_data: numpy array
        test_data: numpy array
        rand_iter: int
            number of iterations
        train_labels: numpy array
        test_labels: numpy array
        Returns
        -------
        float
        float
            Averaged misclassification scores
        """
        np.random.seed(0)
        shuf_tr = [np.random.permutation(train_labels)
                   for _ in range(rand_iter)]
        misclass_tr, misclass_ts = [], []
        for lab in shuf_tr:
            self.class_method.fit(train_data, lab)
            misclass_tr.append(zero_one_loss(lab, self.class_method.predict(train_data)))
            misclass_ts.append(zero_one_loss(test_labels,
                                             _kuhn_munkres_algorithm(test_labels,
                                                                     self.class_method.predict(test_data))))
        return np.mean(misclass_tr), np.mean(misclass_ts)
        # start = time.process_time()
        # np.random.seed(0)
        # shuf_tr, shuf_val = zip(*[list(map(lambda x: np.random.permutation(x),
        #                                    [tr_labels, val_labels])) for _ in range(rand_iter)])
        # # print(f'Shuffle train:{shuf_tr}, Shuffle val:{shuf_val}')
        # # logging.info(f'Shuffle labels: {time.process_time()-start}s')
        # # part = time.process_time()
        # model_tr = [self.class_method.fit(train_data, lab) for lab in shuf_tr]
        # # print(f'Models:{model_tr}')
        # # logging.info(f'Train model: {time.process_time()-part}s')
        # # part = time.process_time()
        # misclass_tr = [zero_one_loss(x, y.predict(train_data)) for x, y in
        #                zip(shuf_tr, model_tr)]
        # # print(f'Misclassification tr:{misclass_tr}')
        # # logging.info(f'Misclass TR: {time.process_time()-part}s')
        # # part = time.process_time()
        # misclass_val = [zero_one_loss(x, _kuhn_munkres_algorithm(x,
        #                                                          y.predict(val_data))) for x, y in
        #                 zip(shuf_val, model_tr)]
        # # print(f'Misclassification val:{misclass_val}')
        # # logging.info(f'Misclass VAL: {time.process_time()-part}s')
        # return np.mean(misclass_tr), np.mean(misclass_val)
        # misc_avg_tr, misc_avg_ts = [], []
        # append_tr = misc_avg_tr.append
        # append_ts = misc_avg_ts.append
        # for it in range(rand_iter):
        #     np.random.seed(0)
        #     np.random.shuffle(tr_labels)
        #     np.random.seed(0)
        #     np.random.shuffle(val_labels)
        #     model_tr = self.class_method.fit(train_data, tr_labels)
        #     misclass_tr = zero_one_loss(tr_labels,
        #                                 self.class_method.predict(train_data))
        #     misclass_ts = zero_one_loss(val_labels,
        #                                 _kuhn_munkres_algorithm(val_labels,
        #                                                         model_tr.predict(val_data)))
        #     append_tr(misclass_tr)
        #     append_ts(misclass_ts)
        # return np.mean(misc_avg_tr), np.mean(misc_avg_ts)


def _kuhn_munkres_algorithm(true_lab, pred_lab):
    """
    Implementation of the Hungarian method that selects the best label permutation that minimizes the
    misclassification error
    Parameters
    ----------
    true_lab: array as output from the clustering algorithm (n_samples, )
    pred_lab: array as output from the classification algorithm (n_samples, )

    Returns
    -------
    pred_perm: array of permuted labels (n_samples, )
    """
    nclass = len(set(true_lab))
    nobs = len(true_lab)
    wmat = np.zeros((nclass, nclass))
    for lab in range(nclass):
        for plab in range(lab, nclass):
            n_intersec = len(set(np.transpose(np.argwhere(true_lab == lab))[0]).intersection(
                set(np.transpose(np.argwhere(pred_lab == plab))[0])))
            w = (nobs - n_intersec) / nobs
            if lab == plab:
                wmat[lab, plab] = w
            else:
                wmat[lab, plab] = w
                n_intersec = len(set(np.transpose(np.argwhere(true_lab == plab))[0]).intersection(
                    set(np.transpose(np.argwhere(pred_lab == lab))[0])))
                w = (nobs - n_intersec) / nobs
                wmat[plab, lab] = w
    new_pred_lab = list(linear_sum_assignment(wmat)[1])
    # print(f'Recode: {new_pred_lab}')
    # print(f'From matrix: {wmat}')
    pred_perm = np.array([new_pred_lab.index(i) for i in pred_lab])

    return pred_perm
