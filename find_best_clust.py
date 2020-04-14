from sklearn.model_selection import StratifiedKFold, KFold
from relative_validation import RelativeValidation
from visualization import plot_metrics, plot_dendrogram
import logging
import numpy as np


class FindBestClustCV(RelativeValidation):
    """
    Child class of RelativeValidation

    It performs cross validation on the training set to
    select the best number of clusters, i.e., the number that minimizes the
    misclassification error.

    Initialized with:
    S: inherited classification object
    C: inherited clustering object
    nfold: number of fold for CV (int)
    strat_vect: array, pd Series for stratification (strings, objects or integers)
                (n_samples, )
    n_clust: max number  of clusters to look for (int)
    min_cl: int, minimum number of clusters

    Methods
    -------
    search_best_clust()
    evaluate()
    """

    def __init__(self, nfold, n_clust, min_cl, S, C):
        super().__init__(S, C)
        self.nfold = nfold
        self.n_clust = n_clust
        self.min_cl = min_cl

    def search_best_nclust(self, train_data, rand_iter=None, strat_vect=None):
        """
        This method should be used after the dataset has been split into training and test
        by train_test_split() from sklearn.model_selection module. It is possible
        to stratify the split and return X_tr/ts, y_tr/ts that can then be used to stratify
        X_tr during CV (when strat_vect is not None).
        Parameters
        ----------
        train_data: array of training data (n_samples, n_features).
                    Include only numeric features in the process.
                    Drop missing information before calling the method.
        rand_iter: int
            Number of randomized labels iterations.
            Default: None
        strat_vect: array for stratification. Default: None.

        Returns
        -------
        performance: dictionary of misclassification error in train and validation
                     for each number of clusters and averages.
        bestcl: best number of clusters (i.e., that that minimizes the averaged
               misclassification error).
        """
        misclass_risk = {'train': {},
                         'val': {}}
        stab = {'train': {},
                'val': {}}
        norm_stab = {'train': {},
                     'val': {}}
        cv_score = {'train': {},
                    'val': {}}

        for ncl in range(self.min_cl, (self.n_clust + 1)):
            if strat_vect is not None:
                kfold = StratifiedKFold(n_splits=self.nfold)
                fold_gen = kfold.split(train_data, strat_vect)
            else:
                kfold = KFold(n_splits=self.nfold)
                fold_gen = kfold.split(train_data)
            rndvect_tr, rndvect_val = [], []
            append_tr = rndvect_tr.append
            append_ts = rndvect_val.append
            for tr_idx, val_idx in fold_gen:
                tr_set, val_set = train_data.reset_index().iloc[tr_idx], train_data.reset_index().iloc[val_idx]
                tr_set.index = tr_set.subjectkey
                tr_set.drop('subjectkey', axis=1, inplace=True)
                val_set.index = val_set.subjectkey
                val_set.drop('subjectkey', axis=1, inplace=True)
                # tr_set, val_set = train_data[tr_idx, :], train_data[val_idx, :]
                self.clust_method.n_clusters = ncl
                miscl, predmodel, tr_labels = super().train(tr_set)
                misclass_risk['train'].setdefault(ncl, list()).append(miscl)
                miscl_val, val_labels = super().test(val_set, predmodel)
                misclass_risk['val'].setdefault(ncl, list()).append(miscl_val)
                if rand_iter is not None:
                    rndmisc_mean_tr, rndmisc_mean_val = super().train_eval_rnd(tr_set, val_set, rand_iter,
                                                                               tr_labels['clustering'],
                                                                               val_labels['clustering'])
                    append_tr(rndmisc_mean_tr)
                    append_ts(rndmisc_mean_val)
            stab['train'][ncl] = np.mean(misclass_risk['train'][ncl])
            stab['val'][ncl] = np.mean(misclass_risk['val'][ncl])
            cv_score['train'][ncl] = [mr for mr in misclass_risk['train'][ncl]/np.mean(rndvect_tr)]
            cv_score['val'][ncl] = [mr for mr in misclass_risk['val'][ncl] / np.mean(rndvect_tr)]
            if rand_iter is not None:
                if np.mean(rndvect_tr) != 0:
                    norm_stab['train'][ncl] = stab['train'][ncl] / np.mean(rndvect_tr)
                else:
                    norm_stab['train'][ncl] = stab['train'][ncl]
                norm_stab['val'][ncl] = stab['val'][ncl] / np.mean(rndvect_val)
            else:
                norm_stab['train'][ncl] = stab['train'][ncl]
                norm_stab['val'][ncl] = stab['val'][ncl]
        val_score = np.array(list(norm_stab['val'].values()))
        bestscore = min(val_score)
        # select the cluster with the minimum misclassification error
        # and the maximum number of clusters
        bestcl = max(np.transpose(np.argwhere(val_score == bestscore))[0]) + self.min_cl

        performance = {'stability_metrics': norm_stab}

        plot_metrics(cv_score)
        return performance, bestcl, cv_score

    def evaluate(self, train_data, test_data, best_nclust):
        """
        Parameters
        ----------
        train_data: array of training data (n_samples, n_features)
        test_data: array of test data (n_samples, n_features)
        best_nclust: int number of clusters

        Returns
        -------
        tr_misc: float misclassification error on test set
        ts_misc: float misclassification error on training set
        labels: labels for clustering and classification from train/test sets
                array of integers (n_samples, )
        """
        self.clust_method.n_clusters = best_nclust
        tr_misc, modelfit, labels_tr = super().train(train_data)
        ts_misc, labels_ts = super().test(test_data, modelfit)

        # self.clust_method.n_clusters = None
        # self.clust_method.distance_threshold = 0
        # model = self.clust_method.fit(test_data)
        # plot_dendrogram(model)

        labels = {'train': {'clustering': labels_tr['clustering'],
                            'classification': labels_tr['classification']},
                  'test': {'clustering': labels_ts['clustering'],
                           'classification': labels_ts['classification']}}
        logging.info(f"Training ACC: {(1 - tr_misc)}\n Test ACC: {(1 - ts_misc)}")
        return tr_misc, ts_misc, labels
