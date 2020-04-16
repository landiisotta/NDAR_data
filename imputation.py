import re
import pandas as pd
import utils as ut
import logging
from sklearn.impute import KNNImputer
import numpy as np
import math
from collections import namedtuple
from scipy import stats


def prepare_imputation(ins_dict, missing_perc=0.35):
    """
    This function
    - excludes subjects from SRS that completely miss subscale scores;
    - exclude GM dimension from Mullen, B2/3 from ADOS, and written score form Vineland;
    - reduce datasets to only P1-P2 interview period [TBD if we add Wechsler subscales].

    Parameters
    ----------
    ins_dict: dictionary of instruments dataframes
    missing_perc: float
        maximum amount of missing information allowed
    Returns
    -------
    dictionary with modified dataframes
    """
    max_perc = 0
    new_dict = {}
    for k, df in ins_dict.items():
        if k == 'srs':
            chk_col = ['aware_t_score', 'cog_t_score',
                       'comm_t_score', 'motiv_t_score',
                       'manner_t_score']
            bool_df = df[chk_col].isna().astype(int)
            vect_count = bool_df.apply(sum, axis=1)
            drop_subj = vect_count.loc[vect_count >= 5].index
            new_dict[k] = df.drop(drop_subj)
        elif k == 'ados':
            new_dict[k] = df.drop(['B2', 'B3'], axis=1)
        elif k == 'mullen':
            new_dict[k] = df.drop(['scoresumm_gm_t_score'], axis=1)
        elif k == 'vineland':
            new_dict[k] = df.drop(['written_vscore'], axis=1)
        else:
            new_dict[k] = df
        idx = []
        for s in ['P3', 'P4', 'P5']:
            idx += new_dict[k].loc[new_dict[k].interview_period == s].index.tolist()
        new_dict[k] = new_dict[k].drop(idx)
        perc = _check_na_perc(new_dict[k])
        if perc > max_perc:
            max_perc = perc
    if max_perc < missing_perc * 100:
        logging.info(f'The maximum percentage of missing information detected from '
                     f'instrument datasets is {max_perc}')
    else:
        logging.warning(f'At leas one feature exceeds the maximum percentage of '
                        f'missing information, which is set at {missing_perc * 100}%')
    return new_dict


def impute_by_age(train_df, test_df):
    """
    Function that perform missing data imputation
    on both train and test stratified by interview period.
    P1: [0; 30m]
    P2: (30; 72]
    P3: (72; 156]
    P4: (156; 204]
    P5: >204

    Parameters
    ----------
    train_df: dataframe
    test_df: dataframe
    Returns
    ------
    imputed dataframe train
    imputed dataframe test
    """
    knnimpute = KNNImputer(n_neighbors=ut.neighbors)
    col_n = [nc for nc in train_df.columns if not re.search('subjectkey|interview|respon|relation', nc)]
    new_dict_tr, new_dict_ts = {}, {}
    for yr in range(len(train_df.interview_period.unique())):
        exp_tr = train_df.interview_period == ''.join(['P', str(yr + 1)])
        exp_ts = test_df.interview_period == ''.join(['P', str(yr + 1)])
        tmp_tr = train_df.loc[exp_tr].copy()
        tmp_ts = test_df.loc[exp_ts].copy()
        tmp_tr[col_n] = knnimpute.fit_transform(tmp_tr[col_n])
        tmp_ts[col_n] = knnimpute.transform(tmp_ts[col_n])
        new_dict_tr[''.join(['P', str(yr + 1)])] = tmp_tr
        new_dict_ts[''.join(['P', str(yr + 1)])] = tmp_ts
    new_tr = pd.concat([df for df in new_dict_tr.values()])
    new_ts = pd.concat([df for df in new_dict_ts.values()])
    return new_tr, new_ts


def imputation_eval(df, ins_col_dict, alpha, n_iter, n_neighbors):
    """
    This function evaluates the KNN imputation technique with simulates MCAR
    at desired alpha frequency. Raw bias (RB), percent bias (PB), coverage rate (CR)
    at 95%, average width (AW), root mean squared error (RMSE) are returned as suggested
    in https://stefvanbuuren.name/fimd/sec-evaluation.html#sec:evaluationcriteria

    RB should be close to zero, PB < 5%, CR should be > 0.95, AW should be as small as
    possible, RMSE is a compromise between bias and variance.

    In this case, the estimated value Q is the mean over all features.

    Parameters
    ----------
    df: dataframe
    ins_col_dict: dictionary of instrument (keys) and column names (values),
        according to df order
    alpha: float
        frequency of MCAR information
    n_iter: int
        number of iterations
    n_neighbors: int number of neighbors to consider for KNNImputer
    """
    imputation_scores = namedtuple('imputation_scores', ['RB', 'PB', 'CR', 'AW', 'RMSE'])
    imputer = KNNImputer(n_neighbors=n_neighbors)
    dict_dfmis = {n: imputer.fit_transform(_mcar_dataset(df, ins_col_dict, alpha))
                  for n in range(n_iter)}
    true_mean = np.mean(df.apply(np.mean, axis=0))
    sample_mean = np.mean([np.mean(np.mean(dict_dfmis[n], axis=0)) for n in range(n_iter)])
    count = [0] * df.shape[1]
    awd_vect = [0] * df.shape[1]
    for n in range(n_iter):
        ci_vect = [_confint(dict_dfmis[n][:, idx]) for idx in range(df.shape[1])]
        for idx, m in enumerate(df.apply(np.mean, axis=0)):
            awd_vect[idx] += ci_vect[idx][1] - ci_vect[idx][0]
            if ci_vect[idx][0] < m < ci_vect[idx][1]:
                count[idx] += 1
    count_perc = [c / n_iter for c in count]
    awd_perc = [c / n_iter for c in awd_vect]
    cr = np.mean(count_perc)
    aw = np.mean(awd_perc)
    rmse = np.sqrt((sample_mean - true_mean) ** 2)
    scores = imputation_scores(RB=sample_mean - true_mean,
                               PB=100 * (abs((sample_mean - true_mean) / true_mean)),
                               CR=cr,
                               AW=aw, RMSE=rmse)
    return scores


"""
Private functions
"""


def _check_na_perc(df):
    nobs = df.shape[0]
    max_perc = 0
    for col in df.columns:
        if not re.search('interview|sex|resp|relat', col):
            perc = (sum(df[[col]].isna().astype(int).sum()) / nobs) * 100
            if perc > max_perc:
                max_perc = perc
    return max_perc


def _mcar_dataset(df, ins_col_dict, alpha=0.20):
    """
    Function that takes as input a complete dataframe
    and returns a dataframe with MCAR values according
    to (Brand, J., P., L., 1999) imputation theory.
    Parameters
    ----------
    df: dataframe
    ins_col_dict: dict, keys=instrument names, values=instrument features
        ordered according to dataframe feature order
    alpha: percentage of missing information
    Returns
    -------
    dataframe
    """
    col_names = np.concatenate(np.array([name for name in ins_col_dict.values()]))
    if not np.array_equal(col_names, df.columns.to_numpy()):
        logging.warning('Column features do not match the order of dictionary values')
    len_col = [len(col) for col in ins_col_dict.values()]
    rand_mat = np.random.randint(0, 2, (df.shape[0], len(ins_col_dict.keys())))
    uniq_patt = np.unique(rand_mat, axis=0)
    sample_idx = []
    for u in uniq_patt:
        idx_list = [idx for idx, el in enumerate(rand_mat) if np.array_equal(u, el)]
        sample_idx.extend(np.random.choice(idx_list,
                                           math.ceil(alpha * len(idx_list))).tolist())
    df_cp = df.copy()
    for idx in sample_idx:
        df_cp.iloc[idx] = df.iloc[idx].mask([bool(a) for a in np.repeat(rand_mat[idx],
                                                                        len_col)],
                                            None)
    return df_cp


def _confint(vect):
    """
    Parameters
    ----------
    vect: list (of performance scores)
    Returns
    ------
    tuple : mean and error
    """
    error = stats.t.ppf(1 - (0.05 / 2), len(vect) - 1) * (np.std(vect) / math.sqrt(len(vect)))
    mean = np.mean(vect)
    return mean - error, mean + error
