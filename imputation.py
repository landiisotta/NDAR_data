import re
import pandas as pd
import utils as ut
import logging
from sklearn.impute import KNNImputer
import numpy as np
import math
from collections import namedtuple
from scipy import stats
from scipy.stats.mstats import mquantiles_cimj
from create_dataset import concatenate_all_ins
import pickle as pkl


def prepare_imputation(ins_dict, age_period='P2', missing_perc=0.35):
    """
    This function
    - excludes subjects from SRS that completely miss subscale scores;
    - exclude GM dimension from Mullen, B2/3 from ADOS, and written score form Vineland;
    - reduce datasets to only P1-P2 interview period [TBD if we add Wechsler subscales].
    **MODIFIED**
    !!We consider only subjects with user defined interview age, default = P2
     and we retrieve longitudinal information if available

    Only subjects with
    Parameters
    ----------
    ins_dict: dictionary of instruments dataframes
    age_period: str
    missing_perc: float
        maximum amount of missing information allowed
    Returns
    -------
    dictionary with modified dataframes
    """
    max_perc = 0
    new_dict = {}
    all_perc = []
    for k, df in ins_dict.items():
        new_dict[k] = df.loc[df.interview_period == age_period].copy()
        if k == 'srs':
            chk_col = ['aware_t_score', 'cog_t_score',
                       'comm_t_score', 'motiv_t_score',
                       'manner_t_score']
            bool_df = new_dict[k][chk_col].isna().astype(int)
            vect_count = bool_df.apply(sum, axis=1)
            drop_subj = vect_count.loc[vect_count >= 5].index
            new_dict[k].drop(drop_subj, inplace=True)
        elif k == 'ados':
            new_dict[k].drop(['B2', 'B3'], axis=1, inplace=True)
        elif k == 'mullen':
            new_dict[k].drop(['scoresumm_gm_t_score'], axis=1, inplace=True)
        elif k == 'vineland':
            new_dict[k].drop(['written_vscore'], axis=1, inplace=True)
        perc, m_perc = _check_na_perc(new_dict[k])
        all_perc.append(m_perc)
        if perc > max_perc:
            max_perc = perc
    if max_perc < missing_perc * 100:
        logging.info(f'The maximum percentage of missing information detected from '
                     f'instrument datasets is {max_perc}')
        new_dict[k]['countna'] = np.zeros((new_dict[k].shape[0],))
    else:
        logging.warning(f'At leas one feature exceeds the maximum percentage of '
                        f'missing information, which is set at {missing_perc * 100}%')
        logging.info(f'Searching for subjects with percentage of missing information > {missing_perc}')
        for k in new_dict.keys():
            mis_count_df = pd.DataFrame(
                [new_dict[k][[c for c in new_dict[k].columns if
                              not re.search('interview|relationship|respon|intersection', c)]].loc[
                     gui].isna().astype(int)
                 for gui in
                 new_dict[k].index])
            outidx = []
            countna = []
            for idx, row in mis_count_df.iterrows():
                naperc = sum(row) / len(row)
                if naperc > missing_perc:
                    outidx.append(idx)
                    countna.append(2)
                elif naperc < 0.15:  # missing percentages > 0.35 are dropped
                    countna.append(0)  # < 0.15 are labeled 0, > 0.15 are labeled 1
                else:
                    countna.append(1)
            new_dict[k]['countna'] = countna
            new_dict[k].drop(outidx, axis=0, inplace=True)
            logging.info(f'{k.upper()}: Dropped {len(outidx)} subjects.')
    logging.info(f'The average percentage of missing information is {np.mean(all_perc)}')
    return new_dict


def prepare_mcar_imputation(ins_dict):
    df = concatenate_all_ins(ins_dict)
    perc, m_perc = _check_na_perc(df)
    logging.info(f'Maximum percentage of missing information: {perc}%')
    logging.info(f'Average percentage of missing information {m_perc}')
    return df


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
    for yr in sorted(train_df.interview_period.unique()):
        exp_tr = train_df.interview_period == yr
        exp_ts = test_df.interview_period == yr
        tmp_tr = train_df.loc[exp_tr].copy()
        tmp_ts = test_df.loc[exp_ts].copy()
        tmp_tr[col_n] = knnimpute.fit_transform(tmp_tr[col_n])
        tmp_ts[col_n] = knnimpute.transform(tmp_ts[col_n])
        new_dict_tr[yr] = tmp_tr
        new_dict_ts[yr] = tmp_ts
    new_tr = pd.concat([df for df in new_dict_tr.values()])
    new_ts = pd.concat([df for df in new_dict_ts.values()])
    return new_tr, new_ts


def impute(train_df, test_df):
    """
    Function that perform missing data imputation
    on both train and test for a unique interview period.

    Parameters
    ----------
    train_df: dataframe feature names and interview-based names
    test_df: dataframe feature names and interview-based names
    Returns
    ------
    imputed dataframe train
    imputed dataframe test
    """
    knnimpute = KNNImputer(n_neighbors=ut.neighbors)
    col_n = [nc for nc in train_df.columns if not re.search('interview', nc)]
    col_out = [nc for nc in train_df.columns if re.search('interview', nc)]
    tmp_tr = pd.DataFrame(knnimpute.fit_transform(train_df[col_n]), columns = col_n)
    tmp_ts = pd.DataFrame(knnimpute.transform(test_df[col_n]), columns = col_n)
    tmp_tr.index = train_df.index
    tmp_ts.index = test_df.index
    for c in col_out:
        tmp_tr[c] = train_df[c]
        tmp_ts[c] = test_df[c]
    return tmp_tr, tmp_ts


def imputation_eval(df, alpha, n_iter, n_neighbors, mdp='MCAR', ins_col_dict=None):
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
    alpha: float
        frequency of MCAR information
    n_iter: int
        number of iterations
    n_neighbors: int number of neighbors to consider for KNNImputer
    mdp: str
        Missing data pattern, "MCAR", "MAR" available
    ins_col_dict: (default None), if not None dictionary
        of instrument (keys) and column names (values),
        according to df order
    Returns
    -------
    eval parameters (mean, quantiles) as namedtuple and dictionary of namedtuples
    """
    imputation_scores = namedtuple('imputation_scores', ['RB', 'PB', 'CR', 'AW'])
    quantiles = [0.25, 0.5, 0.75]
    imputer = KNNImputer(n_neighbors=n_neighbors)
    if mdp == "MCAR":
        if ins_col_dict is None:
            logging.error("Please add a dictionary of the form:"
                          " k: instrument name; value: column names.")
        dict_dfmis = {n: imputer.fit_transform(_mcar_dataset(df, ins_col_dict, alpha))
                      for n in range(n_iter)}
    else:
        dict_dfmis = {n: imputer.fit_transform(_mar_dataset(df, alpha))
                      for n in range(n_iter)}
    true_mean = np.mean(np.mean(df, axis=0))  # 1 x 1
    true_quant = np.mean(np.quantile(df, q=quantiles, axis=0), axis=1)  # 1 x 3
    sample_mean = np.mean(np.mean([np.mean(dict_dfmis[n], axis=0) for n in range(n_iter)],
                                  axis=1))  # 1 x 1
    sample_quant = np.mean(np.mean([np.quantile(dict_dfmis[n], q=quantiles,
                                                axis=0) for n in range(n_iter)], axis=2), axis=0)  # 1 x 3
    awd_vect, count = [0] * df.shape[1], [0] * df.shape[1]
    awd_vectq, count_q = [[0] * 3] * df.shape[1], [[0] * 3] * df.shape[1]
    for n in range(n_iter):
        ci_vect = [_confint(dict_dfmis[n][:, idx]) for idx in range(df.shape[1])]  # n_feat tuples 2-dim
        ci_vect_q = [mquantiles_cimj(dict_dfmis[n][:, feat], quantiles) for feat in  # n_feat tuples 2 x 3
                     range(df.shape[1])]
        for m, q in zip(enumerate(np.mean(df, axis=0)),
                        enumerate(np.quantile(df, q=quantiles, axis=0).transpose())):
            awd_vect[m[0]] += ci_vect[m[0]][1] - ci_vect[m[0]][0]
            awd_vectq[q[0]] = np.add(awd_vectq[q[0]],
                                     np.array(
                                         [ci_vect_q[q[0]][1][qidx] - ci_vect_q[q[0]][0][qidx] for qidx in range(3)]))
            count[m[0]] += int(ci_vect[m[0]][0] <= m[1] <= ci_vect[m[0]][1])
            count_q[q[0]] = np.add(count_q[q[0]], np.array(
                [int(ci_vect_q[q[0]][0][i] <= q[1][i] <= ci_vect_q[q[0]][1][i]) for i in range(3)]))

    count_perc = [c / n_iter for c in count]
    awd_perc = [c / n_iter for c in awd_vect]
    cr = np.mean(count_perc)
    aw = np.mean(awd_perc)

    count_perc_q = np.array([np.divide(c, n_iter) for c in count_q])
    awd_perc_q = np.array([np.divide(c, n_iter) for c in awd_vectq])
    cr_q = [np.mean(count_perc_q[:, idx]) for idx in range(3)]
    aw_q = [np.mean(awd_perc_q[~np.isnan(awd_perc_q[:, idx]), idx]) for idx in range(3)]

    scores_m = imputation_scores(RB=sample_mean - true_mean,
                                 PB=100 * (abs((sample_mean - true_mean) / true_mean)),
                                 CR=cr,
                                 AW=aw)
    scores_q = {q: imputation_scores(RB=sample_quant[idx] - true_quant[idx],
                                     PB=100 * (abs((sample_quant[idx] - true_quant[idx]) / true_quant[idx])),
                                     CR=cr_q[idx],
                                     AW=aw_q[idx]) for idx, q in enumerate(quantiles)}
    return scores_m, scores_q


"""
Private functions
"""


def _check_na_perc(df):
    nobs = df.shape[0]
    max_perc = 0
    perc_vect = []
    for col in df.columns:
        if not re.search('interview|sex|resp|relat', col):
            perc = (sum(df[[col]].isna().astype(int).sum()) / nobs) * 100
            perc_vect.append(perc)
            if perc > max_perc:
                max_perc = perc
    return max_perc, np.mean(perc_vect)


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
    for u in list(filter(lambda x: sum(x) != df.shape[1], uniq_patt)):
        idx_list = [idx for idx, el in enumerate(rand_mat) if np.array_equal(u, el)]
        sample_idx.extend(np.unique(np.random.choice(idx_list,
                                                     math.ceil(alpha * len(idx_list))).tolist()))
    df_cp = df.copy()
    for idx in sample_idx:
        df_cp.iloc[idx] = df.iloc[idx].mask([bool(a) for a in np.repeat(rand_mat[idx],
                                                                        len_col)],
                                            None)
    return df_cp


def _mar_dataset(df, alpha=0.20):
    """
    Function that takes as input a complete dataframe
    and returns a dataframe with MAR values according
    to (Brand, J., P., L., 1999) imputation theory.
    Parameters
    ----------
    df: dataframe
    alpha: percentage of missing information
    Returns
    -------
    dataframe
    """
    f = np.random.choice(range(df.shape[0]), df.shape[0])
    rand_mat = np.random.randint(0, 2, df.shape)[f]
    uniq_patt = np.unique(rand_mat, axis=0)
    # weights
    a_mat = np.random.randint(0, 2, df.shape)
    g_mat = np.random.randint(0, 5, (df.shape[0], 3))  # nxk with k quantiles (k=3)

    choose_idx = []
    for u in list(filter(lambda x: sum(x) != df.shape[1], uniq_patt)):
        # print(u)
        idx_list = [idx for idx, el in enumerate(rand_mat) if np.array_equal(u, el)]
        if len(idx_list) == 1:
            choose_idx.extend(idx_list)
        else:
            sample_idx = []
            for idx in idx_list:
                s = sum(a_mat[idx, :] * u * df.iloc[idx])
                theta = [0, 0.25, 0.5, 0.75, 1]
                c = np.quantile(df.iloc[idx], theta[1:4])
                ctr = sum([c > s][0])
                if ctr == 0:
                    sample_idx.append(idx)
                else:
                    num = (alpha * g_mat[idx, np.where(c > s)[0][0] - 1])
                    den = [0.25] + [(theta[n + 1] - theta[n]) * g_mat[idx][n - 1] for n in range(1, 4)]
                    frac = num / sum(den)
                    if frac > 0:
                        sample_idx.extend([idx] * int(frac * 100))
                    else:
                        sample_idx.append(idx)
            choose_idx.extend(np.unique(np.random.choice(sample_idx, len(idx_list))))

    df_cp = df.copy()
    for idx in choose_idx:
        df_cp.iloc[idx] = df.iloc[idx].mask([bool(a) for a in 1 - rand_mat[idx]],
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
