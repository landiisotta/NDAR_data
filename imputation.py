import re
import pandas as pd
import utils as ut
import logging
from sklearn.impute import KNNImputer


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
    for k, df in ins_dict.items():
        if k == 'srs':
            chk_col = ['aware_t_score', 'cog_t_score',
                       'comm_t_score', 'motiv_t_score',
                       'manner_t_score']
            bool_df = df[chk_col].isna().astype(int)
            vect_count = bool_df.apply(sum, axis=1)
            drop_subj = vect_count.loc[vect_count >= 5].index
            ins_dict[k] = ins_dict[k].drop(drop_subj)
        elif k == 'ados':
            ins_dict[k] = ins_dict[k].drop(['B2', 'B3'], axis=1)
        elif k == 'mullen':
            ins_dict[k] = ins_dict[k].drop(['scoresumm_gm_t_score'], axis=1)
        elif k == 'vineland':
            ins_dict[k] = ins_dict[k].drop(['written_vscore'], axis=1)
        idx = []
        for s in ['P3', 'P4', 'P5']:
            idx += ins_dict[k].loc[ins_dict[k].interview_period == s].index.tolist()
        ins_dict[k] = ins_dict[k].drop(idx)
        nobs = ins_dict[k].shape[0]
        for col in ins_dict[k].columns:
            if not re.search('interview|sex|resp|relat', col):
                perc = (sum(ins_dict[k][[col]].isna().astype(int).sum()) / nobs) * 100
                if perc > max_perc:
                    max_perc = perc

    if max_perc < missing_perc * 100:
        logging.info(f'The maximum percentage of missing information detected from '
                     f'instrument datasets is {max_perc}')
    else:
        logging.warning(f'At leas one feature exceeds the maximum percentage of '
                        f'missing information, which is set at {missing_perc * 100}%')
    return ins_dict


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
