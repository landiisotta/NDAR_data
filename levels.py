import pandas as pd
import os
import utils as ut
import logging
import re

"""
This module has an execution time of:
1min 41s ± 1.4 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

It can probably been improved.
"""


def create_map(codebook_file, scaling=False):
    """
    Function that returns a dictionary which stores the characteristics of
     instrument levels from single items/subscales to general content areas.
     If requested, the function also returns the maximum score for each subscale
     for scaling purposes
     Parameters
     ----------
     codebook_file: str
        file's name. The file stores the level specifics, (e.g., for ADI-R the content area A comes from
        the sum(A1, A2, A3)).
     scaling: bool
        If True, a dictionary with the maximum score for each item is added to the tuple
     Returns
     -------
     list
        item names as in NDAR
     dictionaries
        subscale_map, scale_map, (if scaling) max_val
    """
    cdbook = pd.read_excel(os.path.join(ut.data_folder,
                                        codebook_file))
    subscale_map = {}
    scale_map = {}
    max_val = {}
    item_vect = []
    for _, row in cdbook.iterrows():
        subscale_map.setdefault(row['subscale'],
                                list()).append(row['NDARname'])
        scale_map.setdefault(row['scale'],
                             set()).add(row['subscale'])
        item_vect.append(row['NDARname'])
        if scaling:
            max_val[row['NDARname']] = row['max_score']
    if scaling:
        return item_vect, subscale_map, scale_map, max_val
    return item_vect, subscale_map, scale_map


def create_datalevels(df_instrument, item_vect,
                      subscale_map, scale_map,
                      recode=None, max_val=None):
    """
    The function returns a dictionary with three dataframes corresponding to instrument scores respect to
    three levels of depth
    Parameters
    ----------
    df_instrument: dataframe
        Instrument dataframe
    item_vect: list
        List of item names as in NDAR
    subscale_map: dictionary
        As returned by create_map(), with level 1 - level 2 correspondence
    scale_map: dictionary
        As returned by create_map(), with level 2 - level 3 correspondence
    recode: int
        If not None, the integer from which scores should be replaced with 0
    max_val: if not None, dictionary
        As returned by create_map(), with item max scores
    Returns
    -------
    dictionary
    """
    level_dict = {}
    # Select only existent features (e.g., all ados modules are considered at once)
    item_vect = list(set(item_vect).intersection(set(df_instrument.columns)))

    cnrow = df_instrument.shape[0]  # store current number of subjects
    logging.info(f'Starting processing of {df_instrument.columns[1].strip("_id")}'
                 f' instrument. Current number of subjects {cnrow}')
    # recode selected features if necessary
    if recode is not None:
        for fcol in item_vect:
            df_instrument.loc[df_instrument[fcol] >= 999, fcol] = None
            df_instrument.loc[df_instrument[fcol] < 0, fcol] = None  # -999=NA/missing
            df_instrument.loc[df_instrument[fcol] > recode, fcol] = 0
    # drop subjects with completely missing features or all entries equal zero
    drop_subj = _drop_subj(df_instrument[item_vect])
    df_instrument = df_instrument.drop(drop_subj, axis=0)
    logging.info(f'Dropped {cnrow - df_instrument.shape[0]} '
                 f'subjects with completely missing information')

    # drop duplicates (based on interview age) from dataframe and feature scores
    cnrow = df_instrument.shape[0]
    logging.info(f'Current number of subjects {cnrow}')
    logging.info("Dropping duplicates with the same interview age.")
    df_instrument = df_instrument.reset_index().drop_duplicates(['subjectkey',
                                                                 'interview_age'] +
                                                                item_vect,
                                                                keep='last')
    df_instrument.index = df_instrument['subjectkey']
    df_instrument.drop('subjectkey', axis=1, inplace=True)
    logging.info(f'Dropped {cnrow - df_instrument.shape[0]} subjects')
    cnrow = df_instrument.shape[0]  # store current number of subjects
    logging.info(f'Current number of subjects: {cnrow} '
                 f'-- Number of features: {df_instrument.shape[1]}')

    # create level 1 dataframe
    logging.info("Creating level 1 dataframe...")
    level_dict['level-1'] = df_instrument[['interview_age', 'sex',
                                           'phenotype', 'site'] + item_vect]
    # Create level 2 scaled scores
    # list that stores the subject keys of those
    # subjects that miss a subscale completely
    # (except ados-module1 subscales A3, B2, B3)
    add_subj = True
    mask = pd.Series()
    logging.info("Creating level 2 dataframe...")
    level_dict['level-2'] = pd.DataFrame(columns=subscale_map.keys())
    for sid, row in level_dict['level-1'].iterrows():
        tmp_sr = pd.Series()
        for k, val in subscale_map.items():
            select = [it for it in val if it in row.dropna().index]
            if len(select) == 0:
                if len(set(val).intersection(set(row.index))) == 0:
                    tmp_sr[k] = None
                else:
                    mask = mask.append(pd.Series([False], index=[sid]))
                    add_subj = False
                    break
            else:
                if max_val is not None:
                    tmp_sr[k] = sum(row[select]) / sum([max_val[col] for col in select])
                else:
                    tmp_sr[k] = sum(row[select])
        if add_subj:
            mask = mask.append(pd.Series([True], index=[sid]))
            level_dict['level-2'] = level_dict['level-2'].append(tmp_sr,
                                                                 ignore_index=True)
        add_subj = True
    level_dict['level-2'].index = mask.loc[mask].index
    level_dict['level-2'].index.name = 'subjectkey'

    logging.info(f'Dropped {(cnrow - level_dict["level-2"].shape[0])} -- '
                 f'Current number of subjects {level_dict["level-2"].shape[0]}')

    # Update level-1 dataframe
    # level_dict['level-1'].drop(drop_subj, inplace=True, axis=0)
    level_dict['level-1'] = level_dict['level-1'].loc[mask, :]
    level_dict['level-2'][['interview_age',
                           'sex',
                           'phenotype',
                           'site']] = level_dict['level-1'].loc[:, ['interview_age',
                                                                    'sex',
                                                                    'phenotype',
                                                                    'site']]
    # Create level 3 scaled scores
    logging.info("Creating level 3 dataframe...")
    level_dict['level-3'] = pd.DataFrame(columns=scale_map.keys())
    for sid, row in level_dict['level-2'].iterrows():
        tmp_sr = pd.Series()
        for k, val in scale_map.items():
            select = [it for it in val if it in row.dropna().index]
            if max_val is not None:
                tmp_sr[k] = sum(row[select]) / len(select)
            else:
                tmp_sr[k] = sum(row[select])
        level_dict['level-3'] = level_dict['level-3'].append(tmp_sr,
                                                             ignore_index=True)

    level_dict['level-3'].index = level_dict['level-2'].index
    level_dict['level-3'].index.name = 'subjectkey'

    level_dict['level-3'][['interview_age',
                           'sex',
                           'phenotype',
                           'site']] = level_dict['level-2'].loc[:, ['interview_age',
                                                                    'sex',
                                                                    'phenotype',
                                                                    'site']]
    subjkeys = level_dict['level-3'].index
    logging.info(f'\n Number of unique '
                 f'subjects {len(subjkeys.unique())}')
    rep_sid = subjkeys[subjkeys.duplicated(keep=False)]
    logging.info(f'{len(rep_sid)} multiple assessments of '
                 f' {len(rep_sid.unique())} subjects\n\n')

    return level_dict


def mullen_levels(ins_df, col_names):
    """
    Function that creates a unique level for Mullen Scales of
    Early Learning. It selects scales:
    (1) Gross motor functions;
    (2) Visual Reception;
    (3) Fine Motor Skills;
    (4) Receptive Language;
    (5) Expressive Language;
    (a) Early Learning Composite score from the
    cognitive scales (i.e., 2-5).
    Parameters
    ----------
    ins_df: Mullen dataframe
    col_names: list
        list with the names of the columns to consider
    Returns
    -------
    dict: unique-level dictionary
    """
    for cn in col_names[4:]:
        ins_df.loc[ins_df[cn] >= 999, cn] = None  # 999=NA/missing
        ins_df.loc[ins_df[cn] < 0, cn] = None  # -999=NA/missing
    mullen_df = ins_df[col_names]
    # Recode 999=None
    # Drop duplicate subjects with the same assessment age and feature scores
    cnrow = mullen_df.shape[0]  # store current number of subjects
    logging.info(f'Starting processing of {ins_df.columns[1].strip("_id")}'
                 f' instrument. Current number of subjects {cnrow}')
    logging.info("Dropping duplicates with the same interview age"
                 " and subjects with completely missing information.")
    # Drop subjects with completely missing information
    drop_subj = _drop_subj(mullen_df[col_names[4:]])
    mullen_df = mullen_df.drop(drop_subj, axis=0)
    mullen_df = mullen_df.reset_index().drop_duplicates(['subjectkey',
                                                         'interview_age'] +
                                                        col_names[4:],
                                                        keep='last')
    mullen_df.index = mullen_df['subjectkey']
    mullen_df.drop('subjectkey', axis=1, inplace=True)
    logging.info(f'Dropped {cnrow - mullen_df.shape[0]} subjects')
    cnrow = mullen_df.shape[0]  # store current number of subjects
    logging.info(f'Current number of subjects: {cnrow} '
                 f'-- Number of features: {mullen_df.shape[1]}\n\n')
    return {'level-unique': mullen_df}


def vineland_levels(ins_dict, col_names, mergecol):
    """
    It creates a unique-level dataframe merging different versions of Vineland instrument,
    namely Vineland I, Vineland II (Parent and Caregiver rating), Vineland II (rating form),
    and Vineland 3.
    Parameters
    ----------
    ins_dict: dictionary
        dictionary of datframes
    col_names: list
        columns to select
    mergecol: dictionary
        columns to merge are displayed as keys and values.
        Keys become the new column name
    Returns
    -------
    dictionary
        dictionary with one level storing a dataframe obtained merging all
        Vineland Versions and including the variables of interest.
    """
    ins_dict_vin = {k: ins_dict[k] for k in ins_dict.keys() if re.search('vineland', k)}
    rid_dict = {}
    for k, df in ins_dict_vin.items():
        rid_dict[k] = df[df.columns.intersection(col_names)].copy()
        for c1, c2 in mergecol.items():
            if c1 in list(rid_dict[k].columns) and c2 in list(rid_dict[k].columns):
                rid_dict[k].loc[:, c1] = rid_dict[k].loc[:, c1].fillna(rid_dict[k].loc[:, c2])
                rid_dict[k] = rid_dict[k].drop(c2, axis=1)
            else:
                rid_dict[k] = rid_dict[k].rename(columns={c2: c1})
    for rm in mergecol.values():
        col_names.remove(rm)
    df_concat = pd.concat([rid_dict[k][col_names] for k in rid_dict.keys()])
    for cn in col_names[4:]:
        df_concat.loc[df_concat[cn] >= 999, cn] = None  # 999=NA/missing
        df_concat.loc[df_concat[cn] < 0, cn] = None  # -999=NA/missing
        if cn == 'relationship':
            df_concat.loc[df_concat[cn] == 27, cn] = None  # 27=Missing data

    # Drop duplicate subjects with the same assessment age and feature scores
    cnrow = df_concat.shape[0]  # store current number of subjects
    logging.info(f'Processing of merged {[df.columns[1].strip("_id") for df in ins_dict_vin.values()]}'
                 f' instrument. Current number of subjects {cnrow}')
    logging.info("Dropping duplicates with the same interview age and relationship"
                 " and subjects with completely missing information.")
    # Drop subjects with completely missing information
    drop_subj = _drop_subj(df_concat[[c for c in col_names[5:] if c in df_concat.columns]])
    df_concat = df_concat.drop(drop_subj, axis=0)
    df_concat = df_concat.reset_index().drop_duplicates(['subjectkey',
                                                         'interview_age',
                                                         'relationship'] +
                                                        [c for c in col_names[5:]
                                                         if c in df_concat.columns],
                                                        keep='last')
    df_concat.index = df_concat['subjectkey']
    df_concat.drop('subjectkey', axis=1, inplace=True)
    logging.info(f'Dropped {cnrow - df_concat.shape[0]} subjects')
    cnrow = df_concat.shape[0]  # store current number of subjects
    logging.info(f'Current number of subjects: {cnrow} '
                 f'-- Number of features: {df_concat.shape[1]}\n\n')
    return {'level-unique': df_concat}


def srs_levels(ins_dict, col_names, mergecol, renamecol):
    """
    Returns a unique dataframe with subscales and scales for SRS.
    Merges multiple SRS datasets into one if needed.
    Parameters
    ----------
    ins_dict: dictionary
        dictionary of dataframes
    col_names: list
        list of names of the features to consider
    mergecol: dictionary
        columns to merge into one, replace NAs (e.g., male/female subscales)
    renamecol: dict
        dictionary to uniform column names and merge dataframes
    """
    ins_dict_srs = {k: ins_dict[k] for k in ins_dict.keys() if re.search('srs', k)}
    srs_all = {}
    for name, df in ins_dict_srs.items():
        av_col = [c for c in col_names if c in df.columns.intersection(col_names)]
        srs = df[av_col].copy()
        for k, val in mergecol.items():
            if k in srs.columns:
                srs.loc[:, k] = srs.loc[:, k].fillna(srs.loc[:, val])
                srs.drop(val, axis=1, inplace=True)
        srs = srs.rename(columns={c1: c2 for c1, c2 in renamecol.items() if c1 in srs.columns})
        av_col = [c for c in col_names if c in srs.columns.intersection(col_names)]
        for col in av_col[4:]:
            if col == 'respond':
                srs.loc[srs[col] == 999, col] = None
            else:
                srs.loc[srs[col] == '>90', col] = 91
                srs.loc[srs[col] == '999', col] = None
                srs.loc[srs[col] == '"', col] = None
                srs.loc[srs[col] == '.', col] = None
                srs.loc[srs[col] == '?90', col] = 90
                srs = srs.astype({col: 'float64'})
        cnrow = srs.shape[0]  # store current number of subjects
        logging.info(f'Processing of {df.columns[1].strip("_id")}'
                     f' instrument. Current number of subjects {cnrow}')
        logging.info("Dropping duplicates with the same interview age and relationship"
                     " and subjects with completely missing information.")
        drop_subj = _drop_subj(srs[[c for c in av_col[5:]]])
        srs = srs.drop(drop_subj, axis=0)
        srs = srs.reset_index().drop_duplicates(['subjectkey',
                                                 'interview_age',
                                                 'respond'] +
                                                [c for c in av_col[4:]],
                                                keep='last')
        srs.index = srs['subjectkey']
        srs.drop('subjectkey', axis=1, inplace=True)
        logging.info(f'Dropped {cnrow - srs.shape[0]} subjects')
        cnrow = srs.shape[0]  # store current number of subjects
        logging.info(f'Current number of subjects: {cnrow} '
                     f'-- Number of features: {srs.shape[1]}')
        srs_all[name] = srs

    logging.info(f'Concatenating {[df.columns[1].strip("_id") for df in ins_dict_srs.values()]}\n')
    srs_concat = pd.concat([srs_all[k] for k in srs_all.keys()])
    logging.info('Order entries by interview age and remove duplicated rows with the same respondent '
                 '(or NA)')
    srs_concat.sort_values(by=['interview_age'], inplace=True)
    srs_concat = srs_concat.reset_index().drop_duplicates(['subjectkey',
                                                           'respond'],
                                                          keep='first')
    dupid = srs_concat.loc[srs_concat.subjectkey.duplicated()].subjectkey
    for idx, row in srs_concat.iterrows():
        if row.subjectkey in list(dupid) and str(row.respond) == 'nan':
            srs_concat.drop(idx, axis=0, inplace=True)
    srs_concat.index = srs_concat.subjectkey
    srs_concat.drop('subjectkey', axis=1, inplace=True)
    logging.info(f'Number of subjects: {srs_concat.shape[0]} -- '
                 f'Number of unique subjects: {srs_concat.index.unique().shape[0]}\n\n')

    return {'level-unique': srs_concat}


def arrange_duplicates(level_dict, caregiver=False, pheno=None):
    """
    Function that drops duplicates keeping the first observation (i.e., min interview age). The dictionary
    returned, adds to each dataframe the phenotype column (if pheno is not None)
    Parameters
    ----------
    level_dict: dictionary
        dictionary storing different dataframes according to instrument depth levels
    caregiver: bool
        for instruments such as Vineland it orders the observation according to interview age and relationship
        and it keep the first
    pheno: pandas dataframe, default None
        as output by dataset read_data() with asd_diag not None
    Returns
    -------
    dictionary
    """
    new_level_dict = {}
    for k, df in level_dict.items():
        if caregiver:
            try:
                new_level_dict[k] = df.sort_values(by=['interview_age',
                                                       'relationship'],
                                                   axis=0)
            except KeyError:
                new_level_dict[k] = df.sort_values(by=['interview_age',
                                                       'respond'],
                                                   axis=0)
        else:
            new_level_dict[k] = df.sort_values(by=['interview_age'],
                                               axis=0)
        dupl_keys = new_level_dict[k].index.duplicated(keep='first')
        new_level_dict[k] = new_level_dict[k].loc[~dupl_keys]

        n_obs_dropped = df.shape[0] - new_level_dict[k].shape[0]
        n_subj = len(df.sort_values(by=['interview_age'],
                                    axis=0).loc[dupl_keys].index.unique())
        logging.info(f'Level {k.lstrip("level-")}')
        logging.info(f'Observation dropped {n_obs_dropped}')
        logging.info(f'Observation kept {n_subj}')
        logging.info(f'Unique number of subjects {len(new_level_dict[k].index.unique())}\n\n')

        if pheno is not None:
            dupl_pheno = pheno.index.duplicated(keep='first')
            pheno = pheno.loc[~dupl_pheno]
            new_level_dict[k]['phenotype'] = pheno.loc[new_level_dict[k].index].phenotype
            new_level_dict[k]['phenotype_description'] = pheno.loc[new_level_dict[k].index].phenotype_description
    return new_level_dict


def _drop_subj(df):
    """
    Function that takes as input a dataframe (only the desired features) and return a list of subjects to drop.
    Subjects to drop are those for which all entries are either None or 0.
    Parameters
    ----------
    df: dataframe
    Returns
    -------
    list
    """
    drop_subj = []
    for sid, row in df.iterrows():
        if sum(row.dropna()) == 0:
            drop_subj.append(sid)
        elif row.count() == 0:
            drop_subj.append(sid)
    return drop_subj


def _merge_ins():
    # Function that merges two or more dataframes
    # uniforming feature names (e.g., ados-1m1 and ados-2m1;
    # vineland survey and parent)
    return
