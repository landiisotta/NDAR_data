import pandas as pd
import utils as ut
from collections import namedtuple
import logging
import os
import re

# These named tuples are used to code the DSM-5 vs ADI-R/ADOS
# conversion table, as described by doi: 10.1176/appi.ajp.2012.12020276
levels_feat_scaled = namedtuple('levels_feat_scaled', ['item_vect',
                                                       'scale_map',
                                                       'subscale_map',
                                                       'max_val'])
levels_feat = namedtuple('levels_feat', ['item_vect',
                                         'scale_map',
                                         'subscale_map'])


class CreateLevels:
    """
    The CreateLevels class is composed by one method (i.e., create_map)
    and 5 static methods. The create_map method is only required when ADI-R or ADOS
    are included in the study. The static methods correspond to different instruments
    and allow to build the hierarchical taxonomies based on single item,
    subscales, and scales.
    The organization differs for each instrument (i.e., different versions
    and editions, varying column names, different skills investigated).
    """

    def __init__(self, codebook_list=None):
        self.codebook_list = codebook_list

    def create_map(self, scaling=False):
        """
        Method that reads the conversion table as specified in "codebook_folder" (path)
        and "cdbk_name" (list with names of the files) in utils.py file.

        Parameters
        ----------
        scaling: bool
            If True the maximum score to each item is stored for rescaling purposes.
        Returns
        -------
        dict
            key: name of the codebook, value: named tuple with item list,
            subscale-scale correspondance dictionaries and
            max value dictionary, if required
        """
        codebook_dict = {}
        for cdbk_name in self.codebook_list:
            cdbook = pd.read_excel(os.path.join(ut.codebook_folder,
                                                cdbk_name))
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
                    codebook_dict[cdbk_name.split('_')[2]] = levels_feat_scaled(item_vect=item_vect,
                                                                                scale_map=scale_map,
                                                                                subscale_map=subscale_map,
                                                                                max_val=max_val)
                else:
                    codebook_dict[cdbk_name.split('_')[2]] = levels_feat(item_vect=item_vect,
                                                                         scale_map=scale_map,
                                                                         subscale_map=subscale_map)
        return codebook_dict

    @staticmethod
    def adi_levels(df_instrument, adi_codebook, recode=3, max_val=None):
        """
        This static method directly processes the dataframe that corresponds to ADI-R data
        and needs a specific codebook to reconstruct the DSM-5 diagnostic criteria.

        Parameters
        ----------
        df_instrument: dataframe
        adi_codebook: namedtuple
        recode: int
            all scores greater than recode are set at 0
        max_val: dict
        Returns
        -------
        dataframe
            index of the dataframe are the subject GUIs, the features are:
            (interview age, interview date, DSM-5 items, DSM-5 subscales, DSM-5 scales)
        """
        logging.info("Processing ADI-R")
        item_vect = adi_codebook.item_vect
        subscale_vect = list(adi_codebook.subscale_map.keys())
        scale_vect = list(adi_codebook.scale_map.keys())
        cnrow = df_instrument.shape[0]

        # recode selected features if necessary
        if recode is not None:
            for col in item_vect:
                df_instrument.loc[df_instrument[col] >= 999, col] = None
                df_instrument.loc[df_instrument[col] < 0, col] = None  # -999=NA/missing
                df_instrument.loc[df_instrument[col] > recode, col] = 0
        # drop observations with completely missing features or all entries equal zero
        df_instrument.reset_index(inplace=True)
        dropobs_list = _drop_obs(df_instrument[item_vect])
        df_instrument.drop(dropobs_list, axis=0, inplace=True)
        logging.info(f'Dropped {cnrow - df_instrument.shape[0]} '
                     f'observations with completely missing information')

        # drop duplicates (based on interview age) from dataframe and feature scores
        cnrow = df_instrument.shape[0]
        df_instrument.drop_duplicates(['subjectkey',
                                       'interview_age'] +
                                      item_vect,
                                      keep='last', inplace=True)
        df_instrument.index = df_instrument['subjectkey']
        df_instrument.drop('subjectkey', axis=1, inplace=True)
        logging.info(f'Dropped {cnrow - df_instrument.shape[0]} duplicated observations')
        logging.info(f'Current number of observation: {df_instrument.shape[0]}\n\n')

        # create DSM-5 level dataframe
        for subkey, feat_vect in adi_codebook.subscale_map.items():
            df_instrument[subkey] = df_instrument[feat_vect].apply(lambda x: _create_feat_subscale(x, max_val),
                                                                   axis=1)
        for sckey, subsc_vect in adi_codebook.scale_map.items():
            df_instrument[sckey] = df_instrument[subsc_vect].apply(lambda x: _create_feat_scale(x, max_val),
                                                                   axis=1)
        df_instrument['interview_period'] = _generate_age_bins(df_instrument.interview_age)
        adi_levels = df_instrument[['interview_age',
                                    'interview_date',
                                    'interview_period'] + item_vect + subscale_vect + scale_vect]
        return adi_levels

    @staticmethod
    def ados_levels(ados_dict, ados_codebook, renamecol, recode=3, max_val=None):
        """
        In order to create ados levels, we considered the DSM-5 conversion table
        as described in doi: 10.1176/appi.ajp.2012.12020276
        In order to update it and consider also edition 2 of the ADOS,
        we developed a correspondence table that links DSM-5 dimensions
        to ADOS-G items to ADOS-2 (see ./doc/ados-dsm5).

        The ados_levels method creates a unique dataframe with all the entries
        from all ADOS (modules and editions) for the subscales and scales
        of the DSM-5 criteria. It also returns a dictionary
        for the items, where all editions of the same modules
        are merged into one dataframe, except for module 1.
        In fact, Ados2-module 1 has a feature
        (i.e. Overall Quality of Rapport, codingb_oqrap_b) that previous versions
        do not have. That is why level 1 depth is returned as a separate dataframe.

        Parameters
        ----------
        ados_dict: dict
            Dictionary of ADOS dataframes (separate modules and editions)
        ados_codebook: namedtuple
        renamecol: dict
            Dictionary with the names of the columns as keys and the possible versions
            of the column names (separate tables might code the same items differently)
            as values.
        recode: int
           Maximum score for an item, values > than that should be recoded to 0
        max_val: dict
            Dictionary storing the maximum value for each item for rescaling purposes
        Returns
        -------
        dataframe
            ados dataframe with all merged modules and editions, only scales and subscales
        dict
            dictionary of separate dataframes with only items
        """
        # Rename columns to uniform tables
        item_vect = ados_codebook.item_vect
        col_dict = {}
        for k, df in ados_dict.items():
            df.rename(columns={c1: c2 for c1, c2 in renamecol.items()}, inplace=True)
            col_dict.setdefault('-'.join(sorted(df.columns.intersection(item_vect))),
                                list()).append(k)
        # Apply function that concatenates datasets with uniform features
        mmdict = _merge_and_match(ados_dict, col_dict)
        # Create item, scale, subscale levels
        # Return a unique dataframe for scales and subscales
        ados_level_dict = _mimick_adi_levels(mmdict, ados_codebook,
                                             recode, max_val)
        ados = pd.concat([ados_mod_df[['interview_age', 'interview_date', 'interview_period']
                                      + list(ados_codebook.subscale_map.keys())
                                      + list(ados_codebook.scale_map.keys())]
                          for ados_mod_df in ados_level_dict.values()])
        ados = ados.reset_index().sort_values(by=['subjectkey',
                                                  'interview_age'],
                                              axis=0)
        ados.index = ados['subjectkey']
        ados.drop('subjectkey', axis=1, inplace=True)
        # Return a dictionary of possibly merged datasets for items
        ados_item_dict = {k: df[['interview_age', 'interview_date', 'interview_period']
                                + list(df.columns.intersection(item_vect))]
                          for k, df in ados_level_dict.items()}
        return ados, ados_item_dict

    @staticmethod
    def mullen_levels(df_instrument, col_names):
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
        df_instrument: Mullen dataframe
        col_names: list
            list with the names of the columns to consider
        Returns
        -------
        dataframe
            unique table with all possible levels
        """
        # Recode 999=None
        logging.info('Processing MULLEN')
        id_subj = set()
        df_instrument.reset_index(inplace=True)
        for cn in col_names:
            df_instrument.loc[df_instrument[cn] >= 777, cn] = None  # 999=NA/missing
            df_instrument.loc[df_instrument[cn] < 0, cn] = None  # -999=NA/missing
            id_subj.update(df_instrument.loc[df_instrument[cn] <= 20, cn].index.tolist())  # <=20 not coded
        # Drop duplicate subjects with the same assessment age and feature scores
        cnrow = df_instrument.shape[0]  # store current number of subjects
        # Drop subjects with completely missing information
        drop_obs = list(id_subj.union(_drop_obs(df_instrument[col_names])))
        df_instrument.drop(drop_obs, axis=0, inplace=True)
        logging.info(f'Dropped {cnrow - df_instrument.shape[0]} '
                     f'observations with completely missing information or scores <=20')
        cnrow = df_instrument.shape[0]
        df_instrument.drop_duplicates(['subjectkey',
                                       'interview_age'] +
                                      col_names,
                                      keep='last', inplace=True)
        df_instrument.index = df_instrument['subjectkey']
        df_instrument.drop('subjectkey', axis=1, inplace=True)
        df_instrument['interview_period'] = _generate_age_bins(df_instrument.interview_age)
        logging.info(f'Dropped {cnrow - df_instrument.shape[0]} duplicated observations')
        logging.info(f'Current number of observation: {df_instrument.shape[0]}\n\n')
        return df_instrument[['interview_age', 'interview_date',
                              'interview_period'] + col_names]

    @staticmethod
    def vineland_levels(vineland_dict, col_names, mergecol):
        """
        It creates a unique-level dataframe merging different versions of Vineland instrument,
        namely Vineland I, Vineland II (Parent and Caregiver rating), Vineland II (rating form),
        and Vineland 3.

        Parameters
        ----------
        vineland_dict: dictionary
            dictionary of dataframes (multiple instrument versions for vineland)
        col_names: list
            columns to select
        mergecol: dictionary
            columns to merge are displayed as keys and values.
            Keys become the new column name
        Returns
        -------
        dataframe
            dataframe with one level storing a dataframe obtained merging all
            Vineland Versions and including all strata.
        """
        # When separate columns are filled-in based on sex we merge them into one
        # Rename columns to uniform features and enable merging of the dataframes
        logging.info('Processing VINELAND')
        rid_dict = {}
        for k, df in vineland_dict.items():
            for c1, c2 in mergecol.items():
                if c1 in df.columns and c2 in df.columns:
                    df.loc[:, c1] = df.loc[:, c1].fillna(df.loc[:, c2])
                    df.drop(c2, axis=1, inplace=True)
                else:
                    df.rename(columns={c2: c1}, inplace=True)
            rid_dict[k] = df[['interview_age', 'interview_date']
                             + list(df.columns.intersection(col_names))]
        df_concat = pd.concat([rid_dict[k] for k in rid_dict.keys()],
                              sort=False)
        # Replace with NAs entries that are not conform
        for cn in df_concat.columns.intersection(col_names):
            df_concat.loc[df_concat[cn] > 140, cn] = None  # 999=NA/missing
            df_concat.loc[df_concat[cn] < 0, cn] = None  # -999=NA/missing
            if cn == 'relationship':
                df_concat.loc[df_concat[cn] == 27, cn] = None  # 27=Missing data
            elif re.search('_vscore', cn):
                df_concat.loc[df_concat[cn] > 24, cn] = None  # 24=Upper bound
        # Drop 2 subjects (see utils file)
        # that have wrong entries
        df_concat.drop(ut.drop_subj_vine, axis=0, inplace=True)
        cnrow = df_concat.shape[0]  # store current number of subjects
        # Drop subjects with completely missing information
        df_concat.reset_index(inplace=True)
        drop_obs = _drop_obs(df_concat[df_concat.columns.intersection(col_names[1:])])
        df_concat.drop(drop_obs, axis=0, inplace=True)
        logging.info(f'Dropped {cnrow - df_concat.shape[0]} '
                     f'observations with completely missing information')
        cnrow = df_concat.shape[0]
        # Drop duplicates. All entries displaying the same interview age,
        # caretaker to which it was administered, and scores
        df_concat.drop_duplicates(['subjectkey',
                                   'interview_age',
                                   'relationship'] +
                                  list(df_concat.columns.intersection(col_names)),
                                  keep='last', inplace=True)
        df_concat.sort_values(by=['subjectkey',
                                  'interview_age'],
                              axis=0, inplace=True)
        df_concat.index = df_concat['subjectkey']
        df_concat.drop('subjectkey', axis=1, inplace=True)
        df_concat.insert(0, 'interview_period', _generate_age_bins(df_concat.interview_age))
        logging.info(f'Dropped {cnrow - df_concat.shape[0]} duplicated observations')
        logging.info(f'Current number of observation: {df_concat.shape[0]}\n\n')
        return df_concat

    @staticmethod
    def srs_levels(srs_dict, col_names, mergecol, renamecol):
        """
        Returns a unique dataframe with subscales and scales for SRS.
        Merges multiple SRS datasets into one.

        Parameters
        ----------
        srs_dict: dictionary
            dictionary of dataframes
        col_names: list
            list of names of the features to consider
        mergecol: dictionary
            columns to merge into one, replace NAs (e.g., male/female subscales)
        renamecol: dict
            dictionary to uniform column names and merge dataframes
        """
        logging.info('Processing SRS')
        rid_dict = {}
        # Fill-in gaps as for Vineland (entries based on sex)
        # Rename columns to enable merging
        for k, df in srs_dict.items():
            for c1, c2 in mergecol.items():
                if c1 in df.columns and c2 in df.columns:
                    df.loc[:, c1] = df.loc[:, c1].fillna(df.loc[:, c2])
                    df.drop(c2, axis=1, inplace=True)
                    df.rename(columns={c1: renamecol[c1]}, inplace=True)
            df.rename(columns=renamecol,
                      inplace=True)
            rid_dict[k] = df[['interview_age', 'interview_date']
                             + list(df.columns.intersection(col_names))]
        df_concat = pd.concat([rid_dict[k] for k in rid_dict.keys()],
                              sort=False)
        # Replace entries that are not conform with NAs
        for cn in df_concat.columns.intersection(col_names):
            if cn == 'respond':
                df_concat.loc[df_concat[cn] == 999, cn] = None
            else:
                df_concat.loc[df_concat[cn].astype(str) == '>90', cn] = 91
                df_concat.loc[df_concat[cn].astype(str) == '999', cn] = None
                df_concat.loc[df_concat[cn].astype(str) == '"', cn] = None
                df_concat.loc[df_concat[cn].astype(str) == '.', cn] = None
                df_concat.loc[df_concat[cn].astype(str) == '?90', cn] = 90
                df_concat.loc[df_concat[cn].astype(str) == '-9', cn] = None
                df_concat.loc[df_concat[cn].astype(str) == '888', cn] = None
                df_concat.loc[df_concat[cn].astype(str) == '999.0', cn] = None
                df_concat = df_concat.astype({cn: 'float64'})
                df_concat.loc[df_concat[cn] > 200, cn] = None
        cnrow = df_concat.shape[0]  # store current number of subjects
        df_concat.reset_index(inplace=True)
        # Drop completely missing observations
        drop_obs = _drop_obs(df_concat[df_concat.columns.intersection(col_names[1:])])
        df_concat.drop(drop_obs, axis=0, inplace=True)
        logging.info(f'Dropped {cnrow - df_concat.shape[0]} '
                     f'observations with completely missing information')
        cnrow = df_concat.shape[0]
        # Drop duplicates (same interview age, same respondent, and same scores)
        df_concat.drop_duplicates(['subjectkey',
                                   'interview_age',
                                   'respond'] +
                                  list(df_concat.columns.intersection(col_names)),
                                  keep='last', inplace=True)
        df_concat.sort_values(by=['subjectkey',
                                  'interview_age'],
                              axis=0, inplace=True)
        df_concat.index = df_concat['subjectkey']
        df_concat.drop('subjectkey', axis=1, inplace=True)
        df_concat.insert(0, 'interview_period', _generate_age_bins(df_concat.interview_age))
        logging.info(f'Dropped {cnrow - df_concat.shape[0]} duplicated observations')
        logging.info(f'Current number of observation: {df_concat.shape[0]}\n\n')
        return df_concat


def _drop_obs(df):
    """
    Function that takes as input a dataframe (only the desired features)
    and returns a list of subjects to drop.
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
        try:
            if sum(row.dropna()) == 0:
                drop_subj.append(sid)
            elif row.count() == 0:
                drop_subj.append(sid)
        except TypeError:  # compatible to ados
            if sum(row.astype(float).dropna()) == 0:
                drop_subj.append(sid)
            elif row.astype(float).count() == 0:
                drop_subj.append(sid)
    return drop_subj


# For both ADI and ADOS
def _create_feat_subscale(row, max_val=None):
    """
    Function that creates subscale scores based on the items and according
    to available information and maximum value.
    The sum of the item scores divided by the sum of the maximum available scores is returned.

    Parameters
    ----------
    row: pandas Series
        Row of the dataframe with item entries to rescale
    max_val: dict
        dictionary with maximum score available in the instrument for each item.
        If None the sum of the item scores is returned.
    Returns
    -------
    float
    """
    select = row.dropna().index
    if len(select) == 0:
        return None
    else:
        if max_val is not None:
            return sum(row[select]) / sum([max_val[col] for col in select])
        else:
            return sum(row[select])


def _create_feat_scale(row, max_val=None):
    """
    Function that creates scale scores based on subscales. The average of the available scores
    is returned.
    Prameters
    ---------
    row: pandas Series
    max_val: dict
        If None the sum of the subscale scores is returned.
    Returns
    -------
    float
    """
    select = row.dropna().index
    if len(select) == 0:
        return None
    else:
        if max_val is not None:
            return sum(row[select]) / len(select)
        else:
            return sum(row[select])


def _merge_and_match(df_dict, dict_colnames):
    """
    Function to mach separate datasets with the same features and merge them

    Parameters
    ----------
    df_dict: dict
        dictionary of dataframes
    dict_colnames: dict
        dictionary of column names (joined) as keys and list of
        instruments that share the same columns as values
    Returns
    -------
    dict
        dictionary of merged instruments
    """
    merged_dict = {}
    for colnames, v_names in dict_colnames.items():
        colname_vect = colnames.split('-')
        merged_dict[v_names[0]] = pd.concat([df_dict[name][['interview_age', 'interview_date']
                                                           + sorted(colname_vect)] for name in v_names])
    return merged_dict


def _mimick_adi_levels(dict_merged, ados_codebook, recode, max_val):
    """
    This function replicates the adi_levels method modified in order
    to be applied to ados data.

    Parameters
    ----------
    dict_merged: dict
        dictionary with the datasets merged by _merge_and_match function
    ados_codebook: namedtuple
    recode: int
    max_val: dict
    Returns
    -------
    dictionary
        dictionary of separate dataframes with all levels
    """
    item_vect = ados_codebook.item_vect
    subscale_vect = list(ados_codebook.subscale_map.keys())
    scale_vect = list(ados_codebook.scale_map.keys())

    # recode selected features if necessary
    ados_levels = {}
    for k, mod_df in dict_merged.items():
        logging.info(f"Processing merged {k}:")
        if recode is not None:
            # Ados2_2012 columns need to be changed to float
            for col in mod_df.columns.intersection(item_vect):
                mod_df.loc[mod_df[col].astype(float) >= 999, col] = None
                mod_df.loc[mod_df[col].astype(float) < 0, col] = None  # -999=NA/missing
                mod_df.loc[mod_df[col].astype(float) > recode, col] = 0
        # drop observations with completely missing features or all item entries equal zero
        # we do this for ADOS and ADI, not for the other instruments, because we need the item
        # scores to code for the subscales and the scales.
        cnrow = mod_df.shape[0]
        mod_df.reset_index(inplace=True)
        dropobs_list = _drop_obs(mod_df[mod_df.columns.intersection(item_vect)])
        mod_df.drop(dropobs_list, axis=0, inplace=True)
        logging.info(f'Dropped {cnrow - mod_df.shape[0]} '
                     f'observations with completely missing information')
        # drop duplicates (based on interview age) and feature scores
        # entries collected at the same age reporting the same scores are dropped
        cnrow = mod_df.shape[0]
        mod_df.drop_duplicates(['subjectkey',
                                'interview_age'] +
                               list(mod_df.columns.intersection(item_vect)),
                               keep='last', inplace=True)
        mod_df.index = mod_df['subjectkey']
        mod_df.drop('subjectkey', axis=1, inplace=True)
        logging.info(f'Dropped {cnrow - mod_df.shape[0]} duplicated observations')
        logging.info(f'Current number of observation: {mod_df.shape[0]}\n')

        # create DSM-5 level dataframe
        for subkey, feat_vect in ados_codebook.subscale_map.items():
            if mod_df[mod_df.columns.intersection(feat_vect)].empty:
                mod_df.loc[:, subkey] = pd.Series([None] * mod_df.shape[0])
            else:
                mod_df.loc[:, subkey] = mod_df[mod_df.columns.intersection(feat_vect)].apply(
                    lambda x: _create_feat_subscale(x.astype(float), max_val),
                    axis=1)
        for sckey, subsc_vect in ados_codebook.scale_map.items():
            mod_df.loc[:, sckey] = mod_df[subsc_vect].apply(lambda x: _create_feat_scale(x, max_val),
                                                            axis=1)
        ados_levels[k] = mod_df.loc[:, ['interview_age',
                                        'interview_date'] + list(
            mod_df.columns.intersection(item_vect)) + subscale_vect + scale_vect]
        ados_levels[k]['interview_period'] = _generate_age_bins(ados_levels[k].interview_age)
    return ados_levels


def _generate_age_bins(interview_age):
    """Returns the time period from the age of assessment

    Parameters
    ----------
    interview_age: pd Series with interview ages and subject keys as index

    Return
    ------
    list
        time period string (P1-P5)
    """
    age_bins = []
    for aoa in interview_age:
        if 0 < float(aoa) <= 30.0:
            age_bins.append('P1')
        elif 30.0 < float(aoa) <= 72.0:
            age_bins.append('P2')
        elif 72.0 < float(aoa) <= 156.0:
            age_bins.append('P3')
        elif 156.0 < float(aoa) < 204.0:
            age_bins.append('P4')
        else:
            age_bins.append('P5')
    return pd.Series(age_bins, index=interview_age.index)
