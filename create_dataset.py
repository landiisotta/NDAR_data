import utils as ut
import os
import re
import pickle as pkl
import pandas as pd
from matplotlib import pyplot as plt
from itertools import combinations
from upsetplot import plot, from_memberships
from import_data import ReadData
from create_levels import CreateLevels


def dataset(phenotype=None):
    """
    This function executes all the steps that are required in order to:
    (1) create a a dictionary with cross-sectional data from all the instruments selected
    (2) create a dictionary that stores longitudinal data for each instrument selected
    (3) store the subjects with longitudinal observations. The name of the instrument
    and the assessment date should be reported.

    The names of the instrument tables should be reported in utils.py
    under instrument_dict variable. The output are saved as pickle objects. The
    output folder where to store the returned data should be stated in utils.py (output_folder
    variable).

    Parameters
    ----------
    phenotype: str
        For now, only enabled "autism". If None all subjects with
        every phenotype label are selected.
    Returns
    -------
    dict
        dictionary with cross sectional dataframes, for now only the first observation is retained
    dict
        dictionary with dataframes including all repeated observations
    dict
        dictionary of named tuple that stores the information on the subjects with repeated observations.
        Single ados modules are considered for longitudinal ADOS data.
    """
    data = ReadData(ut.instrument_dict, phenotype=phenotype)
    instrument_dict = data.data_wrangling()

    new_dict = {}
    # If ADI and ADOS are required we initiate the CreateLevels class
    # with the list of codebooks we need.
    if re.search('adi|ados', '-'.join([k for k in instrument_dict.keys()])):
        create_lev = CreateLevels(ut.codebooks)
        maps = create_lev.create_map(scaling=True)
        for name in instrument_dict.keys():
            # Create levels
            if re.match('adi', name):
                new_dict['adi'] = create_lev.adi_levels(instrument_dict[name],
                                                        maps['adi'],
                                                        max_val=maps['adi'].max_val)
            elif re.match('ados', name) and 'ados' not in new_dict:
                new_dict['ados'], ados_dict = create_lev.ados_levels(
                    {k: ins for k, ins in instrument_dict.items() if re.match('ados', k)},
                    maps['ados'], ut.ados_rename,
                    max_val=maps['ados'].max_val)
                new_dict.update(ados_dict)
            elif re.match('mullen', name):
                new_dict['mullen'] = create_lev.mullen_levels(instrument_dict[name],
                                                              ut.mullen_col_names)
            elif re.match('srs', name) and 'srs' not in new_dict:
                new_dict['srs'] = create_lev.srs_levels(
                    {k: ins for k, ins in instrument_dict.items() if re.match('srs', k)},
                    ut.srs_col_names, ut.srs_mergecol,
                    ut.srs_rename)
            elif re.match('vineland', name) and 'vineland' not in new_dict:
                new_dict['vineland'] = create_lev.vineland_levels(
                    {k: ins for k, ins in instrument_dict.items() if re.match('vineland', k)},
                    ut.vineland_col_names,
                    ut.vineland_mergecol)
    else:
        create_lev = CreateLevels()
        for name in instrument_dict.keys():
            if re.match('mullen', name):
                new_dict['mullen'] = create_lev.mullen_levels(instrument_dict[name],
                                                              ut.mullen_col_names)
            elif re.match('srs', name) and 'srs' not in new_dict:
                new_dict['srs'] = create_lev.srs_levels(
                    {k: ins for k, ins in instrument_dict.items() if re.match('srs', k)},
                    ut.srs_col_names, ut.srs_mergecol,
                    ut.srs_rename)
            elif re.match('vineland', name) and 'vineland' not in new_dict:
                new_dict['vineland'] = create_lev.vineland_levels(
                    {k: ins for k, ins in instrument_dict.items() if re.match('vineland', k)},
                    ut.vineland_col_names,
                    ut.vineland_mergecol)

    long_info = data.assess_longitudinal({k: new_dict[k] for k in new_dict.keys() if k != 'ados'})
    cs_dict = {}
    # Generate cross-sectional data
    # drop duplicates and save first observation (consider only ados subscales and scales, not single items)
    for ins in [k for k in new_dict.keys() if not re.match('ados[0-9|t]', k)]:
        cs_dict[ins] = new_dict[ins].reset_index().drop_duplicates(subset='subjectkey',
                                                                   keep='first')
        cs_dict[ins].index = cs_dict[ins]['subjectkey']
        cs_dict[ins].drop('subjectkey', axis=1, inplace=True)
    # Save cross-sectional dataset
    pkl.dump(cs_dict,
             open(os.path.join(ut.out_folder, 'cross-sectional_data.pkl'), 'wb'))
    # Save longitudinal datasets
    pkl.dump((long_info, new_dict),
             open(os.path.join(ut.out_folder, 'longitudinal_data.pkl'), 'wb'))

    return cs_dict, new_dict, long_info


def plot_intersection(ins_dict, save_fig=False):
    """
    Visualize an upsetplot displaying the number of unique subjects found simultaneously
    in a pair of instruments

    Parameters
    ----------
    ins_dict: dictionary
    save_fig: bool
    """

    ins_names = list(ins_dict.keys())
    list_comb = sum([list(map(list,
                              combinations(ins_names,
                                           i + 1))) for i in range(len(ins_names) + 1)], [])
    list_uniquesubj = []
    for lc in list_comb:
        list_uniquesubj.append([set(ins_dict[n].index) for n in lc])
    int_counts = list(map(_count_intersection, list_uniquesubj))
    inter_plot = from_memberships(list_comb, data=int_counts)
    plot(inter_plot, show_counts='%d', element_size=50, orientation='horizontal')
    if save_fig:
        plt.savefig(os.path.join(ut.out_folder, 'intersection_plot'), format='pdf')
    else:
        plt.show()


def concatenate_instrument(ins_dict, save=False):
    """
    Function that horizontally concatenates the cross-sectional dataframes. We aim at maximizing
    the number of subjects we can consider selecting different instruments.

    Parameters
    ----------
    ins_dict: dict
        dictionary of cross-sectional dataframes
    save: bool
        if True the cross-sectional wide dataframe is dumped
    Returns
    -------
    dataframe
        wide cross-sectional dataframe.
        The interview age and date are joined with the name of the corresponding instrument.
    """
    subj_ids = pd.Index([])
    for tab in ins_dict.values():
        if len(subj_ids) == 0:
            subj_ids = tab.index
        else:
            subj_ids = subj_ids.intersection(tab.index)
    wide_df = pd.concat([tab.loc[subj_ids].rename(columns={'interview_age': '_'.join(['interview_age', k]),
                                                           'interview_date': '_'.join(['interview_date', k])})
                         for k, tab in ins_dict.items()],
                        axis=1, sort=False)
    if save:
        pkl.dump(wide_df, open(os.path.join(ut.out_folder,
                                            'cs_wide_dataset.pkl'), 'wb'))

    return wide_df


def concatenate_all_ins(ins_dict, save=False):
    try:
        ins_dict['ados'].columns = ['_'.join([c, 'ados']) for c in ins_dict['ados'].columns]
    except KeyError:
        pass
    df_mcar = pd.concat([df[[col for col in df.columns
                             if not re.search('interview|sex|respond|relation', col)]]
                         for df in ins_dict.values()], axis=1)

    if save:
        pkl.dump(df_mcar, open(os.path.join(ut.out_folder,
                                            'mcar_wide_dataset.pkl'), 'wb'))
    return df_mcar


"""
Private functions
"""


def _count_intersection(set_subj):
    """
    Returns the intersection of one or more sets

    Parameters
    ----------
    set_subj: list of sets
    Returns
    -------
    int: intersection cardinality
    """
    count_set = set_subj[0]
    if len(set_subj) > 1:
        for s in set_subj[1:]:
            count_set = count_set.intersection(s)
    return len(count_set)
