import pandas as pd
import utils as ut
import os
import logging
from datetime import datetime
from dataclasses import dataclass
from collections import namedtuple
import pickle as pkl
from upsetplot import plot, from_memberships
from itertools import combinations
from matplotlib import pyplot as plt
from select_phenotype import select_phenotype


# Dataclass to store patient demographics
# and patient interview info.
@dataclass
class Sinfo:
    interview_age: int
    interview_date: datetime
    instrument: str
    sex: str
    diagnosis: str


# Namedtuple with read_data output elements
dataobj = namedtuple('dataobj', ['ins_dict', 'col_names', 'subjinfo'])


class Dataset:
    """Each instance of this class is initialized with a list
    of strings corresponding to instrument names.
    """

    def __init__(self, list_insnames):
        self.list_insnames = list_insnames

    def read_data(self, save_out=False, select_diag=True):
        """Method that imports the dataset specified by the user
        and returns a dataframe for each file. If two data files share the same
        columns, they are merged into a unique dataframe
        (e.g., Vineland-II administered to teachers and caregivers). If save_out == True
        a namedtuple object storing the outputs is dumped into the selected output folder

        Parameters
        ----------
        save_out: bool
            default False. Set to True if a namedtuple of outputs is to be saved
            in the output folder
        select_diag: bool
            if True it enables the selection of a cohort with a specific diagnosis
            (only case ASC active for now)
        Returns
        -------
        dictionary
            each key is the name of an instrument and the correspondent value
            is a dataframe that stores subject assessments
        dictionary
            keys are the column names of the NDAR database tables. Values
            store the correspondent long format name (i.e., second row from tables)
        dictionary
            keys are NDAR_UI and value is a list of all the entries info (e.g., interview date/age,
            sex, instrument name)
        """
        ins_dict = {}
        col_names = {}
        duplicate_ins = []  # concatenate equal instruments
        logging.info('Loading dataset:')
        for n in self.list_insnames:
            name = n.strip(".txt")
            logging.info(f'{name}')
            # Create dictionary of column names
            with open(os.path.join(ut.data_folder, n), 'r') as f:
                first_line = f.readline().strip('\n"').split('\t')
                second_line = f.readline().strip('\n"').split('\t')
            for idx, s in enumerate(first_line):
                if s not in col_names:
                    col_names[s.strip('"')] = second_line[idx].strip('"')
            check_str = '\t'.join(first_line[2:])
            if check_str not in duplicate_ins:
                duplicate_ins.append(check_str)
                # Create dictionary of instrument dataframes
                ins_dict[name] = pd.read_csv(os.path.join(ut.data_folder, n),
                                             sep='\t', header=0, skiprows=[1],
                                             low_memory=False, index_col=ut.subjectid,
                                             parse_dates=[ut.date_var])
                nsubj = ins_dict[name].index.unique().shape[0]
                logging.info(f'Read table {name} -- N subjects (unique): {nsubj}\n')
            else:
                idx = duplicate_ins.index(check_str)
                new_df = pd.read_csv(os.path.join(ut.data_folder, n),
                                     sep='\t', header=0, skiprows=[1],
                                     low_memory=False, index_col=ut.subjectid,
                                     parse_dates=[ut.date_var])
                ins_dict[self.list_insnames[idx]] = pd.concat([ins_dict[self.list_insnames[idx]], new_df])

        if select_diag:
            gui_diag, gui_site = select_phenotype(ut.exp_asd, ut.data_folder,
                                                  ut.pheno_data, ut.exceptions)
            logging.info('\n Filtering subjects with diagnosis, instrument:')
            for name in ins_dict.keys():
                logging.info(f'{name}')
                nsubj = ins_dict[name].index.unique().shape[0]
                ins_dict[name] = ins_dict[name].loc[ins_dict[name].index.intersection(gui_diag.keys())]
                nsubj_diag = ins_dict[name].index.unique().shape[0]
                logging.info(f'Selected {nsubj_diag}/{nsubj} subjects -- '
                             f'Dropped {nsubj - nsubj_diag}\n')
                phen_vect = [gui_diag[idx] for idx in ins_dict[name].index]
                ins_dict[name]['phenotype'] = phen_vect
                if 'site' not in ins_dict[name].columns:
                    ins_dict[name]['site'] = [gui_site[idx] for idx in ins_dict[name].index]
        else:
            for name in ins_dict.keys():
                ins_dict[name]['phenotype'] = ['NA'] * ins_dict[name].shape[0]
        logging.info('\n Storing Subject info')
        subjinfo = {}
        for ins, df in ins_dict.items():
            for idx, row in df.iterrows():
                subjinfo.setdefault(idx, list()).append(Sinfo(interview_age=row.interview_age,
                                                              interview_date=row.interview_date,
                                                              instrument=ins,
                                                              sex=row.sex,
                                                              diagnosis=row.phenotype))

        if save_out:
            logging.info('Saving outputs...')
            pkl.dump(dataobj(ins_dict, col_names, subjinfo),
                     open(os.path.join(ut.out_folder, 'data_obj.pkl'), 'wb'))
        if select_diag is not None:
            return ins_dict, col_names, subjinfo
        else:
            return ins_dict, col_names, subjinfo

    @staticmethod
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
            plt.savefig(os.path.join(ut.out_folder, 'intersection_plot.jpg'))
        else:
            plt.show()

    @staticmethod
    def merge_df(ins_dict):
        df_all = pd.DataFrame(columns=ut.shared_col)
        for n_ins in ins_dict.keys():
            df_all = pd.concat([df_all, ins_dict[n_ins][ut.shared_col]])
        return df_all


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
