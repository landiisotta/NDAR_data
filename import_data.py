from dataclasses import dataclass
from collections import namedtuple
import pandas as pd
import pickle as pkl
import re
import os
import logging
import utils as ut


@dataclass
class DemInfo:
    """
    Dataclass to store demographic information
    on subjects of interest.
    gui: NDAR unique identifiers
    sex
    phenotyep
    ethnic groups
    dataset_id: int
        number of the dataset from which the
        observation was taken
    """
    gui: str
    sex: str
    phenotype: str
    ethnic_group: str
    dataset_id: int


# Namedtuple for storing longitudinal data information
# (i.e., name of the instrument, age at assessment, date at assessment)
longdata = namedtuple('longdata', ['instrument', 'interview_age', 'interview_date'])


class ReadData:
    """
    The ReadData class enables the data_wrangling method and the assess_longitudinal method.
    It is initialized by
    instrument_file_names: dict
        A dictionary with the descriptive names of the instrument we want to import as keys,
        and the corresponding name of the data files as values.
    phenotype: str
        A string specifying the phenotype of the subjects we want to consider. Two options
        have been implemented so far: "autism" for the individuals who received an ASD
        diagnosis and None to import all the subjects available in NDAR.
    """

    def __init__(self, instrument_file_names,
                 phenotype=None):
        self.instrument_file_name = instrument_file_names
        self.phenotype = phenotype  # only 'autism' enabled for now

    def data_wrangling(self):
        """
        This method (1) selects the subjects with a specific (if required) diagnosis. In order to
        do this the user needs to create a utils.py file with a variable "data_folder", that stores the
        path to the data, and a variable "phenotype_file", that stores the name of the table to read
        in order to select the subjects who received the diagnosis required. (2) Creates and dumps a list of
        DemInfo dataclass objects that store the demographic information of the selected individuals.
        (3) Reads the dataset of the instrument desired and stores them in a dictionary, with the
        descriptive names of the instruments as keys and the dataframes with the observation of the
        subjects selected by phenotype as values.

        In order to input the regular expression to select all the possible strings describing the
        phenotype of interest, the utils.py file should include two variables (for the autistic profile).
        Namely, the phenotype_regex and the exception_regex (see _select_phenotype private function below).

        Returns
        -------
        dictionary
            Dictionary as created at step (3).
        """
        # Select individuals with desired phenotype
        if self.phenotype == 'autism':
            logging.info(f"Select individuals with {self.phenotype} and save "
                         f"demographic info.")
            gui_index, demo_info = _select_phenotype(os.path.join(ut.data_folder,
                                                                  ut.phenotype_file),
                                                     ut.phenotype_regex,
                                                     ut.exception_regex)
            logging.info(f'Number of subjects with {self.phenotype}: {len(demo_info)}\n\n')
        else:
            logging.info('Phenotype not selected, considering all subjects in the database.\n\n')
            gui_index, demo_info = _select_phenotype(os.path.join(ut.data_folder,
                                                                  ut.phenotype_file))
        # Dumping the demographic info
        pkl.dump(demo_info, open(os.path.join(ut.out_folder, 'demographics.pkl'), 'wb'))
        # Read tables
        logging.info('Loading datasets:')
        table_dict = {}
        for tab_name in self.instrument_file_name.values():
            table_dict[tab_name.strip('.txt')] = _read_ndar_table(tab_name,
                                                                  gui_index,
                                                                  ut.data_folder)
        return table_dict

    @staticmethod
    def assess_longitudinal(new_dict):
        """
        This method assess data longitudinality based on interview age (in months).
        Parameters
        ----------
        new_dict: dict
            Dictionary of instruments as returned by data_wrangling_method
        Returns
        -------
        dictionary
            Dictionary with GUIs as keys and list of named tuples as value.
            Each named tuple is a 'longdata' that includes information on the
            instrument name, and interview age/date. Only subjects with multiple observation
            in at least one instrument dataset are included.
        """
        # Assess longitudinal data
        logging.info("Assessing data longitudinality")
        long_ass = {}
        for ins_name, ins_tab in new_dict.items():
            nsubj = 0
            for subj in ins_tab.index.unique():
                if ins_tab.loc[subj].shape[0] < ins_tab.columns.shape[0]:
                    nsubj += 1
                    long_ass.setdefault(subj, list()).extend(list(dict.fromkeys(([longdata(ins_name,
                                                                                           row.interview_age,
                                                                                           row.interview_date)
                                                                                  for _, row in
                                                                                  ins_tab.loc[subj].iterrows()]))))
                else:
                    pass
            logging.info(f"For instrument {ins_name} -- longitudinal observation found for {nsubj} subjects")
        return long_ass


"""
Private functions
"""


def _read_ndar_table(tab_name, gui_index, data_folder):
    """
    Private function used by data_wrangling method to import instrument datasets.
    Parameters
    ----------
    tab_name: str with name of the dataset to read
    gui_index: Index with GUIs from subjects to consider (e.g., specific phenotype)
    data_folder: str with name of the folder where data are stored

    Returns
    -------
    pandas dataframe
    """
    name = tab_name.strip(".txt")
    logging.info(f'{name}')
    ins_table = pd.read_csv(os.path.join(data_folder, tab_name),
                            sep='\t', header=0, skiprows=[1],
                            low_memory=False,
                            parse_dates=['interview_date'])
    # Ordered by interview age and subjectkey
    ins_table.sort_values(by=['subjectkey',
                              'interview_age'],
                          axis=0, inplace=True)
    ins_table.index = ins_table['subjectkey']
    ins_table = ins_table.loc[gui_index.intersection(ins_table.index)]
    ins_table.drop('subjectkey', axis=1, inplace=True)
    logging.info(f'Read table {name} -- N subjects (unique): {ins_table.shape[0]} '
                 f'({ins_table.index.unique().shape[0]})\n')
    return ins_table


def _select_phenotype(file_path,
                      phenotype_regex=None,
                      exception_regex=None):
    """
    Function that reads the NDAR phenotype file, filter the desired phenotypes
    (designed for autistic phenotype) and returns the pandas Index Series of unique GUIs and
    the list of dataclass objects storing demographic info.
    Parameters
    ----------
    phenotype_regex: str
        Regular expression to detect the autistic phenotype
        Default: None (all subjects are considered)
    exception_regex: str
        Regular expression exceptions for autistic phenotypes (i.e., 'non control|calculated').
        For these cases the phenotype_description column should match the phenotype_regex regular expression.
        Otherwise the subject is dropped.
        Default: None (all subjects are considered)
    file_path: str
        Path to the file and file name where phenotypic information are stored.
    Returns
    -------
    pandas Index Series
        unique subject GUI with desired phenotype
        list of dataclass objects storing demographic information of the
        selected subjects
    """
    phenotype_table = pd.read_table(file_path,
                                    sep='\t',
                                    header=0,
                                    skiprows=[1],
                                    low_memory=False,
                                    index_col='subjectkey')

    if phenotype_regex is not None and exception_regex is not None:
        # Drop rows with missing phenotype
        phenotype_table.dropna(subset=['phenotype'], inplace=True)

        # Save GUI subjects with required phenotype
        subjindex = phenotype_table[list(map(lambda x: bool(re.search(phenotype_regex,
                                                                      str(x).lower())),
                                             phenotype_table.phenotype))].index
        phenotype_table = phenotype_table.loc[subjindex]
        # Check if all duplicate entries are consistent,
        # (i.e., they all have accepted strings).
        # Otherwise, add to list of subjects to drop.
        subjout = set()
        dupsubj = phenotype_table.loc[phenotype_table.index.duplicated(keep=False)].index.unique()
        for idsubj in dupsubj:
            uniquediag = set([str(d).lower() for d in phenotype_table.loc[idsubj].phenotype.unique()])
            if len(uniquediag) > 1:
                for d in uniquediag:
                    ctr = bool(re.search(phenotype_regex, d))
                    if not ctr:
                        subjout.add(idsubj)
        phenotype_table.drop(subjout, axis=0, inplace=True)
        # Save exception phenotype
        checksubj = phenotype_table[list(map(lambda x: bool(re.search(exception_regex,
                                                                      str(x).lower())),
                                             phenotype_table.phenotype))].index
        dropexc = set()
        for s in checksubj:
            if not re.search(phenotype_regex,
                             str(phenotype_table.loc[s].phenotype_description).lower()):
                dropexc.add(s)
        phenotype_table.drop(dropexc, axis=0, inplace=True)
    else:
        subjindex = phenotype_table.index.unique()
        phenotype_table = phenotype_table.loc[subjindex]

    # Drop duplicates with all allowed phenotypes
    phenotype_table = phenotype_table.loc[~phenotype_table.index.duplicated(keep='first')]

    demolist = [DemInfo(gui=gui,
                        sex=row.sex,
                        phenotype=str(row.phenotype).lower(),
                        ethnic_group=row.ethnic_group,
                        dataset_id=row.dataset_id)
                for gui, row in phenotype_table.iterrows()]

    return phenotype_table.index, demolist
