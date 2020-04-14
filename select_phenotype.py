import pandas as pd
import os
import re
import logging
import utils as ut


def select_phenotype(phenotype_regex,
                     datafolder,
                     phenodata,
                     exception_regex,
                     dbname='NDAR'):
    """
    Function that takes as input the regular expression of the
    phenotype of interest and the file containing subject ids and
    correspondent phenotype.
    This function includes an exception intrinsic to NDAR data,
    other exceptions of interest should be added.
    Parameters
    ----------
    phenotype_regex: regular expression
        regular expression needed to select the
        phenotype of interest
    datafolder: str
        data folder path
    phenodata: str
        name of the file including list of subjects and phenotype
    exception_regex: regular expression
        particular case of the NDAR phenotype dataset. "Calculated using
        ADOS vars|Non Control" phenotyes, should match "ASD|Autism Spectrum
        Disorder" in phenotype_description column. Otherwise they need to be
        dropped.
    dbname: str
        initialized to NDAR, it returns an error if other data are used
    Returns
    -------
    dict
        {GUI: phenotype as indicated in phenodata (lowercase)}
    """
    if dbname != 'NDAR':
        logging.exception('A DB different from NDAR has been used.\n'
                          'Please check the import table parameters,'
                          ' and manage possible exceptions.')
        return
    # Read table with phenotypes (created for NDAR data)
    diag = pd.read_table(os.path.join(datafolder,
                                      phenodata),
                         sep='\t', header=0, skiprows=[1],
                         low_memory=False, index_col='subjectkey')
    # Save list of subjects with required phenotype
    listsubj = diag[list(map(lambda x: bool(re.search(phenotype_regex,
                                                      str(x).lower())),
                             diag.phenotype))].index
    diag = diag.loc[listsubj]
    # Check if all duplicate entries are consistent,
    # (i.e., they all have accepted strings).
    # Otherwise, add to list of subjects to drop.
    subjout = set()
    dupsubj = diag.loc[diag.index.duplicated(keep=False)].index.unique()
    for idsubj in dupsubj:
        uniquediag = set([str(d).lower() for d in diag.loc[idsubj].phenotype.unique()])
        if len(uniquediag) > 1:
            for d in uniquediag:
                ctr = bool(re.search(phenotype_regex, d))
                if not ctr:
                    subjout.add(idsubj)
    diag.drop(subjout, axis=0, inplace=True)

    # Remove duplicates from phenotype dataset
    diag = diag.loc[~diag.index.duplicated(keep='first')]

    # Save subject IDs with exception phenotype
    excsubj = diag[list(map(lambda x: bool(re.search(exception_regex,
                                                     str(x).lower())),
                            diag.phenotype))].index
    excsubjtodrop = set()
    for s in excsubj:
        if not re.search(phenotype_regex,
                         str(diag.loc[s].phenotype_description).lower()):
            excsubjtodrop.add(s)

    diag.drop(excsubjtodrop, axis=0, inplace=True)
    pheno_dict = {}
    site_dict = {}
    for gui in diag.index:
        if gui not in pheno_dict and gui not in site_dict:
            pheno_dict[gui] = diag.loc[gui].phenotype.lower()
            site_dict[gui] = diag.loc[gui].site
    return pheno_dict, site_dict
