"""This file includes the paths to folders and parameters."""

# Folder to txt NDAR tables
data_folder = ''
# Name of the file with phenotypes for subject selection
phenotype_file = ''
# Output folder
out_folder = ''
# Folder to codebooks for DSM-5 features
codebook_folder = './doc'
# Codebook file name list
codebooks = ['huerta_dsm5_adi_codebook.xlsx',
             'huerta_dsm5_ados_codebook.xlsx']

instrument_dict = {
    'adir': '.txt',
    'mullen': '.txt',
    'vinelands': '.txt',
    'vinelandp': '.txt',
    'vineland3': '.txt',
    'ados2mt': '.txt',
    'ados2m1': '.txt',
    'ados2m2': '.txt',
    'ados2m3': '.txt',
    'ados2m4': '.txt',
    'adosmt': '.txt',
    'adosm1': '.txt',
    'adosm2': '.txt',
    'adosm3': '.txt',
    'adosm4': '.txt',
    'ados2007m1': '.txt',
    'ados2007m2': '.txt',
    'ados2007m3': '.txt',
    'srs': '.txt',
    'srs2': '.txt',
    'srsadults': '.txt',
    'srspreschool': '.txt'
}

# Column names shared by all the instruments
shared_col = ['dataset_id',
              'interview_age',
              'interview_date',
              'sex']

# Regular expression to filter subjects with ASC diagnosis from NDAR dataset
phenotype_regex = r''
# for these terms check for asd|autism spectrum and asd in
# phenotype_description column, respectively
exception_regex = ''

mullen_col_names = []
vineland_col_names = []
vineland_mergecol = {'': ''}

srs_col_names = []
srs_mergecol = {'': ''}
srs_rename = {'': ''}

ados_rename = {'': ''}
