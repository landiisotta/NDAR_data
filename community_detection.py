import numpy as np
import itertools
from scipy.sparse import csr_matrix


def create_cooccurences_matrix(gui_vect, cluster_vect):
    """
    Build cooccurrence matrix from the list of the unique GUIs
    and the list of the GUIs in each cluster, irrespective
    of instruments.
    Parameters
    ----------
    gui_vect: list
    cluster_vect: list
    Returns
    ------
    gui_cooc_matrix: sparse matrix
    gui_to_id: dictionary
        {gui: idx}
    """
    gui_to_id = dict(zip(gui_vect, range(len(gui_vect))))
    cluster_as_ids = [np.sort([gui_to_id[gui] for gui in cl
                               if gui in gui_to_id]).astype('uint32')
                      for cl in cluster_vect]
    row_ind, col_ind = zip(*itertools.chain(*[[(i, gui) for gui in cl] for i, cl in enumerate(cluster_as_ids)]))
    data = np.ones(len(row_ind), dtype='uint32')  # use unsigned int for better memory utilization
    max_gui_id = max(itertools.chain(*cluster_as_ids)) + 1
    clust_gui_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(cluster_as_ids), max_gui_id))
    gui_cooc_matrix = clust_gui_matrix.T * clust_gui_matrix
    gui_cooc_matrix.setdiag(0)
    return gui_cooc_matrix, gui_to_id
