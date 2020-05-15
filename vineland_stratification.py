from reval.best_nclust_cv import FindBestClustCV
import pickle as pkl
import csv
import pandas as pd
import numpy as np
import imputation as imp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import re
from visualization import plot_metrics, _scatter_plot
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, \
    ColorBar, HoverTool
from bokeh.plotting import figure, show
import logging
from create_dataset import plot_intersection
import umap
from math import pi
from scipy.spatial.distance import cdist

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO, datefmt='%I:%M:%S')


# Enable preprocessing
def run_validation(data_tr, data_ts, raw_ts, n_neighbors, period, hierarchy, cl_range=(2, 11)):
    """Function that performs the relative clustering validation.
    :param data_tr: training dataset
    :type data_tr: dataframe
    :param data_ts: test dataset
    :type data_ts: dataframe
    :param raw_ts: original test dataset before imputation
    :type raw_ts: dataframe
    :param n_neighbors: number of neighbors to consider for UMAP preprocessing step
    :type n_neighbors: int
    :param cl_range: range of number of clusters to consider, default (2, 11)
    :type cl_range: tuple
    :param period: interview period
    :type period: str
    :param hierarchy: hierarchical feature level
    :type hierarchy: str
    """
    logging.info(f'Processing Vineland feature level {hierarchy} at period {period}...')
    transform = umap.UMAP(random_state=42, n_neighbors=n_neighbors, min_dist=0.0)
    X_tr = transform.fit_transform(data_tr)
    X_ts = transform.transform(data_ts)

    # Initialize classes
    knn = KNeighborsClassifier(n_neighbors=10)
    clust = AgglomerativeClustering(affinity='euclidean', linkage='ward')

    relval = FindBestClustCV(s=knn, c=clust, nfold=10, nclust_range=cl_range,
                             nrand=100)  # This runs a 10-fold cross validation with number of clusters from 2 to 10
    # Run the model
    metric, ncl, cv_scores = relval.best_nclust(X_tr)  # the strat_vect parameter can be used to perform a stratified CV
    logging.info(f"Best number of clusters: {ncl}")
    out = relval.evaluate(X_tr, X_ts, ncl)
    logging.info(f"Training ACC: {out.train_acc}, Test ACC: {out.test_acc}")
    plot_metrics(metric)
    unique, counts = np.unique(out.train_cllab, return_counts=True)
    logging.info(f'Training set (N = {X_tr.shape[0]})\n')
    for a, b in zip(unique, counts):
        logging.info(f'N subjects in cluster {a}: {b}')

    unique, counts = np.unique(out.test_cllab, return_counts=True)
    logging.info(f'\n\nTest set (N = {X_ts.shape[0]})\n')
    for a, b in zip(unique, counts):
        logging.info(f'N subjects in cluster {a}: {b}')

    umap_tr = X_tr
    umap_ts = X_ts
    #     umap_tr = transform.fit_transform(X_tr)
    #     umap_ts = transform.fit_transform(X_ts)

    flatui = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
              "#7f7f7f", "#bcbd22", "#17becf", "#8c564b", "#a55194"]
    _scatter_plot(umap_tr,
                  [(gui, cl) for gui, cl in zip(data_tr.index, out.train_cllab)],
                  flatui,
                  10, 20, {str(ncl): '-'.join(['cluster', str(ncl)]) for ncl in sorted(np.unique(out.train_cllab))},
                  title=f'Subgroups of UMAP preprocessed Vineland TRAINING '
                        f'dataset (period: {period} -- level: {hierarchy})')

    _scatter_plot(umap_ts,
                  [(gui, cl) for gui, cl in zip(data_ts.index, out[2])],
                  flatui,
                  10, 20, {str(ncl): '-'.join(['cluster', str(ncl)]) for ncl in sorted(np.unique(out[2]))},
                  title=f'Subgroups of UMAP preprocessed Vineland TEST '
                        f'dataset (period: {period} -- level: {hierarchy})')

    # Plot heatmap
    raw_ts = raw_ts.loc[data_ts.index]
    raw_ts['cluster'] = out[2]
    mis_perc = {}
    for lab in np.unique(out.test_cllab):
        ts_rid = raw_ts.loc[raw_ts.cluster == lab].copy()
        mis_perc[lab] = (sum([ts_rid.iloc[indx].isna().astype(int) for indx in range(ts_rid.shape[0])]) / ts_rid.shape[
            0]) * 100
    # Save missingness percentage dataset
    mis_count_df = pd.DataFrame(
        [raw_ts[[c for c in raw_ts.columns if c != 'cluster']].iloc[indx].isna().astype(int) for indx in
         range(raw_ts.shape[0])])
    mis_count_df['cluster'] = raw_ts.cluster
    cl_labels = np.repeat(sorted(raw_ts.cluster.unique().astype(str)), raw_ts.shape[1])
    feat = np.array(raw_ts.columns)
    values = np.array(mis_perc[0])
    for lab in range(1, len(raw_ts.cluster.unique())):
        feat = np.append(feat, np.array(raw_ts.columns))
        values = np.append(values, np.array(mis_perc[lab]))
    plot_miss_heat(raw_ts, cl_labels, feat, values, period=period, hierarchy=hierarchy)
    return out, mis_count_df


def plot_miss_heat(X_ts, cl_labels, feat, values, period, hierarchy):
    heat_df = pd.DataFrame({'cl_labels': cl_labels, 'feat': feat, 'values': values})

    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2",
              "#dfccce", "#ddb7b1", "#cc7878", "#933b41",
              "#550b1d"][::-1]

    mapper = LinearColorMapper(palette=colors,
                               low=0,
                               high=100)
    p = figure(x_range=[c for c in X_ts.columns if c != 'cluster'],
               y_range=[str(lab) for lab in sorted(np.unique(cl_labels))],
               x_axis_location="above",
               plot_width=900,
               plot_height=300,
               toolbar_location='below',
               title=f'Percentage of originally missing information by subcluster '
                     f'for each feature {hierarchy} of the Vineland TEST dataset at period {period}')

    TOOLTIPS = [('score', '@values')]

    p.add_tools(HoverTool(tooltips=TOOLTIPS))

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.xaxis.major_label_text_font_size = "7pt"
    p.yaxis.major_label_text_font_size = "7pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 4

    p.rect(x="feat", y="cl_labels",
           width=1, height=1,
           source=heat_df,
           fill_color={'field': 'values',
                       'transform': mapper},
           line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%.2f"),
                         label_standoff=8, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
    show(p)


# Load demographic info and longitudinal dataset for Vineland
demo_info = pkl.load(open('./out/demographics.pkl', 'rb'))
long_dict = pkl.load(open('./out/longitudinal_data.pkl', 'rb'))
vineland_df = long_dict[1]['vineland']
# Select only P1-P3 interview periods
vineland = vineland_df.loc[vineland_df.interview_period.str.contains('P1|P2|P3')].copy()
vineland.drop(['interview_date', 'relationship'], axis=1, inplace=True)
vineland.sort_values(['subjectkey', 'interview_period'], inplace=True)

# Drop duplicated rows with the same interview_period, retain the entry with the
# highest number of available scores
mask = vineland.reset_index().duplicated(['subjectkey', 'interview_period'], keep=False)
vinedup = vineland.loc[mask.tolist()].copy()
gui_list = np.unique(vinedup.index)
mask_drop = []
for idx in gui_list:
    cou = [
        sum(vinedup.loc[idx].iloc[n][[c for c in vineland.columns if re.search('score|total', c)]].isna().astype(int))
        for n in range(vinedup.loc[idx].shape[0])]
    tmp = [False] * vinedup.loc[idx].shape[0]
    tmp[cou.index(min(cou))] = True
    mask_drop.extend(tmp)
vineland = pd.concat([vinedup.loc[mask_drop], vineland.loc[(~mask).tolist()]])

vineland.sort_values(['subjectkey', 'interview_period'], inplace=True)

uni, cou = np.unique(vineland.index, return_counts=True)
add_count = []
for c in cou:
    add_count.extend([c] * c)

vineland['intersection_count'] = add_count

df_demo = {}
for el in demo_info:
    if el.gui in vineland.index:
        df_demo.setdefault('subjectkey', list()).append(el.gui)
        df_demo.setdefault('sex', list()).append(el.sex)
        df_demo.setdefault('site', list()).append(el.dataset_id)
        df_demo.setdefault('phenotype', list()).append(el.phenotype)
df_demo = pd.DataFrame(df_demo, index=df_demo['subjectkey'])
df_demo.sort_values('subjectkey', inplace=True)
# df_demo.to_csv('./out/VINELANDsubjinfo.csv', index_label='subjectkey')

# Drop subjects with too much missing information
raw_df = {'P1': imp.prepare_imputation({'vineland': vineland}, 'P1', 0.60)['vineland'],
          'P2': imp.prepare_imputation({'vineland': vineland}, 'P2', 0.60)['vineland'],
          'P3': imp.prepare_imputation({'vineland': vineland}, 'P3', 0.60)['vineland']}

chk_str = 'interview_period|countna|intersection'
tr_dict, ts_dict = {}, {}
for k in raw_df.keys():
    idx_tr, idx_ts = train_test_split(raw_df[k].index,
                                      stratify=raw_df[k][['countna', 'intersection_count']],
                                      test_size=0.45,
                                      random_state=42)  # Train/Test stratified by percentage of missing
    # information and availability between the three datasets
    tr_dict[k] = raw_df[k].loc[idx_tr][[c for c in raw_df[k].columns if not re.search(chk_str, c)]]
    ts_dict[k] = raw_df[k].loc[idx_ts][[c for c in raw_df[k].columns if not re.search(chk_str, c)]]

# plot_intersection(tr_dict)
# plot_intersection(ts_dict)

imp_dict = {}
for k in raw_df.keys():
    imp_dict[k] = imp.impute(tr_dict[k], ts_dict[k])

# Run relative validation
vinesubdomain_names = [c for c in imp_dict['P1'][1].columns if re.search('vscore', c)]
vinedomain_names = [c for c in imp_dict['P1'][1].columns if
                    re.search('communication|livingskills|socialization', c)]
vine_oth_names = [c for c in imp_dict['P1'][1].columns if
                  re.search('interview', c)]

# Run relative clustering validation for each time period
for k in imp_dict.keys():
    vinesubdomain_out, vinesubdomain_mis_count = run_validation(
        imp_dict[k][0][vinesubdomain_names].copy(),
        imp_dict[k][1][vinesubdomain_names].copy(),
        ts_dict[k][vinesubdomain_names].copy(),
        n_neighbors=30, period=k, hierarchy='L1-subdomains')
    imp_dict[k][1]['cluster_subdomain'] = vinesubdomain_out.test_cllab + 1

    vinedomain_out, vinedomain_mis_count = run_validation(
        imp_dict[k][0][vinedomain_names].copy(),
        imp_dict[k][1][vinedomain_names].copy(),
        ts_dict[k][vinedomain_names].copy(),
        n_neighbors=30, period=k, hierarchy='L2-domains')
    imp_dict[k][1]['cluster_domain'] = vinedomain_out.test_cllab + 1

    imp_dict[k][1].sort_values(['subjectkey'], inplace=True)
    imp_dict[k][1][['sex', 'site', 'phenotype']] = df_demo.loc[imp_dict[k][1].index][['sex', 'site', 'phenotype']]
    imp_dict[k][1].to_csv(f'./out/VINELANDdata{k}.csv', index_label='subjectkey')

    with open(f'./out/VINELANDmiss_perc_L1{k}.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['gui'] + list(vinesubdomain_mis_count.columns))
        for row in vinesubdomain_mis_count.iterrows():
            wr.writerow([row[0]] + list(row[1]))
    with open(f'./out/VINELANDmiss_perc_L2{k}.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['gui'] + list(vinedomain_mis_count.columns))
        for row in vinedomain_mis_count.iterrows():
            wr.writerow([row[0]] + list(row[1]))

    # Save distance matrices
    feat_dict = {'subdomain': [vinesubdomain_names, vinesubdomain_out],
                 'domain': [vinedomain_names, vinedomain_out]}
    scaler = MinMaxScaler()
    for kfeat, featv in feat_dict.items():
        train = imp_dict[k][0]
        test = imp_dict[k][1]
        df_tr = train[featv[0]].copy()
        df_cl_tr = pd.DataFrame({'subjectkey': df_tr.index, 'cluster': featv[1].train_cllab + 1},
                                index=df_tr.index).sort_values('cluster')
        sc_tr = scaler.fit_transform(df_tr.loc[df_cl_tr.index])
        distmat_tr = pd.DataFrame(cdist(sc_tr, sc_tr), columns=df_cl_tr.index)
        distmat_tr.index = df_cl_tr.index
        distmat_tr[f'cluster_{kfeat}'] = df_cl_tr['cluster'].astype(str)

        df_ts = test[featv[0]].copy()
        df_cl_ts = pd.DataFrame({'subjectkey': df_ts.index, 'cluster': test[f'cluster_{kfeat}']},
                                index=df_ts.index).sort_values('cluster')
        sc_ts = scaler.fit_transform(df_ts.loc[df_cl_ts.index])
        distmat_ts = pd.DataFrame(cdist(sc_ts, sc_ts), columns=df_cl_ts.index)
        distmat_ts.index = df_cl_ts.index
        distmat_ts[f'cluster_{kfeat}'] = df_cl_ts['cluster'].astype(str)

        with open(f'./out/VINELAND_dist{kfeat.upper()}TR{k}.csv', 'w') as f:
            wr = csv.writer(f)
            wr.writerow(['subjectkey'] + distmat_tr.columns.tolist())
            for gui, row in distmat_tr.iterrows():
                wr.writerow([gui] + row.tolist())
        with open(f'./out/VINELAND_dist{kfeat.upper()}TS{k}.csv', 'w') as f:
            wr = csv.writer(f)
            wr.writerow(['subjectkey'] + distmat_ts.columns.tolist())
            for gui, row in distmat_ts.iterrows():
                wr.writerow([gui] + row.tolist())