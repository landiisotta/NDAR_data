import re
import numpy as np
import pandas as pd
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, \
    ColorBar, HoverTool
from bokeh.plotting import figure, show
from bokeh.io import export_svgs
from math import pi
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import math


def heatmap_missing(df, save_fig=None):
    """
    This function takes as input a cross-sectional dataframe with selected features
    and output a heatmap with red boxes where a score is missing and green boxes where
    the score is available. Rows are GUIs and columns are features.

    Parameters
    ----------
    df: dataframe
    save_fig: str
        Plot name with folder path, if None plot is only visualized
    """
    bin_dict = {col: list(df[col].isna().astype(int)) for col in df.columns if
                not re.match('interview|relationship', col)}
    bin_df = pd.DataFrame(bin_dict)
    bin_df.index = df.index

    gui_list = list(bin_df.index) * len(bin_df.columns)
    feat = np.repeat([col for col in bin_df.columns if not re.match('interview|relationship', col)],
                     bin_df.shape[0], axis=0).tolist()
    values = []
    for col in bin_df.columns:
        values += list(bin_df[col])

    heat_df = pd.DataFrame({'gui': gui_list, 'feat': feat, 'values': values})

    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2",
              "#dfccce", "#ddb7b1", "#cc7878", "#933b41",
              "#550b1d"]

    mapper = LinearColorMapper(palette=colors,
                               low=0,
                               high=1)
    p = figure(x_range=list(bin_df.columns),
               y_range=list(bin_df.index),
               x_axis_location="above",
               plot_width=900,
               plot_height=4000,
               toolbar_location='below')

    # TOOLTIPS = [('clpid', '@clpid'),
    #             ('sex', '@sex'),
    #             ('bdate', '@bdate'),
    #             ('feat', '@feat'),
    #             ('score', '@score'),
    #             ('n_enc', '@n_enc')]

    # p.add_tools(HoverTool(tooltips=TOOLTIPS))

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.xaxis.major_label_text_font_size = "7pt"
    p.yaxis.major_label_text_font_size = "2pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 4

    p.rect(x="feat", y="gui",
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
    if save_fig is not None:
        p.output_backend = 'svg'
        export_svgs(p, f'{save_fig}.svg')
    else:
        show(p)


def heatmap_missing_freq(df, title, save_fig=None):
    """
    This function takes as input a cross-sectional dataframe with selected features
    and output a heatmap with red boxes where a score is missing and green boxes where
    the score is available. Rows are GUIs and columns are features.

    Parameters
    ----------
    df: dataframe
    title: str
        Instrument name
    save_fig: str
        Plot name with folder path, if None plot is only visualized
    """
    bin_dict = {col: list(df[col].isna().astype(int)) for col in df.columns if
                not re.match('interview|relationship|respond', col)}
    bin_df = pd.DataFrame(bin_dict)
    bin_df.index = df.index

    gui_list = ['0', '1'] * len(bin_df.columns)
    feat = np.repeat([col for col in bin_df.columns if not re.match('interview|relationship|respond', col)],
                     2, axis=0).tolist()
    values = []
    for col in bin_df.columns:
        perc = sum(list(bin_df[col])) / bin_df.shape[0] * 100
        values += [perc, 100 - perc]

    heat_df = pd.DataFrame({'gui': gui_list, 'feat': feat, 'values': values})

    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2",
              "#dfccce", "#ddb7b1", "#cc7878", "#933b41",
              "#550b1d"][::-1]

    mapper = LinearColorMapper(palette=colors,
                               low=0,
                               high=100)
    p = figure(x_range=list(bin_df.columns),
               y_range=['0', '1'],
               x_axis_location="above",
               plot_width=700,
               plot_height=300,
               toolbar_location='below',
               title=title)

    TOOLTIPS = [('score', '@values')]

    p.add_tools(HoverTool(tooltips=TOOLTIPS))

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.xaxis.major_label_text_font_size = "7pt"
    p.yaxis.major_label_text_font_size = "7pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 4

    p.rect(x="feat", y="gui",
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
    if save_fig is not None:
        p.output_backend = 'svg'
        export_svgs(p, f'{save_fig}.svg')
    else:
        show(p)


def plot_dist(df, feat, print_mean=False, size=(10, 15), name=None):
    """
    Function that plots a grid with score distributions
    for each feature considered. The plot is saved if requested.
    Each plot is labeled according to the feature name and the
    number of observation != NA.

    Parameters
    ----------
    df: dataframe
        pandas dataframe with only the desired columns
    feat: int
        Number of features to arrange the grid
    print_mean: bool
        whether to display mean (dashed) and median (solid) lines
    size: tuple
        tuple of int with width and height of grid
    name: str (default: None)
        path to the saved file
    """
    fig = plt.figure(figsize=size)
    i = 1
    fig.subplots_adjust(hspace=0.5, wspace=0.8)
    for c in df.columns:
        if feat % 3 == 0:
            ax = fig.add_subplot(int(feat / 3), 3, i)
        else:
            ax = fig.add_subplot(int(feat / 3) + 1, 3, i)
        ax.hist(df[c].dropna(), bins=50)
        if print_mean:
            ax.axvline(np.mean(df[c].dropna()), color='k', linestyle='dashed', linewidth=2)
            ax.axvline(np.median(df[c].dropna()), color='k', linestyle='solid', linewidth=2)
        plt.xlim(xmin=min(df[c].dropna()), xmax=max(df[c].dropna()))
        plt.title(f'{c} \n ({df[c].dropna().shape[0]})')
        i += 1
    if name is not None:
        fig.savefig(f'{name}', format='pdf')
    fig.show()


def plot_metrics(cv_score, figsize=(20, 10)):
    """
    Function that plot the average performance (i.e., normalized stability) over cross-validation
    for training and validation sets. Plot is saved in out folder.

    Parameters
    ----------
    cv_score: dictionary
        Dictionary with all the CV metrics as output by search_best_nclust() function.
    figsize: tuple (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(list(cv_score['train'].keys()),
            [np.mean(me) for me in cv_score['train'].values()],
            label='training set')
    ax.errorbar(list(cv_score['val'].keys()),
                [np.mean(me) for me in cv_score['val'].values()],
                [_confint(me) for me in cv_score['val'].values()],
                label='validation set')
    ax.legend()
    plt.xticks([lab for lab in cv_score['train'].keys()])
    plt.xlabel('Number of clusters')
    plt.ylabel('Normalized stability')
    plt.show()
    plt.savefig(f"./out/performance_cv_nclust{(len(cv_score['train'].keys()) + 1)}",
                format='pdf')


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


"""
Private functions
"""


def _confint(vect):
    """
    Parameters
    ----------
    vect: list (of performance scores)
    Returns
    ------
    float: value to +/- to stability error for error bars (95% CI)

    """
    error = np.mean(vect)
    return 1.96 * math.sqrt((error * (1 - error)) / len(vect))
