from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns
import matplotlib as mpl

import matplotlib.pyplot as plt

from numpy import ndarray
from torch import Tensor

from .figures_ import save_figure

def make_line_plot(
    data : list[int | float] | ndarray | Tensor, 
    title : str = '' , 
    xlabel : str = '' , 
    ylabel : str = '' , 
    filename : str = '.png' , 
    figsize : tuple[int , int] = (15 , 5) , 
    dpi : int = 400 , 
    save : bool = False
) -> Axes : 

    mpl.rcParams['agg.path.chunksize'] = 10000

    plt.figure(figsize = figsize)

    final_data : ndarray

    if isinstance(data , Tensor) : 
        final_data = data.cpu().numpy()

    elif isinstance(data , ndarray) : 
        final_data = data

    elif isinstance(data , (list)) : 
        final_data = np.array(data)

    plot_axes : Axes = sns.lineplot(data = final_data)

    plot_axes.set_title(title)
    plot_axes.set_xlabel(xlabel)
    plot_axes.set_ylabel(ylabel)

    figure : Figure | None = plot_axes.get_figure()

    if save : 
        save_figure(
            figure = figure ,
            filename = filename , 
            dpi = dpi
        )

    return plot_axes