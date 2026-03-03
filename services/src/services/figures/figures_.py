from matplotlib.axes import Axes
from matplotlib.figure import Figure


def save_figure(
    figure : Figure | None ,  
    filename : str , 
    dpi : int = 300 , 
    bbox_inches : str = 'tight' ,
    pad_inches : float = 0.1 
) -> None : 

    if figure : 

        figure.savefig(
            filename , 
            dpi = dpi , 
            bbox_inches = bbox_inches , 
            pad_inches = pad_inches
        )

    else : 
        raise ValueError('No figure to save')