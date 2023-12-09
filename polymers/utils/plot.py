import matplotlib.pyplot as plt

__all__ = ['plot_walk']

def plot_walk(data, gap=0.15, head_width=0.05, head_length=0.1, dx=0.05, save=None, grid=True):
    """
    Plots a simple random walk of length n
    
    Args:
        data (dict): a dictionary with keys 'x' and 'y' 
            containing the x and y coordinates of the walk, 
            along with the number of steps taken 'n', and 
            an optional key 'early_stop' indicating whether
            the walk was stopped early for any reason.
        
        gap (float): the fraction of a single arrow to 
            remove from the beginning and end of each arrow, 
            for visualization purposes.
        
        head_width (float): the width of the arrow head
        
        head_length (float): the length of the arrow head
        
        dx (float): the distance to shift the step up and
        down to avoid overlapping arrows
        
        save (str): the filename to save the plot to (optional)
        
    Returns:
        Plot of a simple random walk of length n
    """
    ZBOTTOM, ZTOP = -999, 999
    
    if not isinstance(data, dict):
        data = dict(x=data[:,0], y=data[:,1], n=len(data))
    x = data['x']
    y = data['y']
    show_early_finish = data.get('early_stop', False)
    ntries = data.get('ntries', 1)
    
    plt.figure(figsize = (8, 8), dpi=100)
    plt.scatter(x[0], y[0], c='limegreen', marker='s', s=50, label = 'Start', zorder=ZBOTTOM)
    plt.scatter(x[-1], y[-1], c='orangered', marker='s', s=50, label = 'End', zorder=ZBOTTOM)
    
    # plot each step as an arrow
    for i in range(1,len(x)):
        if (x[i]-x[i-1]) == 0:     # |
            yshift = 0
        elif (x[i]-x[i-1]) > 0:    # -->
            yshift = dx
        else:                      # <--
            yshift = -dx
            
        if (y[i]-y[i-1]) == 0:     # --
            xshift = 0
        elif (y[i]-y[i-1]) > 0:    # ^
            xshift = dx
        else:                      # v
            xshift = -dx
        
        diff_x = (x[i] - x[i-1])
        diff_y = (y[i] - y[i-1])

        plt.arrow(x[i-1] + diff_x*gap + xshift,
                  y[i-1] + diff_y*gap + yshift,
                  diff_x*(1 - 2*gap),
                  diff_y*(1 - 2*gap),
                width = 0.01,
                color = 'k',
                head_width=head_width, 
                head_length=head_length,
                length_includes_head=True,
                zorder=ZTOP)

        
    # plot the lattice
    xmin, xmax = int(min(x)), int(max(x)+1)
    ymin, ymax = int(min(y)), int(max(y)+1)
    xlen = xmax - xmin
    ylen = ymax - ymin
    if xlen > ylen:
        to_add = (xlen - ylen)//2
        ymin -= to_add
        ymax += to_add
    else:
        to_add = (ylen - xlen)//2
        xmin -= to_add
        xmax += to_add

    if grid:
        for i in range(int(xmin-0.1*xlen), int(xmax+0.1*xlen+1)):
            plt.axvline(i, c='grey', lw=0.5, zorder=ZBOTTOM-1, alpha=0.5)
        for i in range(int(ymin-0.1*ylen), int(ymax+0.1*ylen+1)):
            plt.axhline(i, c='grey', lw=0.5, zorder=ZBOTTOM-1, alpha=0.5)

    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.axis('off')    

    plt.legend(fontsize=15, framealpha=1, ncols=2, loc='upper right')
    
    txt = f'$N = {len(x)-1}$'
    if show_early_finish:
        txt += f'\nFAILED ($\\neq{data["n"]}$)'
    if ntries > 1:
        txt += f'\n$n_{{tries}} = {ntries}$'
    
    plt.text(0.05, 0.935, txt, fontsize=20, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=1, edgecolor='lightgrey', boxstyle='round,pad=0.2',
                       ))
    
    if save:
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()