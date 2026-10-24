def scatter_helper(ax, xs, ys, zs=None, class_label = None):

    """
    plot a 2d or 3d scatter plot and color using given class labels if any
    """

    if zs is not None:
        if class_label is not None:
            sc = ax.scatter(xs, ys, zs, c=class_label, cmap='tab10', s=10)
        else:
            sc = ax.scatter(xs, ys, zs, s=10)
        return sc

    else:
        if class_label is not None:
            sc = ax.scatter(xs, ys, c=class_label, cmap='tab10', s=10)
        else:
            sc = ax.scatter(xs, ys, s=10)
        return sc
    
    return None

def plot_3d_scatters(X,title_name,y=None,custom_inx=[0,1,2]):

    """
    plot 4 subplots: 1st is a 3d scatter plot, the other 3 show 2d / pairwise marginals
    """

    fig, axs = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={})

    axs[0] = fig.add_subplot(1, 4, 1, projection='3d')
    xs,ys,zs = X[:, custom_inx[0]], X[:, custom_inx[1]], X[:, custom_inx[2]]
    scatter_helper(axs[0],xs,ys,zs,y)
    axs[0].set_title(title_name)

    # 2D scatter plots
    scatter_helper(axs[1],xs,ys,None,y)
    axs[1].set_title(f'{custom_inx[0]} vs {custom_inx[1]}')

    scatter_helper(axs[2],ys,zs,None,y)
    axs[2].set_title(f'{custom_inx[1]} vs {custom_inx[2]}')

    scatter_helper(axs[3],xs,zs,None,y)
    axs[3].set_title(f'{custom_inx[0]} vs {custom_inx[2]}')
        
    # plt.xlim(-2.5, 2.5)
    # plt.ylim(-2.5, 2.5)

    plt.tight_layout()
    plt.show()

