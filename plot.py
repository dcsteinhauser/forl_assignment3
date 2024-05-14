import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import numpy as np

def plot_reward(vector, size, title, tdw = False, show = False):
    matplotlib.rcParams.update({'font.size': 22})
    # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
    if tdw:
        #use off the shelf colors
        cm = plt.get_cmap('viridis')
        norm = colors.Normalize(vmin=vector.min(), vmax=vector.max())

        
    else:    
        col_dict={  -100:"mediumturquoise",
                  -50:"yellow",
                -1:"lightgrey",
                0:"white"}
        cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
  
    

    # Let's also define the description of each category : 1 (blue) is Sea; 2 (red) is burnt, etc... Order should be respected here ! Or using another dict maybe could help.
    if tdw:
        labels = np.array([vector.min(),vector.max()])
    else:
        labels = np.array(["-100", "-50","-1","0"])
    len_lab = len(labels)

    # prepare normalizer
    ## Prepare bins for the normalizer
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    print(norm_bins)
    ## Make normalizer and formatter
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    # Plot our figure
    #plt.figure()
    fig,ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(vector.reshape(size,size), cmap=cm, norm=norm)
    #ax.tick_params(labelsize=20)
    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    cb = fig.colorbar(im, format=fmt, ticks=tickz)
    #cb.ticklabels_params(size = 20)
    if show:
        plt.show()
    else:
        plt.savefig('../plot/'+title+'.png')
        plt.savefig('../plot/'+title+'.pdf')

def plot_value_policy(values, policy, size, title, show = True):
    matplotlib.rcParams.update({'font.size': 10})
    # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...

    cmap = plt.get_cmap('viridis')
    
    norm = colors.Normalize(vmin=-10, vmax=values.max())

    fig,ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(values.reshape(size,size), cmap=cmap, norm=norm)
    #add values into the cells of the grid
    for i in range(size):
        for j in range(size):
            text = ax.text(j, i, f'{values[i*size+j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    #below tha value insert the arrow indicating the policy
    shape = [size, size]
    for s, a in enumerate(policy):    #acs optimal actions
            if a == 0: ##up
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]), 0, -0.3, head_width=0.05) 
            if a == 1: ##right
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  0.3, 0, head_width=0.05)
            if a == 2: ##down
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  0, 0.3, head_width=0.05)
            if a == 3: ##left
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  -0.3, 0, head_width=0.05) 
    ##add colorbar and set the ticks
    cb = fig.colorbar(im)
    cb.set_label('Value function')
    
    if show:
        plt.show()
    else:
        plt.savefig('../plot/'+title+'.png')
        plt.savefig('../plot/'+title+'.pdf')

def plot_reward_policy(w, policy, size, title, show = True):
    matplotlib.rcParams.update({'font.size': 22})
    # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...

    cmap = plt.get_cmap('viridis')
    
    norm = colors.Normalize(vmin=w.min(), vmax=w.max())

    fig,ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(w.reshape(size,size), cmap=cmap, norm=norm)
    #add values into the cells of the grid
    for i in range(size):
        for j in range(size):
            text = ax.text(j, i, f'{w[i*size+j]:.4f}',
                           ha="center", va="center", color="black", fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    #below tha value insert the arrow indicating the policy
    shape = [size, size]
    for s, a in enumerate(policy):    #acs optimal actions
            if a == 0: ##up
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]), 0, -0.3, head_width=0.05) 
            if a == 1: ##right
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  0.3, 0, head_width=0.05)
            if a == 2: ##down
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  0, 0.3, head_width=0.05)
            if a == 3: ##left
                plt.arrow(np.mod(s, shape[1]), int(s / shape[1]),  -0.3, 0, head_width=0.05) 
    ##add colorbar and set the ticks
    cb = fig.colorbar(im)
    cb.set_label('Reward function')
    
    if show:
        plt.show()
    else:
        plt.savefig('../plot/'+title+'.png')
        plt.savefig('../plot/'+title+'.pdf')