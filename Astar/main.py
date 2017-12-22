# Advanced Artificial Intelligence Lab 3 - Search
#
# <https://sustech-cs-courses.github.io/AAI/lab/3/>

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from lab3 import greedysearch, astarsearch


def visualize(mapsize, blocks, init, goal, path=None):
    if path :
        path = np.array(path)
    """Plot map, blocks, starting point and destination
    """
    fig = plt.figure(0)
    axes = fig.add_subplot(111)

    # init and goal positions
    axes.plot(init[0], init[1], 'ro')
    axes.plot(goal[0], goal[1], 'go')

    # blocks
    for b in blocks:
        rec = patches.Rectangle(b, 1, 1, fc='silver')
        axes.add_patch(rec)

    if path is not None:
        axes.plot(path[:, 0], path[:, 1], '-', color='orange')

    # map
    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([0, mapsize[0]])
    axes.set_ylim([0, mapsize[1]])
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.set_xticks(np.arange(mapsize[0]))
    axes.set_yticks(np.arange(mapsize[1]))
    axes.tick_params(length=0)
    axes.grid()

    plt.show()
    ### Uncomment this line if you are using IPython/Jupyter notebook ###
    # plt.pause(1)


def main():
    # Problem settings
    ### Design your own map ###
    init = [2, 7]
    goal = [18, 11]
    mapsize = (21, 13)
    # The simplest example of blocks is
    # `blocks = ((9, 5), (10, 6), ...)`
    blocks = np.array([(6, i) for i in range(4, 13)] +
                      [(13, i) for i in range(0, 8)])

    #visualize(mapsize, blocks, init, goal)

    path = greedysearch(mapsize, blocks, init, goal)

    visualize(mapsize,blocks,  init, goal, path)
    
    #path = astarsearch(mapsize, blocks, init, goal)
    #visualize(mapsize, blocks, init, goal, path)

if __name__ == '__main__':
    main()
