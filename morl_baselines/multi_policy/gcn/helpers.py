import numpy as np
from matplotlib import pyplot as plt

def gen_line_plot_grid(line, grid_x_size, grid_y_size):
    """Generates a grid_x_max * grid_y_max grid where each grid is valued by the frequency it appears in the generated lines.
    Essentially creates a grid of the given line to plot later on.

    Args:
        line (list): list of generated lines of the model
        grid_x_max (int): nr of lines in the grid
        grid_y_mask (int): nr of columns in the grid
    """
    data = np.zeros((grid_x_size, grid_y_size))

    for station in line:
        data[station[0], station[1]] += 1
 
    return data

def highlight_cells(cells, ax, **kwargs):
    """Highlights a cell in a grid plot. https://stackoverflow.com/questions/56654952/how-to-mark-cells-in-matplotlib-pyplot-imshow-drawing-cell-borders
    """
    for cell in cells:
        (y, x) = cell
        rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
        ax.add_patch(rect)
    return rect

def crowding_distance(points):
    """Compute the crowding distance of a set of points."""
    # first normalize across dimensions
    points = (points - points.min(axis=0)) / (points.ptp(axis=0) + 1e-8)
    # sort points per dimension
    dim_sorted = np.argsort(points, axis=0)
    point_sorted = np.take_along_axis(points, dim_sorted, axis=0)
    # compute distances between lower and higher point
    distances = np.abs(point_sorted[:-2] - point_sorted[2:])
    # pad extrema's with 1, for each dimension
    distances = np.pad(distances, ((1,), (0,)), constant_values=1)
    # sum distances of each dimension of the same point
    crowding = np.zeros(points.shape)
    crowding[dim_sorted, np.arange(points.shape[-1])] = distances
    crowding = np.sum(crowding, axis=-1)
    return crowding

