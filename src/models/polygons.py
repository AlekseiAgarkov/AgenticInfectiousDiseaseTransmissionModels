from matplotlib import path as mpl_path


def point_in_polygon(x, y, polygon: mpl_path.Path):
    return polygon.contains_point((x, y))
