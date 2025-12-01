from typing import Optional

from matplotlib import path as mpl_path


def point_in_polygon(x, y, polygon: Optional[mpl_path.Path]):
    if not polygon:
        return False

    return polygon.contains_point((x, y))
