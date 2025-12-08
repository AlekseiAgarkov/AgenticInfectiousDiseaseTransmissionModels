from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from simulation_utils.time import msk_now_str

if __name__ == '__main__':
    p = ArgumentParser(prog='Agent & Neighbours generator',
                       description='This utility generates a square plane and populates it with agents. '
                                   'Each agent is randomly assigned coordinates and gets its neighbor assigned.')
    paths = p.add_argument_group('paths')
    paths.add_argument('-o', '--output_path', type=str, required=True, help="Data output folder")

    simulation_params = p.add_argument_group("parameters")
    simulation_params.add_argument('-r', '--random_seed', type=int, default=42, help="Random seed")
    simulation_params.add_argument('-a', '--agents_number', type=int, required=True,
                                   help="Number of agents to generate")
    simulation_params.add_argument('-n', '--neighbors', type=int, required=True,
                                   help="Number of neighbors per agent to assign")
    simulation_params.add_argument('-s', '--size', type=int, required=True, help="Length of square side")

    args = p.parse_args()

    min_coord, max_coord = 0, args.size
    agents_number = args.agents_number

    np.random.seed(args.random_seed)
    x_coords = np.random.uniform(low=min_coord, high=max_coord, size=agents_number).round(decimals=2)
    y_coords = np.random.uniform(low=min_coord, high=max_coord, size=agents_number).round(decimals=2)
    coords = np.column_stack((x_coords, y_coords))

    n_neighbors = args.neighbors + 1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(coords)
    distances, neighbors = nbrs.kneighbors(coords)
    neighbors_df = pd.DataFrame(data=coords, columns=['x', 'y'])
    neighbors_df['neighbors'] = neighbors[:, 1:].tolist()

    generation_ts_str = msk_now_str()
    neighbors_df.to_csv(
        f"{args.output_path}/NeighborsMap-{args.size}by{args.size}-Agents{args.agents_number}-Neighbors{args.neighbors}-{generation_ts_str}.csv",
        index=False)
