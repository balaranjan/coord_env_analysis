import os
from cifkit import Cif
import traceback
import multiprocessing as mp
import pandas as pd
import numpy as np
from collections import defaultdict


def point_in_line(point, start, end):
    point = np.array(point)
    start = np.array(start)
    end = np.array(end)
    
    if np.allclose(np.linalg.norm(start-end), (np.linalg.norm(start-point) + np.linalg.norm(end-point))):
        return True
    return False


def find_linear_units(cif_path, element='Sb'):
    
    cif = Cif(cif_path, is_formatted=True)
    cif.compute_connections()
    conns = cif.connections
    loop_vals = cif._loop_values
    site_symbol_map = dict(zip([l for l in loop_vals[0]], [s for s in loop_vals[1]]))

    results = {}
    for site, neighbors in conns.items():
        if not site_symbol_map[site] == element:
            continue
        
        same_neighbors = [n for n in neighbors if site_symbol_map[n[0]]==element]
        same_neighbors = sorted(same_neighbors, key=lambda x: x[1])[:20]
        # print(site, same_neighbors)
        
        center = same_neighbors[0][2]
        straight = [center]  # keep the center atom's coordinate
        for i in range(len(same_neighbors)):
            for j in range(i+1, len(same_neighbors)):
                if point_in_line(center, same_neighbors[i][3], same_neighbors[j][3]):
                    straight.append([
                        site, 
                        [same_neighbors[i][k] for k in [0, 1, 3]], 
                        [same_neighbors[j][k] for k in [0, 1, 3]]])
        if len(straight) > 1:
            results[site] = straight
    return results
        
        
        
if __name__ == "__main__":
    
    # print(point_in_line([2,2], [3, 3], [4, 4]))
    
    root = "/home/bala/Documents/data/not_prototype_CIFs"
    print(find_linear_units(cif_path=f"{root}{os.sep}1632013.cif"))