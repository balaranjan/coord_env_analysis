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


def get_formula(points_wd, CN, site_symbol_map):
    symbols = [s[2] for s in points_wd[:CN]]
    elements = defaultdict(int)
    for s in symbols:
        elements[site_symbol_map[s]] += 1
    elements = [[k, v] for k, v in elements.items()]
    elements = sorted(elements, key=lambda x: x[1], reverse=True)
        
    formula = ""
    for e, c in elements:
        formula += f"{e}{'' if c==1 else c}"
    return formula



def find_linear_units(cif_path, results, element='Sb'):
    try:
        cif = Cif(cif_path, is_formatted=True)
        cif.compute_connections()
        conns = cif.connections
        loop_vals = cif._loop_values
        site_symbol_map = dict(zip([l for l in loop_vals[0]], [s for s in loop_vals[1]]))

        # results = {}
        for site, neighbors in conns.items():
            if not site_symbol_map[site] == element:
                continue
            
            neighbors = sorted(neighbors, key=lambda x: x[1])[:20]
            
            # sort
            distances = np.array([p[1] for p in neighbors])
            distances /= distances.min()

            gaps = np.array([distances[i] - distances[i-1] for i in range(1, len(distances))])
            ind_gaps = np.argsort(gaps)
            
            inner_gap_index = np.array(ind_gaps[::-1])
            outer_CN = int(inner_gap_index[0]) + 1
            
            same_neighbors = [n for n in neighbors[:outer_CN] if site_symbol_map[n[0]]==element]
            same_neighbors = sorted(same_neighbors, key=lambda x: x[1])
            # print(site, same_neighbors)
            
            if len(same_neighbors) > 1:
                center = same_neighbors[0][2]
                # straight = [center]  # keep the center atom's coordinate
                for i in range(len(same_neighbors)):
                    for j in range(i+1, len(same_neighbors)):
                        if point_in_line(center, same_neighbors[i][3], same_neighbors[j][3]):
                            results.append({
                                'cif': cif.file_name,
                                'site': site,
                                'center': center,
                                'outerCN': outer_CN,
                                # 'outerCN_formula': get_formula(neighbors, outer_CN, site_symbol_map),
                                'n1': [same_neighbors[i][k] for k in [0, 1, 3]],
                                'n2': [same_neighbors[j][k] for k in [0, 1, 3]]
                            })
            #             straight.append([
            #                 site, 
            #                 [same_neighbors[i][k] for k in [0, 1, 3]], 
            #                 [same_neighbors[j][k] for k in [0, 1, 3]]])
            # if len(straight) > 1:
            #     results[site] = straight
            #     results['cif'] = cif.file_name
        return 
    except:
        return
            
        
def mp_aux(*args):
    for arg in args:
        find_linear_units(**arg)
        
        
def run_parallel(cdir):
    manager = mp.Manager()
    results = manager.list()
    
    tasks = []
    for i, cif in enumerate(os.listdir(cdir)):
        if not cif.endswith(".cif"):
            continue
        
        # if i > 1000:
        #     break
        tasks.append({'cif_path': f"{cdir}{os.sep}{cif}", 'element': 'Sb', 'results': results})
        
    with mp.Pool(mp.cpu_count() - 2) as pool:
        pool.map(mp_aux, tasks)
        
    pool.close()
    pool.join()
    
    results = list(results)
    pd.DataFrame(results).to_csv('linear_units.csv', index=False)
        
        
if __name__ == "__main__":
    
    # print(point_in_line([2,2], [3, 3], [4, 4]))
    
    root = "/home/user/Documents/bala/2_CN4_in_Yb16MnSb11/data/im_full_occupancy"
    run_parallel(root)
    # results = []
    # find_linear_units(cif_path=f"{root}{os.sep}1815248.cif", results=results)
    # print(results)