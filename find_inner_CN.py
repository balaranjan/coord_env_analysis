import multiprocessing.process
import os
from cifkit import Cif
import traceback
import multiprocessing as mp
import pandas as pd
import numpy as np
from cifkit.coordination.composition import count_connections_per_site
from scipy.spatial import ConvexHull
from skspatial.objects import Points
import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay
from collections import defaultdict
import multiprocessing



ideal_polyhedron_metrics = {
    # polyhedron: [[distances], [angles]]
    'tetrahedon': [[1., 1., 1., 1.], [60., 60., 60., ]],
    'square_pyramid': [1., 0., ]
}



def polyhedron_metrics(center, points_wd):
    points = [[p][0][0] for p in points_wd]
    # print(points)
    distances = []
    angles = []
    try:
        points = Points(points)
        if points.are_coplanar():
            print("Points are coplanar")
            
        try:
            hull = ConvexHull(points)
            
            # distance from faces
            for simplex in hull.simplices:
                vertices = points[simplex]
                normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
                offset = -np.dot(normal, vertices[0])
                
                distance = abs(normal[0]*center[0] + normal[1]*center[1] + normal[2]*center[2] + offset) / \
                            np.sqrt(np.square(normal).sum())
                distances.append(distance)
            
            for i in range(len(hull.vertices)):
                p1 = points[hull.vertices[i]]
                p2 = points[hull.vertices[(i + 1) % len(hull.vertices)]]
                p3 = points[hull.vertices[(i - 1) % len(hull.vertices)]]

                v1 = p2 - p1
                v2 = p3 - p1

                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                angle = np.degrees(angle)

                angles.append(round(float(angle), 2))
        except:
            print('Error while constructing convex-hull')
            print(traceback.format_exc())
    except:
        print('Error while checking co-planarity')
        print(traceback.format_exc())
        
    if distances:
        distances = np.array(distances)
        distances /= distances.min()
        distances = [round(float(d), 4) for d in distances]
         
    return distances, angles


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
        
        
def d_by_dmin(k, v, center, site_symbol_map):
    
    points_wd =[[p[3], p[1], p[0]] for p in v]
    inner_CN = None
    
    # sort
    points_wd = sorted(points_wd, key=lambda x: x[1])[:20]
    distances = np.array([p[1] for p in points_wd])
    distances /= distances.min()

    gaps = np.array([distances[i] - distances[i-1] for i in range(1, len(distances))])
    ind_gaps = np.argsort(gaps)
    
    inner_gap_index = np.array(ind_gaps[::-1])
    outer_CN = int(inner_gap_index[0]) + 1
    inner_gap_index = inner_gap_index - inner_gap_index[0]
    inner_gap_index = inner_gap_index[:5]
    inner_gap_index = inner_gap_index[inner_gap_index < 0]
    inner_gap_index += outer_CN
    inner_gap_index = inner_gap_index[inner_gap_index >= 4]
    if len(inner_gap_index):
        # keep only the lowest/first
        inner_CN = int(inner_gap_index[0])
        lowest_inner_CN = int(inner_gap_index.min())
        distances, angles = polyhedron_metrics(center, points_wd[:inner_CN])
        
        return {'site': k, 
                'outer_CN': outer_CN, 'formula_outer_CN': get_formula(points_wd, outer_CN, site_symbol_map),
                'inner_CN': inner_CN, 'formula_inner_CN': get_formula(points_wd, inner_CN, site_symbol_map),
                'lowest_inner_CN': lowest_inner_CN, 'formula_lowest_inner_CN': get_formula(points_wd, lowest_inner_CN, site_symbol_map),
                'distances': distances, 'angles': angles}
    return


def find_cenvs_s(cif_path):
    try:
        cif = Cif(cif_path, is_formatted=True)
        # label, symbol, mult, Wyckoff_symbol, x, y, z, occ
        cif.compute_connections()
        loop_vals = cif._loop_values
        site_symbol_map = dict(zip([l for l in loop_vals[0]], [s for s in loop_vals[1]]))

        conns = cif.connections
        data = None
        for i, (k, v) in enumerate(conns.items()):
            wyckoff = f"{loop_vals[2][i]}{loop_vals[3][i]}"
            central_atom = loop_vals[1][i]
            center = v[0][2]
            # points_wd =[[p[3], p[1]] for p in v]
            data = d_by_dmin(k, v, center, site_symbol_map)
            
            if data is not None:
                data['CIF#'] = cif_path.split(os.sep)[-1][:-4]
                data['central_atom'] = central_atom
                data['Wyckoff'] = wyckoff
                # results.append(data) 
    except:
        print(traceback.format_exc())
    return 


def find_cenvs(cif_path, results):
    try:
        cifn = cif_path.split(os.sep)[-1][:-4]
        cif = Cif(cif_path, is_formatted=True)
        # label, symbol, mult, Wyckoff_symbol, x, y, z, occ
        cif.compute_connections()
        loop_vals = cif._loop_values
        site_symbol_map = dict(zip([l for l in loop_vals[0]], [s for s in loop_vals[1]]))

        conns = cif.connections
        data = None
        for i, (k, v) in enumerate(conns.items()):
            wyckoff = f"{loop_vals[2][i]}{loop_vals[3][i]}"
            central_atom = loop_vals[1][i]
            center = v[0][2]
            data = d_by_dmin(k, v, center, site_symbol_map)
            
            if data is not None:
                data['CIF#'] = cifn
                data['central_atom'] = central_atom
                data['Wyckoff'] = wyckoff
                results.append(data) 
    except:
        print(f"Error while processing {cifn}")
        print(traceback.format_exc())
    return 


def aux_mp(*args):
    for arg in args:
        find_cenvs(**arg)


def run_parallel(root_dir, nmax=100):
    tasks = []
    manager = multiprocessing.Manager()
    results = manager.list()
    
    for i, c in enumerate(os.listdir(root_dir)):
        if not c.endswith(".cif"):
            continue
        
        # if i > nmax:
        #     break
        
        tasks.append({'cif_path': f"{root}{os.sep}{c}", 'results': results})
        
    with mp.Pool(multiprocessing.cpu_count() - 2) as pool:
        pool.map(aux_mp, tasks)
    pool.close()
    pool.join()
    
    results = list(results)
    df = pd.DataFrame(results)
    df[['CIF#', 'site','central_atom', 'Wyckoff', 'outer_CN', 'formula_outer_CN', 'inner_CN', 'formula_inner_CN', 'lowest_inner_CN', 'formula_lowest_inner_CN', 'distances', 'angles']].to_csv('innerCN_results.csv', index=False)
            
    
    
if __name__ == "__main__":
    import time
    root = "/home/user/Documents/bala/2_CN4_in_Yb16MnSb11/data/im_full_occupancy"
    # print(find_cenvs_s(cif_path=f"{root}{os.sep}1632013.cif"))
    t0 = time.time()
    run_parallel(root)
    print(round(time.time()-t0, 1))
    