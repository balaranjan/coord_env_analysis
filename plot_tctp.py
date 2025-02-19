from cifkit import Cif
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import traceback
from cifkit.utils import unit
from scipy.spatial import ConvexHull
import os


def has_capped_atom(p1, p2, third_vertex, capping_sites, non_layer_axes):
  
    points = [p[-1][non_layer_axes] for p in capping_sites]

    x1, y1 = p1
    x2, y2 = p2
    m = (y2 - y1) / (x2 - x1)  

    b = y1 - m * x1
    
    A = m
    B = -1
    C = b
    p_hat = lambda p: A*p[0] + B*p[1] + C
    
    phs = []
    third_vertex_positive = p_hat(third_vertex) > 0
    points_opposite_to_third_vertex = 0
    capping_site = None

    for i, point in enumerate(points):
        ph = p_hat(point)
        phs.append(float(ph))
        if third_vertex_positive and ph < 0:
            points_opposite_to_third_vertex += 1
            capping_site = capping_sites[i]
        elif ph > 0 and not third_vertex_positive:
            points_opposite_to_third_vertex += 1
            capping_site = capping_sites[i]
            
    res = True
    if points_opposite_to_third_vertex != 1:
        res = False
        
    return res, capping_site


def CN_numbers_of_site(v):
    
    points_wd =[[p[3], p[1], p[0]] for p in v]
    
    # sort
    points_wd = sorted(points_wd, key=lambda x: x[1])[:20]
    distances = np.array([p[1] for p in points_wd])
    distances /= distances.min()

    gaps = np.array([distances[i] - distances[i-1] for i in range(1, len(distances))])
    ind_gaps = np.argsort(gaps)
    
    CN_values = np.array(ind_gaps[::-1]) + 1
    
    return CN_values


def tricapped_trigonal_prism_data(site, layer_axis, points_wd):
    """
    Check for the presence of tri capped trigonal prism
    """
    center = points_wd[0][2]
    
    # drop center from points_wd
    points_wd = [[p[0], p[1], np.array(p[3])] for p in points_wd]
    
    non_layer_axes = np.array([0, 1, 2]) != layer_axis
    capping_inds = [i for i in range(len(points_wd)) if points_wd[i][2][layer_axis] == center[layer_axis]]
    capping_sites = [p for p in points_wd if p[2][layer_axis] == center[layer_axis]]
                
    tsn = [points_wd[c] for c in range(len(points_wd)) if c not in capping_inds]
    uh = set([round(p[-1][layer_axis], 3) for p in tsn])
    triangle_sites = {}
    for k in uh:
        val = []
        for p in tsn:
            if round(p[-1][layer_axis], 3) == k:
                val.append(p)
        triangle_sites[k] = val
             
    # pair capping site to triangle edges
    # layer_1, layer_2 = sorted(list(triangle_sites.keys()))
    layer_1 = sorted(list(triangle_sites.keys()))[0]
    faces = [[0, 1, 2], [0, 2, 1], [1, 2, 0]]

    tctp_data = {'center_site': site, 'center_coordinate': center,
                 'tctp_present': False, 'top_layer': layer_1}
    cap_atom_present = []

    for i, face in enumerate(faces):
        face_points = [triangle_sites[layer_1][face[j]][2][non_layer_axes] for j in face]
        face_points = np.array(face_points)
        
        try:
            _has_capped_atom, capping_site = \
                has_capped_atom(p1=face_points[0], 
                                p2=face_points[1], 
                                third_vertex=face_points[2], 
                                capping_sites=capping_sites,
                                non_layer_axes=non_layer_axes)
            cap_atom_present.append(_has_capped_atom)
            # tctp_data[f"triangle_face_{i}"] = [[triangle_sites[layer_1][face[j]] for j in face[:2]], [triangle_sites[layer_2][face[j]] for j in face[:2]]]
            # tctp_data[f"cap_face_{i}"] = capping_site
            tctp_data['triangle'] = face_points
            tctp_data['caps'] = capping_sites

        except:
            print(traceback.format_exc())
    if all(cap_atom_present):
        tctp_data['tctp_present'] = True
    
    return tctp_data

    
def get_data(cif_path):
    
    cif = Cif(cif_path)
    unitcell_points = cif.unitcell_points
    coordinates = np.array([c[:-1] for c in unitcell_points])
    
    # drop third axis
    layer_vals, layer_index = None, None
    for i in range(3):
        if len(np.unique(np.abs(coordinates[:, i]))) == 2:
            layer_vals = np.unique(coordinates[:, i])
            layer_index = i

    # remove third layer
    if len(layer_vals) == 3:
        coordinates = coordinates[coordinates[:, layer_index] != layer_vals[0]]
    
    cif.compute_connections()
    conns = cif.connections
    
    tctp_cif = {'layer_axis': layer_index} 
    tctp_cif['non_layer_axes'] = np.array([0, 1, 2]) != layer_index
    site_data = {}
    for site, points_wd in conns.items():
        CN_vals = CN_numbers_of_site(points_wd)
        if CN_vals[0] != 9:
            continue
        
        points_wd = sorted(points_wd, key=lambda x: x[1])[:9]
        _site_data = tricapped_trigonal_prism_data(site=site, 
                                                  layer_axis=layer_index, 
                                                  points_wd=points_wd)
        
        if _site_data['tctp_present']:
            site_data[site] = _site_data
            print(site)

    if site_data:
        tctp_cif['site_data'] = site_data
        return tctp_cif
    return None


def plot_cell(cif_path, title=""):
    
    cif_data = get_data(cif_path)
    if cif_data is None:
        return
    cif = Cif(cif_path)
    
    non_layer_axes = cif_data['non_layer_axes']
    layer_axis = cif_data['layer_axis']
    unitcell_lengths = cif.unitcell_lengths
    unitcell_angles = cif.unitcell_angles
    unitcell_points = cif.unitcell_points
    supercell_points = cif.supercell_points
    loop_vals = cif._loop_values
    site_symbol_map = dict(zip([l for l in loop_vals[0]], [s for s in loop_vals[1]]))
    
    unique = []
    unique_unitcell_points = []
    for i in range(len(unitcell_points)):
        s = unitcell_points[i][-1]
        coord = np.array(unitcell_points[i][:3])
        coord[coord > 1.0] -= 1.0
        coord[coord < 0.0] += 1.0
        
        cart = unit.fractional_to_cartesian(coord,# unitcell_points[i][:3],
                                                          unitcell_lengths, 
                                                          unitcell_angles)
        if len(unique):
            if not np.any(np.linalg.norm(np.vstack([u[non_layer_axes] for u in unique]) - coord[non_layer_axes], axis=1) == 0.0):
                unique.append(coord)
                unique_unitcell_points.append([np.array(cart), s])
        else:
            unique.append(coord)
            unique_unitcell_points.append([np.array(cart), s])
            
    unitcell_points = unique_unitcell_points
    for i in range(len(supercell_points)):
        s = supercell_points[i][-1]
        cart = unit.fractional_to_cartesian(supercell_points[i][:3],
                                            unitcell_lengths,
                                            unitcell_angles)
        supercell_points[i] = [np.array(cart), s]
    
    up = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0], 
                   [0.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0],
                   [1.0, 0.0, 0.0],
                   [1.0, 0.0, 1.0],
                   [1.0, 1.0, 0.0],
                   [1.0, 1.0, 1.0]])
    
    cell = np.array([unit.fractional_to_cartesian(p, unitcell_lengths, unitcell_angles).tolist() for p in up])
    cell = cell[:, non_layer_axes]
    unitcell_lengths = np.array(unitcell_lengths)        
    unitcell_lengths = unitcell_lengths[non_layer_axes]
    
    # plot cell
    x1 = unitcell_lengths[1] * np.cos(unitcell_angles[layer_axis])
    y1 = unitcell_lengths[1] * np.sin(unitcell_angles[layer_axis])
    plt.close()

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.plot([0., unitcell_lengths[0]], [0., 0.], c='k')
    plt.plot([0., x1], [0., y1], c='k')
    plt.plot([x1, x1+unitcell_lengths[0]], [y1, y1], c='k')
    plt.plot([unitcell_lengths[0], x1+unitcell_lengths[0]], [0., y1], c='k')
    
    colors = ['black', 'dimgrey', 'grey', 'darkgrey', 'lightcoral', 'indianred', 'brown', 'maroon', 'red', 'salmon', 'tomato', 'coral', 'orangered', 
              'sienna', 'chocolate', 'saddlebrown', 'peru','darkorange', 'burlywood', 'tan', 'navajowhite', 'blanchedalmond', 'moccasin', 'orange', 'wheat', 'darkgoldenrod', 
              'goldenrod', 'gold', 'khaki', 'darkkhaki', 'olive', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'greenyellow', 'lawngreen', 'darkseagreen', 
              'lightgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'mediumspringgreen', 'aquamarine', 'turquoise', 'lightseagreen', 'paleturquoise', 'darkslategrey', 
              'teal', 'aqua', 'darkturquoise', 'cadetblue', 'deepskyblue', 'lightskyblue', 'steelblue', 'dodgerblue', 'cornflowerblue', 'royalblue', 'navy', 'blue', 'slateblue', 
              'darkslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkviolet', 'purple', 'fuchsia', 'mediumvioletred', 'deeppink', 'hotpink', 'crimson']
    elems = ['B', 'Al', 'Th', 'W', 'Cu', 'Be', 'Au', 'Pd', 'Nd', 'Mn', 'Ho', 'Sc', 'Yb', 'Ru', 'Nb', 'Am', 'Zr', 'Os', 'Te', 'La', 'Hg', 'Li', 'K', 'Er', 'Zn', 'Sn', 'Ga', 'In', 
             'V', 'Cm', 'Sr', 'Tl', 'As', 'Se', 'Cd', 'Ta', 'Sb', 'Pt', 'Rb', 'Ge', 'Ir', 'Eu', 'U', 'Rh', 'Tm', 'Mo', 'Np', 'Si', 'Bi', 'Ti', 'Pr', 'Ag', 'Co', 'Gd', 'Sm', 'Tb', 
             'Fe', 'Hf', 'Mg', 'Ba', 'Ce', 'Y', 'Na', 'P', 'Cr', 'Re', 'Lu', 'Ni', 'Pb', 'C', 'Ca', 'Cs', 'Dy', 'Pu']
    
    color_labels = dict(zip(elems, colors))
    
    tctp_by_height = {}
    
    for k, _ in cif_data['site_data'].items():
        inside_points = [p for p in unitcell_points if p[1]==k]
        for p in inside_points:
            point_wd = get_first_9_neighbors(p, supercell_points)

            try:
                tctp_data_d = tricapped_trigonal_prism_data(site=p[1], layer_axis=layer_axis, points_wd=point_wd)
                center_coordinate = np.array(tctp_data_d["center_coordinate"])
                triangle = tctp_data_d['triangle']
                caps = tctp_data_d['caps']

                center_coordinate = center_coordinate[non_layer_axes]
                layer_height = round(caps[0][2][layer_axis], 3)
                
                if layer_height in tctp_by_height:
                    tctp_by_height[layer_height].append([center_coordinate, triangle, caps])
                else:
                    tctp_by_height[layer_height] = [[center_coordinate, triangle, caps]]

            except:
                print("Error", p)
                print(traceback.format_exc())
                pass
    
    layers = sorted(list(tctp_by_height.keys()), reverse=True)
    
    for i, layer in enumerate(layers):
        if i == 0:
            for center_coordinate, triangle, caps in tctp_by_height[layer]:
                for inds in [[0, 1], [1, 2], [0, 2]]:
                    plt.plot([triangle[inds[0]][0], triangle[inds[1]][0]], [triangle[inds[0]][1], triangle[inds[1]][1]], c='b')
                
                for cap in caps:
                    cap = cap[2][non_layer_axes]
                    plt.plot([center_coordinate[0], cap[0]], [center_coordinate[1], cap[1]], ls='--', c='r')
                    
        elif i == 1:
            for center_coordinate, triangle, caps in tctp_by_height[layer]:
                tx, ty = [], []
                for inds in [[0, 1], [1, 2], [0, 2]]:
                    tx.extend([triangle[inds[0]][0], triangle[inds[1]][0]])
                    ty.extend([triangle[inds[0]][1], triangle[inds[1]][1]])
                    
                plt.fill(tx, ty, color='skyblue', alpha=0.5)
                
                for cap in caps:
                    cap = cap[2][non_layer_axes]
                    plt.plot([center_coordinate[0], cap[0]], [center_coordinate[1], cap[1]], ls='--', c='r')

    legends = [] 
    p0, p1 = unitcell_lengths     
    print(f"Plotting {len(unitcell_points)} atoms for {cif_path.split(os.sep)[-1]}")
    for p, s in unitcell_points:
        sy = site_symbol_map[s]
        p = p[non_layer_axes]
        
        if sy not in legends:
            legends.append(sy)
            plt.scatter(p[0], p[1], c=color_labels[sy], label=sy)
            plt.text(p[0]-(p0*0.0025), p[1]-(p1*0.035), s, size=6)
        else:
            plt.scatter(p[0], p[1], c=color_labels[sy])
            plt.text(p[0]-(p0*0.025), p[1]-(p1*0.035), s, size=6)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(f"images/{title.replace(',', '-')} {cif_path.split(os.sep)[-1][:-4]}.png", dpi=300)
    
    
    
def get_first_9_neighbors(point, supercell_points):
    points_wd = []
    
    sites = [p[-1] for p in supercell_points]
    supercell_points = np.vstack([p[0] for p in supercell_points])
    
    d = np.linalg.norm(supercell_points - point[0], axis=1)
    ind9 = np.argsort(d)[1:10]
    
    for i in ind9:
        points_wd.append((sites[i], round(d[i], 3), [round(v, 3) for v in point[0].tolist()], [round(v, 3) for v in supercell_points[i].tolist()]))

    return points_wd


# def pont_in_hull(poly, point):
#     hull = ConvexHull(poly)
#     new_hull = ConvexHull(np.concatenate((poly, [point])))
#     return np.array_equal(new_hull.vertices, hull.vertices)
    
    
# def inside_cell(points, site, cell_lengths, cell_angles, non_layer_axes):
    
#     up = np.array([[0.0, 0.0, 0.0],
#                    [0.0, 0.0, 1.1], 
#                    [0.0, 1.1, 0.0],
#                    [0.0, 1.1, 1.1],
#                    [1.1, 0.0, 0.0],
#                    [1.1, 0.0, 1.1],
#                    [1.1, 1.1, 0.0],
#                    [1.1, 1.1, 1.1]])
    
#     cell = np.array([np.round(unit.fractional_to_cartesian(p, cell_lengths, cell_angles), 3).tolist() for p in up])

#     inside, xy = [], []
#     for i in range(len(points)):
#         point = points[i]

#         if not pont_in_hull(cell, point[0]):
#             continue
        
#         if site is not None:
#             if point[-1] == site:
#                 pass
#             point, _site = point
#             point = point[non_layer_axes]
            
#             if _site == site:
#                 xy.append(point)
#                 inside.append(points[i])
#         else:
#             inside.append(points[i])
                
#     return inside
    

if __name__ == "__main__":
    from tqdm import tqdm
    
    root = "/home/bala/Documents/data/not_prototype_CIFs"
    plot_cell(f"{root}/1707948.cif", "00")
    exit()
    # 1232634, 1535998, 1150374
    
    df = pd.read_csv('tctp.csv')
    df = df[(df['layered']=='yes') & (df['tctp_present']=='yes')]
    # elements = []
    # for _, row in df.iterrows():
    #     elements.extend(list(_parse_formula(row['Formula']).keys()))
        
    # elements = list(set(elements))
    # print(len(elements))
    # print(elements)
    df = df.drop_duplicates("Structure type")
    df['sg'] = df["Structure type"].map(lambda x: int(x.split(',')[-1]))
    # df = df[(df['sg'] >= 75) & (df['sg'] <= 142)]
    
    c = 0
    errors = []
    t = len(df)
    for i, (_, r) in enumerate(df.iterrows(), 1):
        print(f"{i} of {t}")
        # if i > 10:
        #     exit(0)
        try:
            plot_cell(f"/home/bala/Documents/data/not_prototype_CIFs/{r['CIF']}", r['Structure type'])
        except:
            errors.append(f"Error processing {r['CIF']} {r['Structure type']}\n")
            c += 1
    print(c)
    
    with open('err.txt', 'w') as f:
        for e in errors:
            f.write(e)
