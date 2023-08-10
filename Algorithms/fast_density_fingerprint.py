########################################################################################################
## Version: March 2023 (202303)
## The code was authored by Min-Chun Wu and Chuan-Shen Hu. 
## Min-Chun Wu primarily developed the kernel functions, while Chuan-Shen Hu undertook code refinement, adaptation for practical applications, and bug fixes.
########################################################################################################

import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.linalg import inv
from scipy.spatial import distance_matrix

########################################################################################################
## Distance functions
########################################################################################################

## Given a line segment with endpoints p and u in R3 (i.e., the set {p+tu : t in [0,1]}), for a point x,
## output the distance from the point x to the line segment
## Examples:
## p = np.array([0,0])
## u = np.array([1,0])
## x = np.array([1/2,1])
## print('dist2line: ', dist2line(p,u,x))
def dist2line(p,u,x):
    t = np.inner(u,x-p)/np.inner(u,u)
    if 0<=t<=1:
        return norm(x-p-t*u)
    else:
        return min([norm(x-p), norm(x-p-u)])

## Consider a 2-dimensional parallelogram spanned by vectors u-p and v-p in R3 with vertex p.
## The set of points of the parallelogram can be represented by {p+sv+tw : (s,t) in [0,1]x[0,1] }
## Output the distance from point x to this parallelogram
## Examples:
## p = np.array([0,0,0])
## u = np.array([1,0,0])
## v = np.array([0,1,0])
## x = np.array([1,0,1])
## print('dist2plane: ', dist2plane(p,u,v,x))
def dist2plane(p,u,v,x):
    A = np.array([[np.inner(u,u), np.inner(u,v)], [np.inner(u,v), np.inner(v,v)]])
    s, t = np.matmul(inv(A), np.array([np.inner(u,x-p), np.inner(v,x-p)]))
    if 0<=s<=1 and 0<=t<=1:
        return norm(x-p-s*u-t*v)
    else:
        return min([dist2line(p,u,x), dist2line(p,u,x), dist2line(p+u,v,x), dist2line(p+v,u,x)])

## Given a basis in R2 (or R3) and and x in R2 (or R3),
## compute the maximal distance from x to the parallelogram spanned by the basis
## Recall the maximum must occurs at the boundary of the parallelogram
## Examples: (For diemsion 2)
## basis = [np.array([1,0]), np.array([0,1])]
## x = np.array([1,1])
## print('dist2U_max: ', dist2U_max(basis, x))
## print('dist2U_min: ', dist2U_min(basis, x))
def dist2U_max(basis,x):
    l = len(basis)
    if l==2:
        coeff = [(r1,r2) for r1 in [0,1] for r2 in [0,1]]
        vertices = [r1*basis[0]+r2*basis[1] for (r1,r2) in coeff]
    elif l==3:
        coeff = [(r1,r2,r3) for r1 in [0,1] for r2 in [0,1] for r3 in [0,1]]
        vertices = [r1*basis[0]+r2*basis[1]+r3*basis[2] for (r1,r2,r3) in coeff]
    return max([norm(v-x) for v in vertices])

## Given a basis in R2 (or R3) and and x in R2 (or R3),
## compute the minimal distance from x to the parallelogram spanned by the basis
## Recall the minimum must occurs at the boundary of the parallelogram
## Examples: (For diemsion 3)
## basis = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
## x = np.array([1,1,1])
## print('dist2U_max: ', dist2U_max(basis, x))
## print('dist2U_min: ', dist2U_min(basis, x))
def dist2U_min(basis,x):
    l = len(basis)
    if l==2:
        origin = np.array([0,0])
        v1, v2 = basis
        return min([dist2line(origin,v1,x), dist2line(origin,v2,x),
                    dist2line(v1,v2,x), dist2line(v2,v1,x)])
    elif l==3:
        origin = np.array([0,0,0])
        v1, v2, v3 = basis
        return min([dist2plane(origin,v1,v2,x), dist2plane(origin,v2,v3,x),
                    dist2plane(origin,v1,v3,x), dist2plane(v1,v2,v3,x),
                    dist2plane(v2,v1,v3,x), dist2plane(v3,v1,v2,x)])

########################################################################################################
## Densitiy Fingerprints
########################################################################################################

## An example about how to compute the density fingerprints:
## from fast_density_fingerprint import find_relevant_pts, density_fingerprint, plot_density
## import numpy as np
## basis = [np.array([1,0]), np.array([0,1])]
## Motif = [np.array([0,0])]
## k_up = 8 ## compute \psi_k, k = 0, 1, 2, ..., k_up
## pts_k_up, dists_min_k_up = find_relevant_pts(basis, Motif, k_up)
## eps = 0.0001 ## tolerable density value error
## range_t = (0,1.8) ## range of the x-axis of the curve
## fine_t = 100 ## fineness of the x-axis of the curve
## psi_dict = density_fingerprint(pts_k_up, k_up, basis, range_t, fine_t, eps)
## dpi = 300
## plot_density(psi_dict, range_t, dpi)

## Given a basis in R3 (or R2), Motif, and k_up, compute relevant points to be used in the algorithm
def find_relevant_pts(basis, Motif, k_up):
    ## Transpose the matrix of row vectors to column vectors
    A_basis = np.transpose(np.array(basis)) # i.e. [v1 v2 (v3)]
    d = len(basis[0])
    ## For a 2-dimentional basis for R2
    if d==2:
        directions = [np.array([1, 0]), np.array([-1, 0]),
                      np.array([0, 1]), np.array([0, -1])] ## directions for generating neighbors
    ## For a 3-dimentional basis for R3
    elif d==3:
        directions = [np.array([1, 0, 0]), np.array([-1, 0, 0]),
                      np.array([0, 1, 0]), np.array([0, -1, 0]),
                      np.array([0, 0, 1]), np.array([0, 0, -1])] ## directions for generating neighbors
    else:
        print("The first parameter should be an np-array of a basis in R3 (or R2)")
        return

    visited = [] ## to save visited base-point of qualified unit cells
    queue = [np.array([0 for i in range(d)])] ## centers to be tested
    dists_k_up = [] ## to save the 'distances' between U and the 'nearest' k_up points
    dists_set = set()

    while queue!=[]:
        base = queue.pop() ## base-point of the considered unit cell
        visited.append(base)
        qualified = False
        ind_remain = -1
        if len(dists_set)<k_up+1:
            qualified = True
            for i in range(len(Motif)):
                x = Motif[i]+np.matmul(A_basis, base)
                dist_x2U = dist2U_max(basis, x)
                dists_k_up.append(dist_x2U)
                dists_set.add(dist_x2U)
                if len(dists_set)==k_up:
                    ind_remain = i+1
                    break
            if ind_remain!=-1:
                for i in range(ind_remain, len(Motif)):
                    x = Motif[i]+np.matmul(A_basis, base)
                    dist_x2U = dist2U_max(basis, x)
                    if dist_x2U in dists_set:
                        dists_k_up.append(dist_x2U)
                        dists_set.add(dist_x2U)
                    else:
                        dist_max = max(dists_set)
                        if dist_max>dist_x2U:
                            ind_non_max = (np.array(dists_k_up)!=dist_max)
                            dists_k_up = list(np.array(dists_k_up)[ind_non_max])
                            dists_set.remove(dist_max)
                            dists_k_up.append(dist_x2U)
                            dists_set.add(dist_x2U)
        else:
            for i in range(len(Motif)):
                x = Motif[i]+np.matmul(A_basis, base)
                dist_x2U = dist2U_max(basis, x)
                if dist_x2U in dists_set:
                    qualified = True
                    dists_k_up.append(dist_x2U)
                    dists_set.add(dist_x2U)
                else:
                    dist_max = max(dists_set)
                    if dist_max>dist_x2U:
                        qualified = True
                        ind_non_max = (np.array(dists_k_up)!=dist_max)
                        dists_k_up = list(np.array(dists_k_up)[ind_non_max])
                        dists_set.remove(dist_max)
                        dists_k_up.append(dist_x2U)
                        dists_set.add(dist_x2U)
        if qualified:
            nbs = [base + direct for direct in directions]
            for nb in nbs:
                nb_in_visited = np.array([(nb==a).all() for a in visited]).any()
                nb_in_queue = np.array([(nb==a).all() for a in queue]).any()
                if not (nb_in_visited or nb_in_queue):
                    queue.append(nb)

    dists_k_up = np.array(dists_k_up)

    refined_dists = np.array([]).reshape(0, )
    while len(refined_dists)<k_up+1:
        ind_min = (dists_k_up==min(dists_k_up))
        refined_dists = np.concatenate((refined_dists, dists_k_up[ind_min]), axis=0)
        dists_k_up = dists_k_up[~ind_min]

    ############################################################################
    ## apply DFS again to find necessary points
    ############################################################################
    dist_up = refined_dists[-1]

    visited = [] ## to save visited base-point of qualified unit cells
    queue = [np.array([0 for i in range(d)])]
    pts_k_up = [] ## to save the points necessary to construct \psi_0, ..., \psi_{k_up}
    dists_min_k_up = [] ## to save the 'distances' between U and the 'nearest' k_up points

    while queue!=[]:
        base = queue.pop() ## base-point of the considered unit cell
        visited.append(base)
        qualified = False
        if pts_k_up==[]:
            qualified = True
            for m in Motif:
                pts_k_up.append(m)
                dists_min_k_up.append(0)
        else:
            for i in range(len(Motif)):
                x = Motif[i]+np.matmul(A_basis, base)
                distx2U_min = dist2U_min(basis, x)
                if distx2U_min<=dist_up:
                    qualified = True
                    pts_k_up.append(x)
                    dists_min_k_up.append(distx2U_min)
        ## if qualified, append nbs to 'queue'
        if qualified:
            nbs = [base + direct for direct in directions]
            for nb in nbs:
                nb_in_visited = np.array([(nb==a).all() for a in visited]).any()
                nb_in_queue = np.array([(nb==a).all() for a in queue]).any()
                if not (nb_in_visited or nb_in_queue):
                    queue.append(nb)

    return (pts_k_up, dists_min_k_up)

## For example:
## Using the function find_relevant_pts() to find pts_k_up, dists_min_k_up
## eps = 0.0001      ---> tolerable density value error
## range_t = (0,1.8) ---> range of the x-axis of the curve
## fine_t = 1000     ---> fineness of the x-axis of the curve
def density_fingerprint(pts_k_up, k_up, basis, range_t, fine_t, eps):

    A_basis = np.transpose(np.array(basis)) # i.e. [v1 v2 (v3)]
    d = len(pts_k_up[0])

    k_g = math.ceil((1/eps)**(1/d)) ## the number of grid points in each dimensional direction
    if d==2:
        grid_pts = [np.matmul(A_basis, np.array([l1, l2])/k_g)
                    for l1 in range(1,k_g+1) for l2 in range(1,k_g+1)]
    elif d==3:
        grid_pts = [np.matmul(A_basis, np.array([l1, l2, l3])/k_g)
                    for l1 in range(1,k_g+1) for l2 in range(1,k_g+1) for l3 in range(1,k_g+1)]

    ## the matrix of dist(grid_point, periodic_Motif_point)
    #dists = np.array([[norm(pt_g-pt_k_up) for pt_g in grid_pts] for pt_k_up in pts_k_up])
    dists = distance_matrix(pts_k_up, grid_pts)

    ## compute the density fingerprints
    psi_dict = {k:[] for k in range(k_up+1)}

    for t in np.linspace(range_t[0], range_t[1], fine_t):
        overlap_count = np.sum(dists<t, axis=0)
        (unique, counts) = np.unique(overlap_count, return_counts=True)
        other_k = [k for k in range(k_up+1) if k not in unique]
        for i in range(len(unique)):
            if unique[i] in range(k_up+1):
                psi_dict[unique[i]].append(counts[i]/len(grid_pts))
        for k in other_k:
            psi_dict[k].append(0)

    return psi_dict

## plot the density fingerprints
def plot_density(psi_dict, range_t, k_start, k_end, dpi, title='Density fingerprints'):
    plt.figure(dpi=dpi)
    fine_t = len(psi_dict[0])
    for k in range(k_start, k_end+1):
        plt.plot(np.linspace(range_t[0], range_t[1], fine_t),
                 psi_dict[k],
                 label = r'$\psi_{'+r'{}'.format(k)+r'}$',
                 linewidth = 0.5
                )
        plt.title(title)
        plt.legend()
