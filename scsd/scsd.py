# scsd - Symmetry-Coordinate Structural Decomposition for molecules
# written by Dr. Christopher J. Kingsbury, Trinity College Dublin, with Prof. Dr. Mathias O. Senge
# cjkingsbury@gmail.com / www.kingsbury.id.au
#
# This work is licensed under THE ANTI-CAPITALIST SOFTWARE LICENSE (v 1.4) 
# To view a copy of this license, visit https://directory.fsf.org/wiki/License:ANTI-1.4

from os.path import dirname, abspath
from pathlib import Path

data_path = Path(__file__).parent / "data/scsd/"

import numpy as np

# general imports
from numpy import (
    cos,
    mean,
    min,
    tril,
    pi,
    sort,
    sign,
    sin,
    subtract,
    sqrt,
    shape,
    var,
    where,
    sum as npsum,
    zeros,
)
from numpy.linalg import norm, solve, lstsq, det, LinAlgError
from scipy.optimize import linear_sum_assignment, minimize, basinhopping
from scipy.stats import gaussian_kde

# for Mondrian
from seaborn import color_palette
from matplotlib import cm, colors, use
from matplotlib.pyplot import subplots

use("Agg")

try:
    from sklearn.decomposition import PCA  # for principal_components
except ImportError:
    print("cannot import PCA")
from io import BytesIO  # for non-saving on the webserver (i.e. direct sending)
from plotly.io import to_html

# useful little tool. Webserver could be adapted to fetch plotly source code,

# but it's nice that it's self-contained in one html
from datetime import date  # for timestamps

from flask import Flask, render_template, request, url_for, send_from_directory
from pandas import DataFrame, read_pickle
import plotly.graph_objects as go
import base64
import random
from plotly.express import histogram
import plotly.colors
import networkx as nx

from .scsd_symmetry import (
    point_group_dict,
    operations_dict,
    irrep_typog_html,
    scsd_symmetry,
    symm_typog_html,
    ordered_ops_dict,
    pgt_dict,
    labels_dict_ext,
    labels_dict,
    symm_multiplicity_tables,
    mondrian_transform_dict,
    mondrian_lookup_dict,
    mondrian_orientation_dict,
    irrep_typog,
    symm_typog_lookup,
)

# this will go eventually as these dictionaries get transferred to a JSON representation
from copy import deepcopy


# These are random cmaps (colourmaps) for use with the Mondrian diagram. When calling from scsd datasets these are used.
# Hatch is not implemented with the webserver at the moment.
# good_cmaps_with_sns = 'Spectral,RdYlBu,RdYlBu_r,rainbow,Blues,Greys,sns_RdBu_r,sns_colorblind,sns_Set2,sns_Set3'.split(',')
# + ['sns_ch:-1.5,-1,light=0.98,dark=.3','sns_ch:0.3,-0.5,light=0.98,dark=.3']
good_cmaps = "Spectral,RdYlBu,RdYlBu_r,rainbow,Blues,Greys,RdBu_r,Set2,Set3,colorblind,Paired,gist_heat_r,hls".split(
    ","
)
hatch_picks = [
    ".",
    "..",
    "/",
    "//",
    "\\",
    "\\\\",
    "*",
    "**",
    "-",
    "--",
    "+",
    "++",
    "|",
    "||",
    "x",
    "xx",
    "o",
    "oo",
    "O",
    None,
]
safe_cmaps = [
    "Spectral",
    "viridis",
    "magma",
    "crest",
    "rocket",
    "flare",
    "mako",
    "cubehelix",
]
safe_cmaps = safe_cmaps + [x + "_r" for x in safe_cmaps]
good_cmaps = good_cmaps + safe_cmaps

atoms_color_dict = {
    "C": "black",
    "H": "white",
    "N": "cornflowerblue",
    "S": "yellow",
    "O": "red",
    "Cs": "grey",
    "Li": "violet",
    "Cl": "green",
    "F": "lime",
    "Fe": "firebrick",
    "Mn": "magenta",
    "P": "orange",
    "Br": "maroon",
    "I": "darkviolet",
    "Na": "purple",
    "K": "violet",
    "B": "pink",
    "Cu": "lightblue",
}


# Numerical transforms - these are generally faster than the equivalent implementation through scipy.spatial
def s2c(th1, ph1):
    return np.array((sin(ph1) * cos(th1), sin(ph1) * sin(th1), cos(ph1)))


# spherical to cartesian
def c2q(om, vec):
    return np.hstack([np.array([cos(om / 2)]), (vec * sin(om / 2))])


# cartesian to quaternion
def s2q(th, ph, om):
    return np.array(
        [
            sin(om / 2) * sin(ph) * cos(th),
            sin(om / 2) * sin(ph) * sin(th),
            sin(om / 2) * cos(ph),
            cos(om / 2),
        ]
    )


# spherical to quaternion
def uq2r(x, y, z, w):
    return (
        (1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)),
    )


# unit quaternion (in x,y,z,w) to rotation matrix
def q2uq(x, y, z, w):
    return [x, y, z, w] / norm([x, y, z, w])


# quaterion to unit quaternion
def q2r(x, y, z, w):
    return uq2r(*q2uq(x, y, z, w))


# combination of above functions


# cost matrix generation for the fitting of atoms to other atom groups - used for making assignment matrices,
# making models and for fitting atoms to models
def costmat_gen(m1, m2):
    return np.sum(np.square(np.diagonal(np.subtract.outer(m2, m1), 0, 1, 3)), axis=-1)


def costmat_gen_t(m1, m2):
    return np.sum(np.square(np.diagonal(np.subtract.outer(m2, m1), 0, 0, 2)), axis=-1)


# _t is for the transform  of the matrix - 100x faster than the old version
# def costmat_gen(m1,m2): return np.sum(np.square(np.subtract(np.expand_dims(m2,0),np.expand_dims(m1,1))),axis = -1)
# def costmat_gen_t(m1,m2): return np.sum(np.square(np.subtract(np.expand_dims(m2,2),np.expand_dims(m1,1))),axis = 0)


def fit_atoms_rt(p, ats):
    return np.dot(q2r(*p[3:7]), (ats - p[0:3]).T).T


def fit_atoms_rtt(p, ats):
    return np.dot(q2r(*p[3:7]), subtract(ats, np.expand_dims(p[:3], axis=-1)))


# basically the fit functions for a lot of the atoms - removes the translational and rotational
# elements fo the matrix, which are uninteresting for the sake of this algorithm
def fit_atoms_rot(p, ats):
    return np.dot(q2r(*p), ats.T).T


def fit_atoms_rott(p, ats):
    return np.dot(q2r(*p), ats)


# just the rotation


def mat_dist_t(m1, m2):
    row_ind, col_ind = linear_sum_assignment(np.sqrt(costmat_gen_t(m1, m2)))
    return np.sum(np.square(m1[:, col_ind] - m2[:, row_ind]))


# linear sum assignment of two vector matrices as (3,x) shapes
# returned as the sum of squares


def mat_dist(m1, m2):
    row_ind, col_ind = linear_sum_assignment(np.sqrt(np.sqrt(costmat_gen(m1, m2))))
    return np.sum(np.square(m1[col_ind] - m2[row_ind]))


# linear sum assignment of two vector matrices as (x,3) shapes
# returned as the sum of squares


def import_pdb(filenm, query_atoms=False):
    # reads a pdb file into an x,y,z,atom type (n,4) numpy array
    # this is the starting point for a lot of the computation, and pdb are already in real space, so it makes
    # things a lot easier. Can add other file types in future.
    flist = open(filenm, "r").readlines()
    if query_atoms:
        atoms = np.array(
            [
                (li[31:39], li[39:47], li[47:55], li.split()[-1].rstrip(" 0123456789"))
                for li in flist
                if (li.startswith("ATOM") or li.startswith("HETATM"))
                and li.split()[-1].rstrip(" 0123456789") in query_atoms
            ]
        )
    else:
        atoms = np.array(
            [
                (li[31:39], li[39:47], li[47:55], li.split()[-1].rstrip(" 0123456789"))
                for li in flist
                if (li.startswith("ATOM") or li.startswith("HETATM"))
            ]
        )
    atoms_out = np.hstack(
        [
            atoms[:, :3].astype(float) - mean(atoms[:, :3].astype(float), axis=0),
            np.atleast_2d(atoms[:, 3]).T,
        ]
    )
    return np.array(atoms_out)


def import_pdb_waniso(filenm, query_atoms=False):
    flist = np.array(open(filenm, "r").readlines())
    if query_atoms:
        serials = [
            ix
            for ix, li in enumerate(flist)
            if (li.startswith("ATOM") or li.startswith("HETATM"))
            and li.split()[-1] in query_atoms
        ]
    else:
        serials = [
            ix
            for ix, li in enumerate(flist)
            if (li.startswith("ATOM") or li.startswith("HETATM"))
        ]
    atoms = np.array(
        [(li[31:39], li[39:47], li[47:55], li.split()[-1]) for li in flist[serials]]
    )
    serials_plus_one = [x + 1 for x in serials]
    anisos = np.array(
        [
            li.split()[5:-1] if li.startswith("ANISOU") else [0, 0, 0, 0, 0, 0]
            for li in flist[serials_plus_one]
        ]
    )
    atoms_out = np.hstack(
        [
            atoms[:, :3].astype(float) - mean(atoms[:, :3].astype(float), axis=0),
            np.atleast_2d(atoms[:, 3]).T,
            anisos.astype(float),
        ]
    )
    return atoms_out


def remove_symm(atom_matrix, typo, theta, phi, rotation):
    # takes [xyz] and [xyza]
    # basically, reads a symmetry operation and makes the molecular matrix which adheres to that symmetry
    # operation. This is done by transforming the molecule and subtracting the assigned function net.
    # Not foolproof for very distorted molecules, or when models are not used.
    # in v3, this is used solely for determining models - or temporary models
    omega = (2 * pi) / rotation
    refvec = s2c(theta, phi)
    rot_mat = q2r(*s2q(theta, phi, omega))

    if len(atom_matrix.T) == 4:
        atom_types = np.unique(atom_matrix[:, 3].tolist())
        mean_arr = []

        for elem in atom_types:
            atoms = np.vstack([x[0:3] for x in atom_matrix if x[3] == elem]).astype(float)
            trans_atoms = atoms.copy()

            if typo in ["improperrotation", "mirror"]:
                trans_atoms = trans_atoms - np.outer(
                    2 * np.dot(trans_atoms, refvec), refvec
                )
            if typo in ["rotation", "improperrotation"]:
                trans_atoms = np.dot(rot_mat, trans_atoms.T).T
            if typo == "inversion":
                trans_atoms = -atoms

            costmat = costmat_gen(atoms, trans_atoms)
            row_ind, col_ind = linear_sum_assignment(costmat)
            [
                mean_arr.append(np.hstack([np.add(x, y) / 2, elem]))
                for x, y in zip(atoms[col_ind], trans_atoms[row_ind])
            ]
        return np.array(mean_arr)
    else:
        trans_atoms = atom_matrix.copy()

        if typo in ["improperrotation", "mirror"]:
            trans_atoms = trans_atoms - np.outer(2 * np.dot(trans_atoms, refvec), refvec)
        if typo in ["rotation", "improperrotation"]:
            trans_atoms = np.dot(rot_mat, trans_atoms.T).T
        if typo == "inversion":
            trans_atoms = -atom_matrix

        costmat = costmat_gen(atom_matrix, trans_atoms)
        row_ind, col_ind = linear_sum_assignment(costmat)

        a1, a2 = atom_matrix[col_ind], trans_atoms[row_ind]
        return np.add(a1, a2) / 2


def inv_sq_t(atoms_t):
    # determines square S-value of applying a centre-of-inversion -
    # should minimise to the centre of mass in correctly assigned cases.
    costmat = costmat_gen_t(atoms_t, atoms_t * -1)
    return sum(costmat[linear_sum_assignment(costmat)])


def rot_sq_t(atoms_t, theta, phi, rotation, properness=1):
    # determines square S-value of applying a rotation operation defined by theta and phi (spherical coords)
    # should minimise to the assigned axes of the point group
    # only used in the no-model case
    quat = s2q(theta, phi, (2 * pi) / rotation)
    rotmat = np.dot(q2r(*quat), atoms_t)
    if not properness:
        refvec = s2c(theta, phi)
        rotmat = rotmat - np.outer(
            refvec, 2 * np.dot(refvec, rotmat)
        )  # rotmat-[[2*y*np.dot(x, refvec) for y in refvec] for x in rotmat]
    costmat = costmat_gen_t(atoms_t, rotmat)
    return sum(costmat[linear_sum_assignment(costmat)])


def mir_sq_t(atoms_t, theta, phi, rotation=None, properness=None):
    # as above, with a mirror plane defined as the normal of the vector at theta,phi in spherical coords.
    refvec = s2c(theta, phi)
    refmat = atoms_t - np.outer(refvec, 2 * np.dot(refvec, atoms_t))
    # atom_matrix-[[2*y*np.dot(x, refvec) for y in refvec] for x in atom_matrix]
    # old version
    costmat = costmat_gen_t(atoms_t, refmat)
    return sum(costmat[linear_sum_assignment(costmat)])


def check_vals_t(ats, vals):
    # calls the above functions by the name given in symmetry_dicts
    if vals[0][0] in ["rotation", "improperrotation"]:
        return sum(
            [rot_sq_t(ats, x[1], x[2], x[3], vals[0][0] == "rotation") for x in vals]
        )
    if vals[0][0] == "inversion":
        return sum([inv_sq_t(ats)])
    if vals[0][0] == "mirror":
        return sum([mir_sq_t(ats, x[1], x[2]) for x in vals])


# This all below (_g) supports graph-matching for model yielding - for by_graph = True in that subroutine
# Unfortunately, it's actually slower than linear_sum_assignment over the entire cost matrix, for some
# unknown reason. I'll leave it in, something to come back to

# This stuff was slower
# from scipy.sparse import coo_matrix as coom
# from scipy.sparse.csgraph import min_weight_full_bipartite_matching as mwfbm
# from scipy.sparse import block_diag


def sparse_costmat_gen_t(m1, m2, atl):
    return (
        np.sum(np.square(np.diagonal(subtract.outer(m2[:, x], m1[:, x]), 0, 0, 2)), axis=-1)
        for x in atl
    )


def op_sq_g(atoms_t, op, theta, phi, rotation, atom_type_lists):
    if op == "rotation":
        quat = s2q(theta, phi, (2 * pi) / rotation)
        mat2 = np.dot(q2r(*quat), atoms_t)
    elif op == "improperrotation":
        quat = s2q(theta, phi, (2 * pi) / rotation)
        refvec = s2c(theta, phi)
        rotmat = np.dot(q2r(*quat), atoms_t)
        mat2 = rotmat - np.outer(refvec, 2 * np.dot(refvec, rotmat))
    elif op == "inversion":
        mat2 = atoms_t * -1
    elif op == "mirror":
        refvec = s2c(theta, phi)
        mat2 = atoms_t - np.outer(refvec, 2 * np.dot(refvec, atoms_t))

    return sparse_costmat_gen_t(atoms_t, mat2, atom_type_lists)
    # return sum(costmat.toarray()[mwfbm(costmat)])


# def check_vals_g(ats, vals, atom_type_lists):
#    #calls the above functions by the name given in symmetry_dicts
#    return(sum(op_sq_g(ats, *x, atom_type_lists) for x in vals))


def check_vals_g(ats, vals, atom_type_lists):
    # calls the above functions by the name given in symmetry_dicts
    return sum(
        [
            sum(x[linear_sum_assignment(x)])
            for y in [op_sq_g(ats, *z, atom_type_lists) for z in vals]
            for x in y
        ]
    )
    # print(costmat)
    # return(sum(op_sq_g(ats, *x, atom_type_lists) for x in vals))


def yield_model(
    atom_matrix, point_group_name, bhopping=False, by_graph=False, max_dist=1.75
):
    # Essentially orientates and symmetrizes a molecule to the assigned point group
    # this model can then be used to assign multiple structures to the same coordinate
    # frame (to prevent confusion of e.g. B1,B2 in C2v) or to generate reliable normal
    # coordinate vectors through principal component analysis (via scsd_principal_components),
    # rendered meaningless without a comparative model.
    # Additionally, the model fitting is significantly faster, allowing for ~1s data reduction.
    # Using the same model will ensure totally symmetric e.g. A1 modes are comparable - these are
    # simply a comparison of two structures so the model source should be reported.

    # adding in a "by_graph" version, which should ideally 'colour' the atoms and only allow identical
    # colourings to pair, which I think will speed everything up.
    # Except it doesn't - slows down by a factor of 2. Oh well, worth trying.

    symm = scsd_symmetry(point_group_name)
    # ops = point_group_dict.get(point_group_name)
    # ops_vals = [operations_dict.get(x) for x in ops]

    quats = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]

    if by_graph:
        atom_matrix = atom_matrix[:, :3] - mean(atom_matrix[:, :3], axis=0)
        atoms_t = atom_matrix.T

        G = graph_from_ats(atom_matrix, max_dist)
        GM = nx.algorithms.isomorphism.GraphMatcher(G, G)
        bl = np.array(
            [[y for y, z in x.items()] for x in GM.subgraph_isomorphisms_iter()]
        ).T
        atom_type_lists = [np.unique(x) for x in np.unique(sort(bl, axis=1), axis=0)]
        # print(atom_type_lists)
        vals = [x for y in symm.ops_verb for x in y]

        # print(vals)
        def fitfunc(p):
            return check_vals_g(np.dot(q2r(*p), atoms_t), vals, atom_type_lists)

    elif len(atom_matrix[0]) == 3:
        atom_matrix = atom_matrix - mean(atom_matrix, axis=0)
        atoms_t = atom_matrix.T

        def fitfunc(p):
            return sum(
                [check_vals_t(np.dot(q2r(*p), atoms_t), x) for x in symm.ops_verb]
            )

    elif len(atom_matrix[0]) == 4:
        atom_types = np.unique(atom_matrix[:, 3].tolist())
        atom_type_lists = [
            [i for i, x in enumerate(atom_matrix[:, 3].tolist()) if x == elem]
            for elem in atom_types
        ]
        atoms_t = (
            atom_matrix[:, 0:3].astype(float).T
        )  # - mean(atom_matrix[:,0:3],axis = 0)

        # fitfunc = lambda p: sum([sum([check_vals_t(np.dot(q2r(*p),atoms_t), x) for x in ops_vals]) for ind in atom_type_lists])
        def fitfunc(p):
            return sum(
                [
                    sum(
                        [
                            check_vals_t(np.dot(q2r(*p), atoms_t[:, ind]), x)
                            for x in symm.ops_verb
                        ]
                    )
                    for ind in atom_type_lists
                ]
            )

    if bhopping:
        fits = [basinhopping(fitfunc, q, niter=10) for q in quats]
    else:
        fits = [minimize(fitfunc, q) for q in quats]

    fit = fits[np.argmin([x.fun for x in fits])]

    # Fits the atoms to the symmetry operations
    if len(atom_matrix[0]) == 3:
        # total_symm_output = R.from_quat(fit.x).apply(atom_matrix)
        total_symm_output = fit_atoms_rt(np.hstack([[0, 0, 0], fit.x]), atom_matrix)
    if len(atom_matrix[0]) == 4:
        # total_symm_output = np.array(hstack((R.from_quat(fit.x).apply(atom_matrix[:,0:3].astype(float)),matrix(atom_matrix[:,3]).T)))
        total_symm_output = np.array(
            np.hstack(
                (
                    fit_atoms_rt(
                        np.hstack([[0.0, 0.0, 0.0], fit.x]),
                        atom_matrix[:, 0:3].astype(float),
                    ),
                    np.atleast_2d(np.array(atom_matrix[:, 3])).T,
                )
            )
        )

    # "remove_symm" is a bit of a misnomer - we're adding symmetry - i.e. removing the distortion from a symmetry operation
    for x in symm.ops:
        for typo, t, p, r in operations_dict.get(x):
            total_symm_output = remove_symm(total_symm_output, typo, t, p, r)

    # This part aligns ambiguous symmetric groups to the minimal variance i.e. planar D2h to the x,y plane, planar C2v to the x,z
    # this means that the scsd values will still be comparable when dealing with non-models
    if len(atom_matrix[0]) == 3:
        if point_group_name == "D2h":
            total_symm_output = total_symm_output[
                :, np.argsort(var(total_symm_output[:, :3].astype(float), axis=0))[::-1]
            ]
        if point_group_name == "D4h":
            if (
                min([sum(x - (1.4, 1.4, 0)) for x in total_symm_output]) < 0.2
            ):  # checks if molecule is an off-axis porphyrin-type chelate
                total_symm_output = np.dot(
                    [
                        [np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
                        [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
                        [0, 0, 1],
                    ],
                    total_symm_output.T,
                )
        if point_group_name == "C2v":
            total_symm_output = total_symm_output[
                :,
                np.hstack(
                    [
                        np.argsort(var(total_symm_output[:, :2].astype(float), axis=0))[
                            ::-1
                        ],
                        [2],
                    ]
                ),
            ]
    if len(atom_matrix[0]) == 4:
        if point_group_name == "D2h":
            total_symm_output = total_symm_output[
                :,
                np.hstack(
                    [
                        np.argsort(var(total_symm_output[:, :3].astype(float), axis=0))[
                            ::-1
                        ],
                        [3],
                    ]
                ),
            ]
        if point_group_name == "D4h":
            if (
                min(
                    [
                        sum(x - (1.4, 1.4, 0))
                        for x in total_symm_output[:, :3].astype(float)
                    ]
                )
                < 0.2
            ):  # checks if molecule is an off-axis porphyrin-type chelate
                t2 = np.dot(
                    total_symm_output[:, :3].astype(float),
                    [
                        [np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
                        [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
                        [0, 0, 1],
                    ],
                )
                total_symm_output = np.hstack([t2, total_symm_output])[:, [0, 1, 2, 6]]
        if point_group_name == "C2v":
            total_symm_output = total_symm_output[
                :,
                np.hstack(
                    [
                        np.argsort(var(total_symm_output[:, :2].astype(float), axis=0))[
                            ::-1
                        ],
                        [2, 3],
                    ]
                ),
            ]
    return total_symm_output


def trim_to_model(m1, m2):
    # takes a set of atoms (m1) and a model (m2) and removes extraneous atoms, while ordering the atoms to the model
    # quite useful when using precalculated assignemnt matrices
    if len(m1[0]) == 3:
        costmat = np.array([[norm(v1 - v2) for v1 in m1] for v2 in m2])
        row_ind, col_ind = linear_sum_assignment(costmat)
        return m1[col_ind]
    elif len(m1[0]) == 4:
        atom_types = np.unique(m2[:, 3].tolist())
        a1 = []
        for elem in atom_types:
            m1_sel = np.vstack([x[0:3] for x in m1 if x[3] == elem]).astype(float)
            m2_sel = np.vstack([x[0:3] for x in m2 if x[3] == elem]).astype(float)
            costmat = np.array([[norm(v1 - v2) for v1 in m1_sel] for v2 in m2_sel])
            row_ind, col_ind = linear_sum_assignment(costmat)
            [a1.append(np.hstack([x, elem])) for x in m1_sel[col_ind]]
        return np.array(a1)


def trim_to_model_waniso(m1, m2, anisos):
    # takes a set of atoms (m1) and a model (m2) and removes extraneous atoms, while ordering the atoms to the model
    # quite useful when using precalculated assignemnt matrices
    if len(m1[0]) == 3:
        costmat = np.array([[norm(v1 - v2) for v1 in m1] for v2 in m2])
        row_ind, col_ind = linear_sum_assignment(costmat)
        return m1[col_ind], anisos[col_ind]
    elif len(m1[0]) == 4:
        atom_types = np.unique(m2[:, 3].tolist())
        a1, a2 = [], []
        for elem in atom_types:
            m1_sel = np.vstack([x[0:3] for x in m1 if x[3] == elem]).astype(float)
            m2_sel = np.vstack([x[0:3] for x in m2 if x[3] == elem]).astype(float)
            an_sel = np.vstack([x for x, y in zip(anisos, m1) if y[3] == elem]).astype(
                float
            )
            costmat = np.array([[norm(v1 - v2) for v1 in m1_sel] for v2 in m2_sel])
            row_ind, col_ind = linear_sum_assignment(costmat)
            [a1.append(np.hstack([x, elem])) for x in m1_sel[col_ind]]
            [a2.append(x) for x in an_sel[col_ind]]
        return np.array(a1), np.array(a2)


def graph_from_ats(ats, d=1.8):
    G = nx.Graph()
    G.add_nodes_from(range(len(ats)))
    dists = tril(norm(np.sum([np.meshgrid(a, -a) for a in ats.T], axis=1), axis=0), -1)
    [G.add_edge(b, a) for a, b in zip(*np.where((0.5 < dists) & (dists < d)))]
    return G


def assign_from_graph(ats1, ats2, d=1.8):
    g = graph_from_ats(ats1, d)
    h = graph_from_ats(ats2, d)
    GM = nx.algorithms.isomorphism.GraphMatcher(g, h)
    res = GM.is_isomorphic()
    if res is False:
        return "not isomorphic"
    else:
        ali = np.array(list(GM.mapping.items()))
        return ali[np.argsort(ali.T[0])].T[1]


def assign_from_subgraph(model_ats, query_ats, d=1.8):
    # Model must be smaller than the query
    g, h = [graph_from_ats(x, d) for x in [model_ats, query_ats]]
    GM = nx.algorithms.isomorphism.GraphMatcher(h, g)
    res = GM.subgraph_is_isomorphic()
    if res is True:
        ali = np.array(list(GM.mapping.items()))
        return ali[np.argsort(ali.T[1])].T[0]


class scsd_matrix:
    # This is the guts of v3 - moving all of the scsd operations to a class which has several methods for
    # generating the various images and tables required
    # simply initialise the class with input ats (a n*3 or n*4 matrix) and a scsd_model object or pointgroup
    # Then run calc_scsd() for the full scsd matrix to be stored as self.scsd_matrix
    # from there, the tables, images and values, as well as those of the model are available.

    # This runs with a model object, but will create a temporary model 'temp' if none is provided

    def __init__(self, input_ats, model=None, ptgr="C2v", fixed_atoms=False):
        self.ats_3 = input_ats[:, :3].astype(float)
        # atoms as a 3*n numpy array
        if len(input_ats[0]) == 3:
            self.ats = input_ats
        elif len(input_ats[0]) == 4:
            self.ats = input_ats
        elif len(input_ats[0]) == 10:
            self.ats = input_ats[:, :4]
            self.anisos = input_ats[:, 4:]

        if type(model) in [scsd_model, type(None)]:
            self.model = model
        elif type(model) is str:
            self.model = model_objs_dict.get(model)
        # gets a precomputed model object
        # see scsd_models

        if type(self.model) is scsd_model:
            self.ptgr = self.model.ptgr.capitalize()
        else:
            self.ptgr = ptgr.capitalize()
        # gets the point group, either from the model or the input.

        self.scsd_matrix = []
        # resets the scsd matrix - this is recalculated by calc_scsd

        self.symm = scsd_symmetry(self.ptgr)
        # makes a scsd_symmetry object - collection of point group tables and other stuff
        # see scsd_symmetry

        self.fixed_atoms = fixed_atoms
        # testing object. For Charles' project, to test whether there's a bypass for the glob minimisation

    def remove_symm_w_assign_mat(
        self, atom_matrix, typo, theta, phi, rotation, assign_mat
    ):
        # remove_symm is a misnomer - this adds symmetry to an atom set defined by the operation
        # of rotation (rotation) around the spherical coordinate vector (theta, phi)
        # or reflection in the normal plane of the vector (theta, phi)
        # or both (improper rotation) or an inversion
        # this version uses an assignment matrix, which is calculated for the model, not for
        # the specific set of atoms to cut down on cross-assignment errors.
        omega = (2 * pi) / rotation
        refvec = s2c(theta, phi)
        # spherical to cartesian
        rot_mat = q2r(*s2q(theta, phi, omega))
        # quaternion to rotation matrix ( spherical coords (incl rotation about) to quaternion)
        # turns this into a dot product operation

        if len(atom_matrix.T) == 3:
            trans_atoms = deepcopy(atom_matrix)
            # deepcopy is required to prevent the same object being overwritten
            # typo is type of operation. Self explanatory - makes transformed atoms to compare
            if typo in ["improperrotation", "mirror"]:
                trans_atoms = trans_atoms - np.outer(
                    2 * np.dot(trans_atoms, refvec), refvec
                )
            if typo in ["rotation", "improperrotation"]:
                trans_atoms = np.dot(rot_mat, trans_atoms.T).T
            if typo == "inversion":
                trans_atoms = -deepcopy(atom_matrix)

            # this then averages the two matrices, which gives the new 'symmetrized' version
            # and preserves order, so it can iterate over
            return np.add(atom_matrix, trans_atoms[assign_mat]) / 2

    def add_output_row(self, row_to_add):
        # just adds an extra row to the scsd_matrix object. Which isn't really a matrix, but
        # i just couldn't go and rename everything. That's why there are sttill references
        # to nsd in here - used to be called 'nsd-any'
        self.scsd_matrix.append(row_to_add)

    def trim_to_model(self, m1, m2):
        # takes a set of atoms (m1) and a model (m2) and removes extraneous atoms, while ordering the atoms to the model
        # quite useful when using precalculated assignemnt matrices
        if len(m1[0]) == 3:
            costmat = np.array([[norm(v1 - v2) for v1 in m1] for v2 in m2])
            row_ind, col_ind = linear_sum_assignment(costmat)
            self.atom_assignment = col_ind
            return m1[col_ind]
        elif len(m1[0]) == 4:
            atom_types = np.unique(m2[:, 3].tolist())
            a1 = []
            for elem in atom_types:
                m1_sel = np.vstack([x[0:3] for x in m1 if x[3] == elem]).astype(float)
                m2_sel = np.vstack([x[0:3] for x in m2 if x[3] == elem]).astype(float)
                costmat = np.array([[norm(v1 - v2) for v1 in m1_sel] for v2 in m2_sel])
                row_ind, col_ind = linear_sum_assignment(costmat)
                [a1.append(np.hstack([x, elem])) for x in m1_sel[col_ind]]
                self.atom_assignment.append(col_ind)
            return np.array(a1)

    def calc_scsd(self, bhopping=True, bypass=False, random_init=False, by_graph=False):
        self.scsd_matrix = []
        # gets model with the required assignment matrix.
        if type(self.model) is type(None):
            # change this to make a temp model - and run the rest - done 24/9
            self.model = scsd_model(
                "temp", yield_model(self.ats, self.ptgr, bhopping), self.ptgr
            )
            # self.scsd_matrix = yield_scsd_matrix(self.ats, self.ptgr, bhopping = bhopping)
            # return self

        if type(self.model.assign_mat) is type(None):
            self.model.gen_assign_mat()

        # this will store the 'symmetrized' matrices
        self.irreps = []

        # our starting points for the minimization
        if random_init:
            init_params = [[0, 0, 0] + [random.random() for i in range(4)]]
        else:
            init_params = [
                [0, 0, 0] + x
                for x in [
                    [1, 0, 0, 1],
                    [0, 1, 0, 1],
                    [0, 0, 1, 1],
                    [1, 0, 0, -1],
                    [0, 1, 0, -1],
                    [0, 0, 1, -1],
                ]
            ]

        if bypass is True:
            self.atoms_f = self.ats_3

        elif by_graph is True:
            # This is if there's a fixed representation between the input atoms and the model,
            # which can be calculated by many methods - we're using a mixed connectivity-graph-spatial
            # model, but using both isn't strictly necessary. The relationship is fixed for outputs from Mercury,
            # for example.
            # This will be essential for the BCR routine, as well as pdb fitting generally.

            # but ... still haven't written the bit for rectifying. That's gotta be included in here.
            # Charles says that it's called "Registration"

            assignment = assign_from_subgraph(
                self.model.ats_3, self.ats_3, self.model.maxdist
            )

            if isinstance(assignment, str) or (assignment is None):
                # this means the assignment has failed
                return self.calc_scsd(bhopping=bhopping, by_graph=False)

            atoms_t = np.array(self.ats_3[assignment] - mean(self.ats_3[assignment], axis=0)).T
            model_t = self.model.ats_3.T

            def fitfunc(p):
                return np.sum(
                    np.fromiter(
                        norm(subtract(fit_atoms_rtt(p, atoms_t), model_t), axis=0),
                        float,
                    )
                )

            try:
                if bhopping:
                    fits = [basinhopping(fitfunc, q, niter=3) for q in init_params]
                else:
                    fits = [minimize(fitfunc, q) for q in init_params]
            except nx.NetworkXException:
                return self.calc_scsd(bhopping=bhopping, by_graph=False)
            fit = fits[np.argmin([x.fun for x in fits])]
            self.fit = fit
            self.atoms_f = fit_atoms_rt(
                fit.x, self.ats_3[assignment] - mean(self.ats_3[assignment], axis=0)
            )

        elif (len(self.ats[0]) == 3) or (len(self.model.ats[0]) == 3):
            # cuts both atoms and model to length 3
            atoms_t, model_t = self.ats_3.T, self.model.ats_3.T

            # our fitting function. Charles apparently has a better version which doesn't require a
            # leastsq minimization
            def fitfunc(p):
                return mat_dist_t(fit_atoms_rtt(p, atoms_t), model_t)

            if bhopping:
                fits = [basinhopping(fitfunc, q, niter=3) for q in init_params]
            else:
                fits = [minimize(fitfunc, q) for q in init_params]
            fit = fits[np.argmin([x.fun for x in fits])]

            # these are our atoms aligned to the model
            self.fit = fit
            self.atoms_f = self.trim_to_model(
                fit_atoms_rt(fit.x, self.ats_3), self.model.ats_3
            )

        else:
            # version for when both atoms and model have atom identity info. Probably more used.
            # rewrite so that the fitfunc calls mat_dist_t only once if feeling like it
            # i.e. set arbitrarily large values for cross-assignment. could be much slower (oN^3 remember)
            atom_types = np.unique(self.model.ats[:, 3].tolist())
            model_type_lists = [
                [i for i, x in enumerate(self.model.ats[:, 3].tolist()) if x == elem]
                for elem in atom_types
            ]
            atom_type_lists = [
                [i for i, x in enumerate(self.ats[:, 3].tolist()) if x == elem]
                for elem in atom_types
            ]
            atoms_t, model_t = self.ats_3.T, self.model.ats_3.T

            def fitfunc(p):
                return sum(
                    [
                        mat_dist_t(fit_atoms_rtt(p, atoms_t[:, ind]), model_t[:, ind2])
                        for ind, ind2 in zip(atom_type_lists, model_type_lists)
                    ]
                )

            if bhopping:
                fits = [basinhopping(fitfunc, q, niter=3) for q in init_params]
            else:
                fits = [minimize(fitfunc, q) for q in init_params]

            fit = fits[np.argmin([x.fun for x in fits])]
            # this could still get us into trouble - there is the possibility that cross-assignment could
            # be introduced here.
            # self.atoms_f = trim_to_model(fit_atoms_rt(fit.x,self.ats_3), self.model.ats_3)
            self.fit = fit
            self.atoms_f = self.trim_to_model(
                fit_atoms_rt(fit.x, self.ats_3), self.model.ats_3
            )

        for name, row in self.symm.pgt.items():
            # one row of our point group table i.e. an irreducible representation
            atoms_i = deepcopy(self.atoms_f)
            ops_in_row = [x for x, y in zip(self.symm.ops_order, row) if (y == 1)]
            # a list of the operations we'll apply by iterating through (below)
            for op in ops_in_row:
                for (typo, t, p, r), assign_mat in zip(
                    operations_dict.get(op), self.model.assign_mat.get(op)
                ):
                    atoms_i = self.remove_symm_w_assign_mat(
                        atoms_i, typo, t, p, r, assign_mat
                    )
            # this is a fix for those grous with a 3,5,6-fold axis - averaging C5 then C5(2) etc is not the same as gaining
            # five-fold symmetry, but to add an average of multiple points would require a big rewrite. This is the compromise
            # where we get close enough by iterating many times
            if (type(self.symm.e_group_parent_multi) is not type(None)) and (
                self.ptgr != "Oh"
            ):
                # print(self.ptgr)
                for i in range(5):
                    for op in ops_in_row:
                        for (typo, t, p, r), assign_mat in zip(
                            operations_dict.get(op), self.model.assign_mat.get(op)
                        ):
                            atoms_i = self.remove_symm_w_assign_mat(
                                atoms_i, typo, t, p, r, assign_mat
                            )
            # at this point, the irrep is the positions of the atoms in the irreducible representation.
            # below transforms it to a vector with the appropriate co-kernel symmetry
            self.irreps.append((name, atoms_i))

        self.irrep_dict = dict(self.irreps)
        # just a useful dictionary
        kernel = self.model.ats_3
        # Starts with the kernel as the model (for the totally-symmetric representation)
        # generally, the co-kernel is the summation of irreducible representations in the expanded point group table which
        # have all of the symmetry operations of the irreducible representation under query (i.e. a super-set)
        # It's easiest to just list them.

        for name, atoms_j in self.irreps:
            # special case for e.g. Eg(x) and Eg(y) in D4h where the co-kernel is B1g, not A1g
            if isinstance(self.symm.e_group_parent, dict):
                if self.symm.e_group_parent.get(name, False):
                    kernel = self.irrep_dict.get(self.symm.e_group_parent.get(name))
            # special case for those where there are multiple co-kernels
            if isinstance(self.symm.e_group_parent_multi, dict):
                if (
                    self.symm.e_group_parent_multi.get(name, [False])[0]
                    in self.irrep_dict.keys()
                ):
                    kernel = kernel + np.sum(
                        [
                            {y[0]: y[2] for y in self.scsd_matrix}.get(x)
                            for x in self.symm.e_group_parent_multi.get(name)
                        ],
                        axis=0,
                    )
            # This is where we write each of the 'rows' of the scsd_matrix. There are no atoms in this version, but they can
            # be queried with self.ats[:,3]
            self.add_output_row(
                [
                    name,
                    round(np.sum([norm(x) for x in subtract(atoms_j, kernel)]), 4),
                    (atoms_j - kernel),
                    atoms_j,
                ]
            )
            # resets the kernel to the A1g representation (for non-A1g symmetries)
            kernel = self.scsd_matrix[0][3]
        # this just adds the raw data as a row - good for output and for plotting, though atoms_f can be referenced too
        self.add_output_row(["Atom Posits Data", 0, self.atoms_f * 0, self.atoms_f])
        return self

    def scsd_back_on_itself(self):
        # this generates a non-degenerate set - useful in Dxh x=3,5,6 where the E modes are fractional in their
        # contribution to the 'sum' structure. Basically, it's NSD but using the scsd matrices as the
        # vector sets - which should return the same values as scsd for (in D3h) A'1, A'2, etc. but will
        # make your E'1 E'2 and E'3 still sum to 2 (the ident. value) and likely go to 1, 1 and 0 instead of 2/3,
        # 2/3 and 2/3. So, generate a big matrix (A) and the measured distortion (B) and solve A * x = B for x
        # but, in this version, sequentially, to prevent values from going negative.
        # see numpy.linalg.lstsq for more details.

        # rewrite 09012021 - include E2u4/5/6 for boat cyclohexane-type C2v
        uniq = self.symm.e_group_unique_sets
        # unique symmetry operations i.e. the total representations from one 'E' orientation (C' or + pi/3 or -pi/3)
        uniq_f = np.array([x for y in uniq for x in y]).flatten()
        # ats = np.sum((scsd_matrix[0][3],np.sum([irrep[2]*float(tab) for irrep,tab in zip(scsd_matrix, symm_multiplicity_tables.get(symm))],axis = 0)),
        #  axis = 0)
        ats = self.scsd_matrix[-1][3]
        # the atom positions
        ats_0 = self.scsd_matrix[0][3]
        # the totally-symmetric or model
        mags = {
            self.scsd_matrix[x][0]: self.scsd_matrix[x][1]
            for x in range(len(self.scsd_matrix) - 1)
        }
        # scsd vals. used for ordering modes
        vecs = {
            self.scsd_matrix[x][0]: self.scsd_matrix[x][2]
            / np.sum(norm(self.scsd_matrix[x][2], axis=1))
            for x in range(len(self.scsd_matrix) - 1)
        }
        # these are our matrices, used as part of bigmat
        if self.ptgr == "Oh":
            order = np.argsort([sum([mags.get(x) for x in irreps]) for irreps in uniq])[
                ::-1
            ]
        else:
            order = np.argsort([mags.get(irreps[0]) for irreps in uniq])[::-1]
        # order for the E groups
        output = {self.scsd_matrix[0][0]: self.scsd_matrix[0][1]}
        # A1g isn't modelled this way. Easier this way.
        bigmat = []

        #       with open('C:\\Users\\ckingsbury\\outputs_mercury\\log.txt', 'a') as logfile:
        #           logfile.write(str(vecs))
        #
        for irrep in self.scsd_matrix[:-1]:
            if irrep[0] not in uniq_f:
                if np.sum(norm(irrep[2], axis=1)) > 0.01:
                    bigmat.append(irrep[2].flatten() / np.sum(norm(irrep[2], axis=1)))
                else:
                    bigmat.append(np.zeros(shape(irrep[2].flatten())))
        # matrices for those that don't have to
        for i in [uniq[x] for x in order]:
            for j in i:
                bigmat.append(vecs.get(j).flatten())
        names = [
            irrep[0] for irrep in self.scsd_matrix[:-1] if irrep[0] not in uniq_f
        ] + [a for b in [uniq[x] for x in order] for a in b]

        leastsq_serial = [self.scsd_matrix[0][1]]
        atvecs = (ats - ats_0).flatten()
        for mat in bigmat[1:]:
            val = lstsq(np.atleast_2d(mat).T, atvecs, rcond=None)[0]
            atvecs = atvecs - mat * val
            leastsq_serial.append(val.astype(float).flatten()[0])
        return {x: round(y, 4) for x, y in zip(names, leastsq_serial)}

    def simple_scsd(self):
        if hasattr(self, "scsd_simple"):
            return self.scsd_simple
        if self.ptgr in [
            "C3v",
            "D3d",
            "D3h",
            "D4d",
            "D5h",
            "D6h",
            "D8h",
            "Td",
            "Oh",
        ]:
            self.scsd_simple = self.scsd_back_on_itself()
            return self.scsd_simple
        scsd_simple = dict([[x[0], x[1]] for x in self.scsd_matrix])
        if self.ptgr == "D4h":
            egv = [scsd_simple.get(x) for x in "Egx,Egy,Egx+y,Egx-y".split(",")]
            euv = [scsd_simple.get(x) for x in "Eux,Euy,Eux+y,Eux-y".split(",")]
            if np.argmax(egv) < 1.5:
                [scsd_simple.pop(x) for x in ["Egx+y", "Egx-y"]]
            else:
                [scsd_simple.pop(x) for x in ["Egx", "Egy"]]
            if np.argmax(euv) < 1.5:
                [scsd_simple.pop(x) for x in ["Eux+y", "Eux-y"]]
            else:
                [scsd_simple.pop(x) for x in ["Eux", "Euy"]]
        if self.ptgr == "C4v":
            ev = [scsd_simple.get(x) for x in "Ex,Ey,Ex+y,Ex-y".split(",")]
            if np.argmax(ev) < 1.5:
                [scsd_simple.pop(x) for x in ["Ex+y", "Ex-y"]]
            else:
                [scsd_simple.pop(x) for x in ["Ex", "Ey"]]
        if self.ptgr == "D2d":
            ev = [scsd_simple.get(x) for x in "Ex,Ey,Ex+y,Ex-y".split(",")]
            if np.argmax(ev) < 1.5:
                [scsd_simple.pop(x) for x in ["Ex+y", "Ex-y"]]
            else:
                [scsd_simple.pop(x) for x in ["Ex", "Ey"]]

        scsd_simple.pop("Atom Posits Data", False)
        self.scsd_simple = scsd_simple
        return self.scsd_simple

    def scsd_plotly(self, maxdist=1.75, as_type="html", exaggerate=1.0, **kwargs):
        # Also figured out the D3h stuff here
        # Initialises the figure object
        fig = go.Figure()
        # figures out the connectivity, for the lines. Much more efficient than drawing a series per bond
        line_thru = generate_line_thru_points(self.model.ats_3, maxdist)
        xs, ys, zs = self.model.ats_3[line_thru, :].T
        fig.add_trace(
            go.Scatter3d(x=xs, y=ys, z=zs, mode="markers+lines", name="Model", **kwargs)
        )
        kernel = self.model.ats_3
        # this is the "Model" trace

        # This plots each of the irreducible representations. The data is kept in the 'nsd' list-of-lists.
        for irrep in self.scsd_matrix[:-1]:
            # xs,ys,zs = np.round(np.array([kernel[a] + irrep[2][a]*exaggerate for a in line_thru]).T,4)
            xs, ys, zs = np.round(
                (kernel[line_thru, :] + irrep[2][line_thru, :] * float(exaggerate)).T, 4
            )
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers+lines",
                    name=irrep_typog_html.get(irrep[0], irrep[0]),
                    visible=(True if float(irrep[1]) > 0.2 else "legendonly"),
                    **kwargs,
                )
            )

        # takes the 'data positions' from the nsd list, aligned to the same coordinates.
        # All going well, these should be same or at least within 0.0001A of the Atom Posits Sum
        # These aren't in the older versions - may have to recalculate databases.

        xs, ys, zs = np.round(np.array([self.atoms_f[a] for a in line_thru]).T, 4)
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers+lines",
                name=self.scsd_matrix[-1][0],
                visible=True,
                **kwargs,
            )
        )

        unique_irreps = [x for x, y in self.simple_scsd().items()]
        # Special stuff for D3h - the E modes need to be multiplied by 2/3
        # this will be similar for each of the Dxh and Dxd modes, I think...

        if isinstance(self.symm.symm_multiplicity_tables, list):
            ats = np.sum(
                (
                    np.array(self.scsd_matrix[0][3]),
                    np.sum(
                        [
                            float(tab) * np.array(irrep[2])
                            for irrep, tab in zip(
                                self.scsd_matrix, self.symm.symm_multiplicity_tables
                            )
                        ],
                        axis=0,
                    ),
                ),
                axis=0,
            )
            # for multiplying out the multiplicity! This broke when we used scsd_matrix = df['scsd_matrix'] so just recalculated
            xs, ys, zs = np.round(np.array([ats[x] for x in line_thru]).T, 4)
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers+lines",
                    name="Atom Posits Sum(*)",
                    **kwargs,
                )
            )

            xs, ys, zs = np.round(
                np.array(
                    np.array([self.atoms_f[a] for a in line_thru]).T
                    - np.array([self.scsd_matrix[0][2][a] for a in line_thru]).T
                    - np.sum(
                        [
                            np.array([float(tab) * irrep[2][a] for a in line_thru]).T
                            for irrep, tab in zip(
                                self.scsd_matrix, self.symm.symm_multiplicity_tables
                            )
                            if irrep[0] in unique_irreps
                        ],
                        axis=0,
                    )
                ),
                4,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers+lines",
                    name="Atom Posits Diff + Model",
                    **kwargs,
                )
            )

        else:
            xs, ys, zs = np.round(
                np.array([self.model.ats_3[a] for a in line_thru]).T
                + np.sum(
                    [
                        np.array([irrep[2][a] for a in line_thru]).T
                        for irrep in self.scsd_matrix
                        if irrep[0] in unique_irreps
                    ],
                    axis=0,
                ),
                4,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers+lines",
                    name="Atom Posits Sum",
                    **kwargs,
                )
            )

        fig.layout.scene.camera.projection.type = "orthographic"
        fig.layout.scene.aspectmode = "data"
        if as_type.lower() == "html":
            return to_html(fig)
        if as_type.lower() == "html_min":
            return to_html(fig, include_plotlyjs="cdn")
        elif as_type.lower() == "fig":
            return fig

    def mondrian(
        self,
        imagepath="",
        as_type=False,
        cmap=random.choice(good_cmaps),
        hatchwork=False,
        custom_axes=False,
        dpi=300,
    ):
        # work on some analysis of how to get reasonable axes.
        # hatchwork = False
        # this can wait, change for the colourblind or for b/w printing, etc.
        linewidth, bleed = 3, 0.0001
        # looks fine

        # cmap = 'Spectral', being the default colourmap, looks kinda nice.
        if cmap == "random":
            cmap = random.choice(good_cmaps)
        # lookup_dict = mondrian_lookup_dict.get(symmetry)
        # orientation_dict = mondrian_orientation_dict.get(symmetry)
        # pgt = pgt_dict.get(symmetry)

        if type(self.model.mondrian_limits) is type(None):
            xmin, ymin = -1, -1
        else:
            xmin, ymin = self.model.mondrian_limits
        # takes the total distortion as the starting point, ignores zeros
        self.simple_scsd()
        vvs = np.array(
            [
                x
                for y, x in self.scsd_simple.items()
                if (
                    (self.symm.mondrian_orientation_dict.get(y) == "v")
                    and (x > 10**ymin)
                )
            ]
        ).astype(float)
        vns = np.array(
            [
                y
                for y, x in self.scsd_simple.items()
                if (
                    (self.symm.mondrian_orientation_dict.get(y) == "v")
                    and (x > 10**ymin)
                )
            ]
        )
        hvs = np.array(
            [
                x
                for y, x in self.scsd_simple.items()
                if (
                    (self.symm.mondrian_orientation_dict.get(y) == "h")
                    and (x > 10**xmin)
                )
            ]
        ).astype(float)
        hns = np.array(
            [
                y
                for y, x in self.scsd_simple.items()
                if (
                    (self.symm.mondrian_orientation_dict.get(y) == "h")
                    and (x > 10**xmin)
                )
            ]
        )
        vss, hss = sort(vvs), sort(hvs)
        # Sets the borders of the plot
        # if type(custom_axes) in [bool]:
        #    xmax = np.max([log10(x)+0.1 for x in concatenate((vvs, hvs,[8.0]))])
        # else:
        xmax = np.max(
            [np.log10(x) + 0.1 for x in np.concatenate((vvs, hvs, [(10**xmin) * 80]))]
        )
        ymax = xmax

        # print(xmin,ymin,xmax,ymax)
        # defines the 'flat' panes of the plot, by making points at corners of a rectangle.
        # This is more efficient than making a grid of the entire area, as symmetry doesn't change within panels.
        x_op = sort(np.hstack([10**xmin, hss - bleed, hss + bleed, 10**xmax]))
        y_ip = sort(np.hstack([10**ymin, vss - bleed, vss + bleed, 10**ymax]))

        # generates an ordered list of symmetry operations
        sym1 = [
            find_symmetry_arbitrary(
                [n for v, n in zip(vvs, vns) if v > y]
                + [n for v, n in zip(hvs, hns) if v > x],
                self.symm.pgt,
                self.symm.mondrian_lookup_dict,
            )
            for x, y in zip(*[z.flatten() for z in np.meshgrid(x_op, y_ip)])
        ]
        ops1 = [
            [n for v, n in zip(vvs, vns) if v > y]
            + [n for v, n in zip(hvs, hns) if v > x]
            for x, y in zip(*[z.flatten() for z in np.meshgrid(x_op, y_ip)])
        ]
        # [print(x,y) for x,y in zip(sym1, ops1)]
        sym2, sym3 = np.unique(sym1, return_index=True)
        try:
            sym3.sort()
        except TypeError:
            pass
        syms = [sym1[x] for x in sym3]

        ludict = {y: x + 0.5 for x, y in enumerate(syms)}

        if ";" in cmap:
            cmap = cmap.split(";")
            cmap = [x.lstrip(" ") for x in cmap]
        cmap_len = len(color_palette(cmap).as_hex())

        if cmap in safe_cmaps:
            cmap_o = color_palette(cmap, len(syms)).as_hex()
        elif len(syms) > cmap_len:
            cmap_o = color_palette(cmap, len(syms)).as_hex()
        else:
            cmap_o = color_palette(cmap).as_hex()

        if hatchwork is True:
            hats = hatch_picks.copy()
            random.shuffle(hats)
        elif hatchwork is False:
            hats = [None]
        else:
            hats = hatchwork

        nx, ny = np.meshgrid(x_op, y_ip)
        nz = np.array([ludict.get(x) for x in sym1]).reshape(len(y_ip), len(x_op))

        # plot setup
        fig, ax = subplots(figsize=(9, 7))

        # The primary plot
        im = ax.contourf(
            nx,
            ny,
            nz,
            colors=cmap_o,
            corner_mask=False,
            levels=np.abs(len(syms) - 2),
            hatches=hats,
        )
        ax.set_yscale("log")
        ax.set_xscale("log")

        # The colourbar
        cbar = fig.colorbar(im, drawedges=True)
        for index, value in enumerate(syms):
            cbar.ax.text(
                x=1.66,
                y=(index + 0.5) / (len(syms)),
                s=symm_typog_lookup.get(value, value),
                ha="center",
                va="center",
                transform=cbar.ax.transAxes,
            )
        cbar.ax.get_yaxis().set_ticks([])
        cbar.outline.set_linewidth(linewidth)
        cbar.dividers.set_linewidth(linewidth)

        # plots horizonatal and vertical lines, indicating symmetry operations above threshold
        lineweave_x_x = [i for j in [[x, x, x] for x in hvs] for i in j]
        lineweave_x_y = [
            i
            for j in [[a, b, a] for a, b in [[np.min(ny), np.max(ny)]] * len(hvs)]
            for i in j
        ]
        lineweave_y_x = [
            i
            for j in [[a, b, a] for a, b in [[np.min(nx), np.max(nx)]] * len(vvs)]
            for i in j
        ]
        lineweave_y_y = [i for j in [[y, y, y] for y in vvs] for i in j]
        # for x in hvs: ax.plot([x,x],[np.min(ny),np.max(ny)], 'black', lw = linewidth)
        ax.plot(lineweave_x_x, lineweave_x_y, "black", lw=linewidth)
        ax.plot(lineweave_y_x, lineweave_y_y, "black", lw=linewidth)

        # Labels on horizontal/vertical lines.
        if len(vns) > 0:
            [
                ax.text(
                    1.005,
                    1 - (ymax - np.log10(value)) / (ymax - ymin),
                    irrep_typog.get(name, name),
                    ha="left",
                    va="center",
                    transform=ax.transAxes,
                )
                for name, value in zip(vns, vvs)
            ]
        if len(hns) > 0:
            [
                ax.text(
                    1 - (xmax - np.log10(value)) / (xmax - xmin),
                    1.01,
                    irrep_typog.get(name, name),
                    ha="center",
                    va="bottom",
                    rotation=45,
                    transform=ax.transAxes,
                )
                for name, value in zip(hns, hvs)
            ]

        ax.set_ylabel("Sum Distortion ($\\AA$)")
        ax.set_xlabel("Sum Distortion ($\\AA$)")
        [i.set_linewidth(linewidth) for i in ax.spines.values()]
        [i.set_linewidth(linewidth) for i in cbar.ax.spines.values()]
        if as_type == "buffer":
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, backend="Agg")
            image_base64 = (
                base64.b64encode(buf.getvalue()).decode("utf-8").replace("\n", "")
            )
            buf.close()
            return image_base64
        if as_type == "fig":
            return fig
        else:
            fig.savefig(imagepath, dpi=dpi)
            return cmap

    def mondrian_plotly(
        self, as_type="fig", cmap="Spectral"
    ):  # ,linewidth = 3,bleed = 0.0001,
        # random_order = False, regular_axes = True, hatchwork = False):
        # nsd, symmetry, imagepath = '', as_fig = False

        custom_axes = self.model.mondrian_limits
        bleed = 0.0001
        nsd = self.simple_scsd().items()
        symmetry = self.ptgr

        if type(custom_axes) in [bool]:
            xmin, xmax, ymin, ymax = -1.5, 1, -1.5, 1
        else:
            xmin, xmax, ymin, ymax = custom_axes

        lookup_dict = mondrian_lookup_dict.get(symmetry)
        orientation_dict = mondrian_orientation_dict.get(symmetry)
        pgt = pgt_dict.get(symmetry)

        # takes the total distortion as the starting point, ignores zeros
        vvs = np.array(
            [
                x
                for y, x in self.simple_scsd().items()
                if (
                    (self.symm.mondrian_orientation_dict.get(y) == "v")
                    and (x > 10**ymin)
                )
            ]
        ).astype(float)
        vns = np.array(
            [
                y
                for y, x in self.simple_scsd().items()
                if (
                    (self.symm.mondrian_orientation_dict.get(y) == "v")
                    and (x > 10**ymin)
                )
            ]
        )
        hvs = np.array(
            [
                x
                for y, x in self.simple_scsd().items()
                if (
                    (self.symm.mondrian_orientation_dict.get(y) == "h")
                    and (x > 10**xmin)
                )
            ]
        ).astype(float)
        hns = np.array(
            [
                y
                for y, x in self.simple_scsd().items()
                if (
                    (self.symm.mondrian_orientation_dict.get(y) == "h")
                    and (x > 10**xmin)
                )
            ]
        )
        vss, hss = sort(vvs), sort(hvs)

        # defines the 'flat' panes of the plot, by making points at corners of a rectangle.
        # This is more efficient than making a grid of the entire area, as symmetry doesn't change within panels.
        x_op = sort(np.hstack([10**xmin, hss - bleed, hss + bleed, 10**xmax]))
        y_ip = sort(np.hstack([10**ymin, vss - bleed, vss + bleed, 10**ymax]))

        # gernerates an ordered list of symmetry operations
        sym1 = [
            find_symmetry_arbitrary(
                [n for v, n in zip(vvs, vns) if v > y]
                + [n for v, n in zip(hvs, hns) if v > x],
                pgt,
                lookup_dict,
            )
            for x, y in zip(*[z.flatten() for z in np.meshgrid(x_op, y_ip)])
        ]
        sym2, sym3 = np.unique(sym1, return_index=True)
        try:
            sym3.sort()
        except RuntimeError:
            pass
        syms = [sym1[x] for x in sym3]
        syms_dict = dict([[y, x] for x, y in enumerate(syms)])
        ludict = {y: x + 0.5 for x, y in enumerate(syms)}

        x0s, x1s = x_op[::2], x_op[1::2]
        y0s, y1s = y_ip[::2], y_ip[1::2]

        cmap_o = color_palette(cmap, len(syms)).as_hex()

        # plot setup
        fig2 = go.Figure()
        fig2.update_xaxes(type="log", range=[-1.5, 1.1], showgrid=False)
        fig2.update_yaxes(type="log", range=[-1.5, 1.1], showgrid=False)
        colorbar_info = []
        for x0, x1 in zip(x0s, x1s):
            for y0, y1 in zip(y0s, y1s):
                ops = [n for v, n in zip(vvs, vns) if v > y0] + [
                    n for v, n in zip(hvs, hns) if v > x0
                ]
                symm = find_symmetry_arbitrary(ops, pgt, lookup_dict)
                trans = find_symmetry_arbitrary(
                    ops, pgt, mondrian_transform_dict.get(symmetry)
                )
                for k, v in irrep_typog_html.items():
                    trans = trans.replace(k, v)
                fig2.add_shape(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    fillcolor=cmap_o[syms_dict.get(symm)],
                    layer="below",
                )
                fig2.add_trace(
                    go.Scatter(
                        x=[np.sqrt(x1 * x0)],
                        y=[np.sqrt(y1 * y0)],
                        marker=dict(color=cmap_o[syms_dict.get(symm)]),
                        name=symm,
                        text="Symmetry : "
                        + symm_typog_html.get(symm, symm)
                        + "<br> IRs : "
                        + ", ".join([irrep_typog_html.get(x, "") for x in ops])
                        + "<br> transforms : "
                        + trans,
                        hoverinfo=["text"],
                        showlegend=False,
                    )
                )

        fig2.update_layout(
            autosize=False,
            width=800,
            height=800,
            title="Mondrian symmetry diagram",
            yaxis=dict(title_text="Sum Distortion (&#8491;)"),
            xaxis=dict(title_text="Sum Distortion (&#8491;)"),
            plot_bgcolor="white",
        )

        for n, v in zip(hns, hvs):
            fig2.add_annotation(
                x=np.log10(v), y=1.1, text=irrep_typog_html.get(n), showarrow=False
            )
        for n, v in zip(vns, vvs):
            fig2.add_annotation(
                x=1.1, y=np.log10(v), text=irrep_typog_html.get(n), showarrow=False
            )

        if as_type.lower() == "html":
            return to_html(fig2)
        elif as_type.lower() == "fig":
            return fig2

    def nsd_dict(self, n_modes=False):
        scsd_dict = dict([[x[0], x[2]] for x in self.scsd_matrix])
        nsd_dict = {}

        if type(n_modes) is int:
            pc = {x: y[: min((n_modes, len(y)))] for x, y in self.model.pca.items()}

        else:
            pc = self.model.pca
        for key, pc_list in pc.items():
            a1 = np.atleast_2d(np.array(
                [np.array(component).flatten() for component in pc_list]
            )).T.astype(np.float32)
            # a1 is the PCAs - 3*N coordinates x M components
            a2 = np.atleast_2d(np.array(scsd_dict.get(key).flatten())).T.astype(np.float32)
            # a2 is the symmetry coordinates - 3*N coordinates x 1
            x, residuals, rank, s = lstsq(a1, a2, rcond=None)
            # x being the NSD value
            a3 = np.subtract(a2.T, np.sum(np.multiply(x.T, a1), axis=1).T).T
            # a3 should be the vector between the best fit a1*x = a2 and the measured position
            linear_err = norm(a3.reshape(int(len(a3) / 3), 3), axis=1).sum()
            nsd_dict[key] = [x, linear_err]
        return nsd_dict

    def simple_nsd(self, n_modes=False):
        simple_nsd = {}
        for key, val_list in self.nsd_dict(n_modes).items():
            for index, val in enumerate(np.array(val_list[0]).flatten()):
                simple_nsd[key + "(" + str(index + 1) + ")"] = val
        return simple_nsd

    def scsd_dict(self):
        return dict([[x[0], x[1]] for x in self.scsd_matrix])

    def html_table(self, n_modes=False):
        # scsd_matrix, scsd_pc, model, n_modes = False):
        # makes a simple html table from a simple nsd table
        if (type(self.model.pca) is type(None)) or (n_modes == 0):
            # makes a simple html table from a simple nsd table
            arr_df = DataFrame(self.simple_scsd().items())
            arr_df.columns = ["Symm", "Sum (&#8491;)"]
            arr_df["Symm"] = [irrep_typog_html.get(x) for x in arr_df["Symm"]]
            arr_df["Sum (&#8491;)"] = [round(x, 2) for x in arr_df["Sum (&#8491;)"]]
            return arr_df.to_html(
                escape=False, index=False, justify="center", table_id="scsd_table"
            )
        else:
            nsd_dict = {x: np.round(y[0], 2) for x, y in self.nsd_dict(n_modes).items()}
            err_dict = {x: np.round(y[1], 2) for x, y in self.nsd_dict(n_modes).items()}
            scsd_dict = dict([[x[0], np.round(x[1], 2)] for x in self.scsd_matrix])
            keys_short = [x for x in nsd_dict.keys()]
            keys_extra = [
                x
                for x in scsd_dict.keys()
                if (x not in keys_short) and (x != "Atom Posits Data")
            ]
            arr = [
                [
                    key,
                    scsd_dict.get(key, ""),
                    err_dict.get(key, ""),
                    *nsd_dict.get(key, [""]).flatten(),
                ]
                for key in keys_short
            ] + [[key, scsd_dict.get(key, "")] for key in keys_extra]
            arr_df = DataFrame(arr).fillna("")

            arr_df.columns = ["Symm", "Sum (SCSD, &#8491;)", "PCA Err"] + [
                "PCA (" + str(i + 1) + ")"
                for i in range(max([len(y) for x, y in nsd_dict.items()]))
            ]
            arr_df["Symm"] = [irrep_typog_html.get(x, x) for x in arr_df["Symm"]]
            return arr_df.to_html(
                escape=False, index=False, justify="center", table_id="scsd_table"
            )

    def dev_table(self):
        if self.ptgr == "C2v":
            ax = (0, 2, 1)
        else:
            ax = (0, 1, 2)

        def cc_transform_thetaonly_deg(mat):
            return np.atleast_2d(
                np.degrees(
                    np.array((np.arctan2(mat[:, ax[1]], mat[:, ax[0]])) % 6.28318)
                )
            ).T

        def cc_transform_ronly(mat):
            return np.atleast_2d(norm(mat[:, ax], axis=1)).T

        th = cc_transform_thetaonly_deg(self.atoms_f)
        r = cc_transform_ronly(self.atoms_f)
        out = DataFrame(np.hstack([self.atoms_f, th, r]))
        out.columns = ["x", "y", "z", "th", "r"]
        return out

    def dev_plot(self, rotate=0.0):
        xs, ys, zs, th, r = self.dev_table().values.T
        th = np.array([(a - rotate) % 360 for a in th])
        if self.ptgr == "C2v":
            ax = (0, 2, 1)
        else:
            ax = (0, 1, 2)
        line_thru = generate_line_thru_points_plus_theta(self.atoms_f[:, ax], th, 1.8)
        for i in range(5):
            if (len(np.unique(line_thru)) < len(xs)) and (rotate == 0.0):
                th = np.array([(a - random(0, 25)) % 360 for a in th])
                line_thru = generate_line_thru_points_plus_theta(self.atoms_f, th, 1.8)
        z2, th2 = np.array([[zs[a], th[a]] for a in line_thru]).T
        import plotly.express as px

        fig = px.line(y=z2, x=th2)
        fig.update_layout(xaxis_title="&#x3B8; (&deg;)", yaxis_title="Z (&#8491;)")
        fig.data[0].update(mode="markers+lines")
        return fig

    def check_residuals(self):
        xs, ys, zs, th, r = self.dev_table().values.T
        sums = [str(sum(i - j)) for i, j in zip([xs, ys, zs], self.model.ats_3.T)]

        def cct(mat):
            np.atleast_2d(np.degrees(np.array((np.arctan2(mat[:, 1], mat[:, 0]))))).T

        # cylindrical coordinate transform
        x_rot = (
            sum(cct(self.atoms_f[:, [1, 2, 0]]))
            - sum(cct(self.model.ats_3[:, [1, 2, 0]])) % 360
        )
        y_rot = (
            sum(cct(self.atoms_f[:, [2, 0, 1]]))
            - sum(cct(self.model.ats_3[:, [2, 0, 1]])) % 360
        )
        z_rot = (
            sum(cct(self.atoms_f[:, [0, 1, 2]]))
            - sum(cct(self.model.ats_3[:, [0, 1, 2]])) % 360
        )
        [sums.append(str(min([i, 360 - i]))) for i in [x_rot, y_rot, z_rot]]
        return ", ".join(sums)

    def dev_html(self):
        return "\n".join(
            [
                "",
                "dev_table:",
                str(np.round(self.dev_table().values, 4)),
                "",
                "line thru:",
                str(generate_line_thru_points(self.atoms_f, 1.8)),
            ]
        )  # ,

    #            'residuals:', str(self.check_residuals())])
    # residuals check needs work

    def raw_data(self):
        labels, out = (
            "mode_name, sum_dist, dev_mat, dev_mat_plus_total, atom_names".split(", "),
            "",
        )
        for mode in self.scsd_matrix:
            mode_dict = {x: y for x, y in zip(labels, mode)}
            out += (
                "\n".join([" : ".join([x, str(y)]) for x, y in mode_dict.items()])
            ) + "\n"
        out += self.dev_html()
        return out

    def compare_table(self, data_path=str(Path(dirname(__file__)) / 'data' / 'scsd'), bypass=False, nearest_dict={}, struct_num=5):

        if "pdb" in self.model.name:
            if bypass is True:
                return make_table_2_pdb(nearest_dict.keys(), nearest_dict.values())
            elif self.model.database_path is None:
                return ""
            else:
                try:
                    df = read_pickle(Path(data_path) / self.model.database_path)
                except FileNotFoundError:
                    return ""
                # if self.ptgr in ['C2h','D2h','C2v']:
                # some work needs to be done on a rational similarity measure for 3,4,5-etc. axes E/T groups
                mindices = np.argsort(
                    np.array(
                        np.sum(
                            np.vstack(
                                [
                                    np.square(df[x].values - y)
                                    for x, y in self.simple_scsd().items()
                                ]
                            ),
                            axis=0,
                        )
                    )
                ).flatten()[:struct_num]
                min_names = df["name"][mindices].values
                min_vals = sort(
                    np.array(
                        np.sqrt(
                            np.sum(
                                np.vstack(
                                    [
                                        np.square(df[x][mindices].values - y)
                                        for x, y in self.simple_scsd().items()
                                    ]
                                ),
                                axis=0,
                            )
                        )
                    )
                ).flatten()
                return make_table_2_pdb(min_names, min_vals)

        if bypass is True:
            return make_table_2(nearest_dict.keys(), nearest_dict.values())
        if self.model is None:
            return ""
        elif ((self.model.pca is None) or (self.model.database_path is None)):
            return ""
        else:
            pca = self.model.pca
            try:
                df = read_pickle(Path(data_path) / self.model.database_path)
            except FileNotFoundError:
                return ""

            mindices = np.argsort(
                np.array(
                    np.sum(
                        np.vstack(
                            [
                                np.square(df[x].values - y)
                                for x, y in self.simple_scsd().items()
                            ]
                        ),
                        axis=0,
                    )
                )
            ).flatten()[:struct_num]
            min_names = df["name"][mindices].values
            min_vals = sort(
                np.array(
                    np.sqrt(
                        np.sum(
                            np.vstack(
                                [
                                    np.square(df[x][mindices].values - y)
                                    for x, y in self.simple_scsd().items()
                                ]
                            ),
                            axis=0,
                        )
                    )
                )
            ).flatten()
            return make_table_2(min_names, min_vals)


def check_unmoved_atoms(atom_matrix, typo, theta, phi, rotation):
    omega = (2 * pi) / rotation
    refvec = s2c(theta, phi)
    rot_mat = q2r(*s2q(theta, phi, omega))
    trans_atoms = atom_matrix.copy()
    if typo in ["improperrotation", "mirror"]:
        trans_atoms = trans_atoms - np.outer(2 * np.dot(trans_atoms, refvec), refvec)
    if typo in ["rotation", "improperrotation"]:
        trans_atoms = np.dot(rot_mat, trans_atoms.T).T
    if typo == "inversion":
        trans_atoms = -atom_matrix
    costmat = costmat_gen(atom_matrix, trans_atoms)
    row_ind, col_ind = linear_sum_assignment(costmat)
    return int(sum([1 if x == y else 0 for x, y in zip(row_ind, col_ind)]))


def vibran_mod(ptgr, model):
    # Adapted from VIBRAN (see notes)
    ops = [operations_dict.get(x)[0] for x in ordered_ops_dict.get(ptgr)]
    pgt = np.array([y for x, y in pgt_dict.get(ptgr).items()])

    r = [check_unmoved_atoms(model, *x) for x in ops]

    if ptgr == "C2v":
        A = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])
        b = [[(r[0] - 2) * 3], [(r[1] - 2) * -1], [r[2] * 1], [r[3] * 1]]
    elif ptgr == "C2h":
        A = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
        b = [[(r[0] - 2) * 3], [(r[1] - 2) * -1], [r[2] * -3], [r[3] * 1]]
    elif ptgr == "D2h":
        A = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, 1, -1, 1, -1, 1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1],
                [1, 1, 1, 1, -1, -1, -1, -1],
                [1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, 1, -1, -1, 1, -1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1],
            ]
        )
        b = [
            [(r[0] - 2) * 3],
            [(r[1] - 2) * -1],
            [(r[2] - 2) * -1],
            [(r[3] - 2) * -1],
            [(r[4]) * -3],
            [(r[5]) * 1],
            [(r[6]) * 1],
            [(r[7]) * 1],
        ]
    elif ptgr == "D4h":
        A = np.array(
            [
                [1, 1, 1, 1, 2, 1, 1, 1, 1, 2],
                [1, 1, -1, -1, 0, 1, 1, -1, -1, 0],
                [1, 1, 1, 1, -2, 1, 1, 1, 1, -2],
                [1, -1, 1, -1, 0, 1, -1, 1, -1, 0],
                [1, -1, -1, 1, 0, 1, -1, -1, 1, 0],
                [1, 1, 1, 1, 2, -1, -1, -1, -1, -2],
                [1, 1, -1, -1, 0, -1, -1, 1, 1, 0],
                [1, 1, 1, 1, -2, -1, -1, -1, -1, 2],
                [1, -1, 1, -1, 0, -1, 1, -1, 1, 0],
                [1, -1, -1, 1, 0, -1, 1, 1, -1, 0],
            ]
        )
        b = [
            [(r[0] - 2) * 3],
            [(r[1] - 2) * 1],
            [(r[2] - 2) * -1],
            [(r[3] - 2) * -1],
            [(r[5] - 2) * -1],
            [(r[7]) * -3],
            [(r[8]) * -1],
            [(r[9]) * 1],
            [(r[10]) * 1],
            [(r[12]) * 1],
        ]
    elif ptgr == "D3h":
        A = np.array(
            [
                [1, 1, 2, 1, 1, 2],
                [1, 1, -1, 1, 1, -1],
                [1, -1, 0, 1, -1, 0],
                [1, 1, 2, -1, -1, -2],
                [1, 1, -1, -1, -1, 1],
                [1, -1, 0, -1, 1, 0],
            ]
        )
        b = [
            [(r[0] - 2) * 3],
            [(r[1] - 2) * (1 + 2 * (-0.5))],
            [(r[2] - 2) * (-1)],
            [r[5]],
            [r[6] * (-2)],
            [r[7]],
        ]
    elif ptgr == "D6h":
        A = np.array(
            [
                [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2],
                [1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1],
                [1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
                [1, 1, -1, -1, -2, 2, 1, 1, -1, -1, -2, 2],
                [1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0],
                [1, -1, -1, 1, 0, 0, 1, -1, -1, 1, 0, 0],
                [1, 1, 1, 1, 2, 2, -1, -1, -1, -1, -2, -2],
                [1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
                [1, 1, -1, -1, -2, 2, -1, -1, 1, 1, 2, -2],
                [1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 0, 0],
                [1, -1, -1, 1, 0, 0, -1, 1, 1, -1, 0, 0],
            ]
        )
        b = [
            [(int(r[0])) * (2 * cos(0) + 1)],
            [(int(r[1])) * (2 * (0.5) + 1)],
            [(int(r[2])) * (2 * (-0.5) + 1)],
            [(int(r[3])) * (2 * cos(pi) + 1)],
            [(int(r[4])) * (2 * cos(pi) + 1)],
            [(int(r[7])) * (2 * cos(pi) + 1)],
            [(-int(r[10])) * (2 * cos(2 * pi) + 1)],
            [(-int(r[11])) * (2 * (0.5) + 1)],
            [(-int(r[12])) * (2 * (-0.5) + 1)],
            [(-int(r[13])) * (2 * cos(pi) + 1)],
            [(-int(r[14])) * (2 * cos(pi) + 1)],
            [(-int(r[17])) * (2 * cos(pi) + 1)],
        ]
    # unfinished from here
    elif ptgr == "D8h":
        return {lab: 2 for lab in labels_dict_ext.get("D8h").keys()}
    elif ptgr == "Td":
        A = np.array(
            [
                [1, 1, 2, 3, 3],
                [1, 1, -1, 0, 0],
                [1, 1, 2, -1, -1],
                [1, -1, 0, 1, -1],
                [1, -1, 0, -1, 1],
            ]
        )
        b = [
            [(int(r[0]) - 2) * (1 + 2 * cos(0))],
            [(int(r[1]) - 2) * (2 * (-0.5) - 1)],
            [(int(r[9]) - 2) * (1 + 2 * cos(pi))],
            [(int(r[12])) * (-1 + 2 * 0)],
            [(int(r[18])) * (-1 + 2 * cos(0))],
        ]
    elif ptgr == "Oh":
        A = np.array(
            [
                [1, 1, 2, 3, 3, 1, 1, 2, 3, 3],
                [1, 1, -1, 0, 0, 1, 1, -1, 0, 0],
                [1, -1, 0, -1, 1, 1, -1, 0, -1, 1],
                [1, -1, 0, 1, -1, 1, -1, 0, 1, -1],
                [1, 1, 2, -1, -1, 1, 1, 2, -1, -1],
                [1, 1, 2, 3, 3, -1, -1, -2, -3, -3],
                [1, -1, 0, 1, -1, -1, 1, 0, -1, 1],
                [1, 1, -1, 0, 0, -1, -1, 1, 0, 0],
                [1, 1, 2, -1, -1, -1, -1, -2, 1, 1],
                [1, -1, 0, -1, 1, -1, 1, 0, 1, -1],
            ]
        )
        b = [
            [(int(r[0]) - 2) * (1 + 2 * cos(0))],
            [(int(r[1]) - 2) * (2 * (-0.5) + 1)],
            [(int(r[9]) - 2) * (1 + 2 * cos(pi))],
            [(int(r[15]) - 2) * (1 + 2 * 0)],
            [(int(r[21]) - 2) * (1 + 2 * cos(pi))],
            [int(r[24]) * (-1 + 2 * cos(pi))],
            [int(r[25]) * (-1 + 2 * 0)],
            [int(r[31]) * (-1 + 2 * (0.5))],
            [int(r[39]) * (-1 + 2 * cos(0))],
            [int(r[42]) * (-1 + 2 * cos(0))],
        ]
    else:
        return {lab: 2 for lab in labels_dict_ext.get(ptgr).keys()}

    x = solve(A, b)
    dict_1 = {lab: int(v) for lab, v in zip(labels_dict.get(ptgr), x)}
    if ptgr in labels_dict_ext.keys():
        return {l1: dict_1.get(l2) for l1, l2 in labels_dict_ext.get(ptgr).items()}
    else:
        return dict_1


class scsd_collection:
    # catch-all for 'more than one' structure - collects the input and output types for database handles
    # requires a model
    def __init__(self, model_input, simple_df=None):
        import pandas as pd
        try:
            from . import scsd_models_user
        except ImportError:
            pass

        if isinstance(model_input, scsd_model):
            self.model = model_input
        elif isinstance(model_input, str):
            self.model = model_objs_dict.get(
                model_input, scsd_models_user.model_objs_dict.get(model_input)
            )
        self.simple_df = simple_df

        if self.model.pca is not None:
            self.pca = self.model.pca
        if (self.model.database_path is not None) and (self.simple_df is None):
            try:
                self.simple_df = read_pickle(data_path / self.model.database_path)
                self.gen_simple_df(bypass=True)
            except FileNotFoundError:
                print(f"file not found at {data_path}{self.model.database_path}")
                self.simple_df = simple_df
#        repl_dict = {"NAME": "name", "NSD_matrix": "scsd", "coords_matrix": "coords"}
#        try:
#            self.simple_df.columns = [
#                repl_dict.get(x, x) for x in self.simple_df.columns.values
#            ]
#        except:
#            pass

    def sd_file_to_simple_df(
        self, filepath, verbose=False, n_structures=-1, bhop=False, by_graph=False
    ):
        sd_file = open(filepath, "r").read()
        sd_list = [x.split("\n") for x in sd_file.split("$$$$\n")][:n_structures]
        output_list = []

        for num, data in enumerate(sd_list):
            n_atoms = int(data[3][0:3])
            n, xyzc = data[0], [
                [float(a) for a in i.split()[0:3]] + [i.split()[3]]
                for i in data[4: 4 + n_atoms]
            ]
            if len(xyzc) < len(self.model.ats_3):
                print("skipped:" + str(n))
            else:
                scsd = scsd_matrix(np.array(xyzc), self.model)
                scsd.calc_scsd(bhop, by_graph=by_graph)
                output_list.append([n, scsd.scsd_matrix, scsd.atoms_f])
                if verbose:
                    print(str(num + 1) + "/" + str(len(sd_list)) + ":" + str(n))

        scsd_df = DataFrame(output_list)
        scsd_df.columns = ["name", "scsd", "coords"]
        self.simple_df = scsd_df
        return scsd_df

    def gen_simple_df(
        self, input_fdl=False, verbose=False, bhop=False, bypass=False, by_graph=False
    ):
        # filepath (of sd or pickle of dataframe), dataframe (with 'name' and coords' columns) or list of
        # names and coordinates
        if type(input_fdl) is str:
            if input_fdl.endswith(".sd"):
                return self.sd_file_to_simple_df(
                    input_fdl, verbose, bhop=bhop, by_graph=by_graph
                )
            if input_fdl.endswith(".pkl"):
                df = read_pickle(input_fdl)
        elif type(input_fdl) is bool:
            df = self.simple_df
        elif type(input_fdl) is DataFrame:
            df = input_fdl

        if type(input_fdl) is list:
            iterator = input_fdl
        else:
            n_ident = [x for x in df.columns if x.lower().startswith("name")][0]
            c_ident = [x for x in df.columns if x.lower().startswith("coord")][0]
            iterator = df[[n_ident, c_ident]].values

        output_list = []

        for num, (n, xyzc) in enumerate(iterator):
            scsd = scsd_matrix(np.array(xyzc), self.model)
            scsd.calc_scsd(bhop, bypass=bypass, by_graph=by_graph)
            output_list.append([n, scsd.scsd_matrix, scsd.atoms_f])
            if verbose:
                print(str(num + 1) + "/" + str(len(iterator)) + ":" + str(n))

        scsd_df = DataFrame(output_list)
        scsd_df.columns = ["name", "scsd", "coords"]
        self.simple_df = scsd_df
        return scsd_df

    def recalc_simple_df_row(self, name, retry=True, by_graph=False):
        row_in = self.simple_df[self.simple_df["name"] == name]
        index = row_in.index[0]
        xyz = row_in["coords"].values[0]
        if retry:
            xyz = fit_atoms_rt([random.random() for x in range(7)], xyz)
        scsd = scsd_matrix(np.array(xyz), self.model)
        scsd.calc_scsd(True, False, True, by_graph)
        df_out = DataFrame([[index, name, scsd.scsd_matrix, scsd.atoms_f]])
        df_out.columns = ["index", "name", "scsd", "coords"]
        self.simple_df.update(df_out.set_index("index"))

    def get_pca_from_model(self):
        self.pca = self.model.pca

    def gen_pca(self, n_modes=False):
        modes = [x[0] for x in self.simple_df["scsd"].values[0]]
        try:
            mode_lengths = vibran_mod(self.model.ptgr, self.model.ats_3)
        except LinAlgError:
            mode_lengths = {x: 2 for x in labels_dict_ext.get(self.model.ptgr).keys()}

        if n_modes:
            mode_lengths = {x: min((y, n_modes)) for x, y in mode_lengths.items()}
        # if self.model.ptgr in labels_dict_ext.keys():
        #    mode_lengths = {x:mode_lengths.get(y) for x,y in labels_dict_ext.get(self.model.ptgr).items()}
        self.pca = {}
        for index, mode in enumerate(modes):
            if mode_lengths.get(mode) not in (None, 0):
                flatmat = [y[index][2].flatten() for y in self.simple_df["scsd"]]
                pca = PCA(n_components=mode_lengths.get(mode), random_state=2022)
                pca.fit(flatmat)
                pc = np.array(
                    [
                        x / np.sum(norm(x.reshape((int(len(x) / 3), 3)), axis=1))
                        for x in pca.components_
                    ]
                )
                self.pca[mode] = pc
        return self.pca

    def gen_complex_df(self):
        self.complex_df = self.simple_df.copy(deep=True)
        if type(self.pca) is type(None):
            self.gen_pca()

        for index, mode in enumerate(self.model.symm.pgt.keys()):
            self.complex_df[mode] = [y[index][1] for y in self.complex_df["scsd"]]
            self.complex_df[mode + "m"] = [y[index][2] for y in self.complex_df["scsd"]]

        for index, (name, modes) in enumerate(self.pca.items()):
            flatmat = [mat.flatten() for mat in self.complex_df[name + "m"].values]
            mode_nms = [name + "(" + str(i + 1) + ")" for i in range(len(modes))]
            x, residuals, rank, s = lstsq(
                np.atleast_2d(modes).T, np.atleast_2d(np.array(flatmat)).T, rcond=None
            )
            if len(mode_nms) == 1:
                self.complex_df[name + "(1)"] = x.flatten()
            else:
                for lab, val in zip(mode_nms, x):
                    self.complex_df[lab] = val

        for index, (name, modes) in enumerate(self.pca.items()):
            if index != 0:
                mode_nms_f = [name + "(" + str(i + 1) + ")f" for i in range(len(modes))]
                mode_arrs_f = self.complex_df[
                    [name + "(" + str(i + 1) + ")" for i in range(len(modes))]
                ].values.T * np.sign(self.complex_df[name + "(1)"].values)
                if len(mode_nms_f) == 1:
                    self.complex_df[name + "(1)f"] = mode_arrs_f.flatten()
                else:
                    for x, y in zip(mode_nms_f, mode_arrs_f):
                        self.complex_df[x] = y
        return self.complex_df

    def gen_web_df(self):
        self.web_df = DataFrame(
            [(y, np.round(x, 5)) for y, x in self.simple_df[["name", "coords"]].values],
            columns=["name", "coords"],
        )
        if self.model.ptgr in labels_dict_ext.keys():
            labels = labels_dict_ext.get(self.model.ptgr).keys()
        else:
            labels = labels_dict.get(self.model.ptgr)
        for index, label in enumerate(labels):
            self.web_df[label] = [x[index][1] for x in self.simple_df["scsd"].values]

        nearest_mat = []
        if self.model.ptgr in symm_multiplicity_tables.keys():
            value_adjust = symm_multiplicity_tables.get(self.model.ptgr)[:-1]
            value_adjust[0] = 1
        else:
            value_adjust = 1

        for i, x in enumerate(self.web_df[labels].values):
            nearest_mat.append(
                np.sqrt(
                    np.sum(
                        np.square(self.web_df[labels].values - x) * value_adjust, axis=-1
                    )
                )
            )
        nearest_ind = [np.argsort(x)[:5] for x in nearest_mat]
        nearest_names = [self.web_df.name[x].values.tolist() for x in nearest_ind]
        nearest_vals = [nearest_mat[i][x].tolist() for i, x in enumerate(nearest_ind)]
        self.web_df["nearest"] = [
            {k: v for k, v in zip(ks, vs)}
            for ks, vs in zip(nearest_names, nearest_vals)
        ]
        return self.web_df

    def plot_pca(self, mode=None, scale=1, n_modes=None):
        if mode is None:
            a1 = self.model.ats_3
            fig = plotly_points_joined(
                [a1]
                + [
                    a1 + np.array(v[0]).reshape(shape(a1)) * scale
                    for k, v in self.pca.items()
                ],
                ["model"]
                + [irrep_typog_html.get(k, k) + " (1)" for k, v in self.pca.items()],
                self.model.maxdist,
            )
            fig.layout.scene.camera.projection.type = "orthographic"
            fig.layout.scene.aspectmode = "data"
            return fig
        else:
            pca = np.array(self.pca.get(mode))
            a1, a_n = self.model.ats_3, []

            if n_modes is None:
                n_modes = len(pca)
            for i in range(n_modes):
                a_n.append(
                    self.model.ats_3 + (pca[i].reshape(shape(self.model.ats_3)) * scale)
                )

            fig_style = "3d"
            for index, axis in enumerate(a_n[0].T):
                if norm(axis) < 0.001:
                    fig_style, ax_indices = "2d", [x for x in [0, 1, 2] if x != index]

            if fig_style == "2d":
                a1f = a1[:, ax_indices]
                a_nf = [x[:, ax_indices] for x in a_n]
                fig = plotly_points_joined_2d(
                    [a1f] + a_nf,
                    ["model"]
                    + [
                        irrep_typog_html.get(mode) + "(" + str(i + 1) + ")"
                        for i in range(n_modes)
                    ],
                    self.model.maxdist,
                )
            else:
                fig = plotly_points_joined(
                    [a1] + a_n,
                    ["model"]
                    + [
                        irrep_typog_html.get(mode) + "(" + str(i + 1) + ")"
                        for i in range(n_modes)
                    ],
                    self.model.maxdist,
                )
                fig.layout.scene.camera.projection.type = "orthographic"
                fig.layout.scene.aspectmode = "data"
            return fig

    def pca_kdeplot(
        self, ir, npts=100, fixed=True, as_type="fig", cmap="Spectral", margin=0.5
    ):
        npts = 100
        if ir is None:
            return "None type"
        if (
            fixed
            and (ir + "(1)f" in self.complex_df.columns)
            and (ir + "(2)f" in self.complex_df.columns)
        ):
            ir_labs = [ir + f"({n})f" for n in ["1", "2"]]
            ir_labs_typo = [
                irrep_typog_html.get(ir, ir) + f"({n})f" for n in ["1", "2"]
            ]
            x, y = self.complex_df[ir_labs].values.T
            try:
                kde = gaussian_kde([np.hstack((x, -x)), np.hstack((y, -y))])
            except LinAlgError:
                return "singular matrix on " + ir

        elif (ir + "(1)" in self.complex_df.columns) and (
            ir + "(2)" in self.complex_df.columns
        ):
            ir_labs = [ir + f"({n})" for n in ["1", "2"]]
            ir_labs_typo = [irrep_typog_html.get(ir, ir) + f"({n})" for n in ["1", "2"]]

            x, y = self.complex_df[ir_labs].values.T
            try:
                kde = gaussian_kde([x, y])
            except LinAlgError:
                return "singular matrix on " + ir

        elif ir + "(1)" in self.complex_df.columns:
            return self.pca_histogram(ir, fixed=fixed, as_type=as_type)
        else:
            return "no data"

        xs, ys = np.linspace(min(x) - margin, max(x) + margin, npts), np.linspace(min(y) - margin, max(y) + margin, npts)
        xgrid, ygrid = np.meshgrid(xs, ys)
        zs = [kde((xp, yp)) for xp, yp in zip(xgrid, ygrid)]
        fig = go.Figure(go.Contour(x=xs, y=ys, z=zs, colorscale=cmap))
        fig.add_trace(
            go.Scatter(
                x=self.complex_df[ir_labs[0]],
                y=self.complex_df[ir_labs[1]],
                mode="markers",
                marker={"symbol": "circle-open", "color": "black"},
                hovertext=self.complex_df["name"],
            )
        )
        fig.layout.xaxis.title = ir_labs_typo[0] + " (&#8491;)"
        fig.layout.yaxis.title = ir_labs_typo[1] + " (&#8491;)"
        if as_type == "fig":
            return fig
        if as_type == "html":
            return fig.to_html()

    def pca_histogram(self, ir, fixed=True, as_type="fig"):
        if fixed and (ir + "(1)f" in self.complex_df.columns):
            xlab, xlab_typo = ir + "(1)f", irrep_typog_html.get(ir, ir) + "(1)f"
        elif ir + "(1)" in self.complex_df.columns:
            xlab, xlab_typo = ir + "(1)", irrep_typog_html.get(ir, ir) + "(1)"
        else:
            return ""

        # print(self.complex_df[xlab])
        fig = histogram(
            self.complex_df, x=xlab, hover_data=["name", xlab], marginal="rug"
        )
        fig.layout.xaxis.title = xlab_typo + " (&#8491;)"

        if as_type == "fig":
            return fig
        if as_type == "html":
            return fig.to_html()

    def write_df(self):
        if not hasattr(self, "pca"):
            self.gen_pca()
        if not hasattr(self, "complex_df"):
            self.gen_complex_df()
        self.model.pca = self.pca
        if not hasattr(self, "web_df"):
            self.gen_web_df()
        self.model.database_path = (
            f"{self.model.name}_scsd_{str(date.today()).replace('-','')}.pkl"
        )
        self.web_df.to_pickle(data_path / self.model.database_path)

        user_model_filepath = dirname(__file__) + "/scsd_models_user.py"
        f2 = open(user_model_filepath, "a")
        f2.writelines("\n" + self.model.importable())
        f2.close()
        return self.model.importable()


def pdb_link(s):
    return "<a href = https://www.rcsb.org/structure/{}>{}</a>".format(s, s)
def ccdc_link(s):
    if "_" in s:
        return "<a href = https://www.ccdc.cam.ac.uk/structures/Search?Ccdcid={}&DatabaseToSearch=Published>{}</a>".format(s[s.rfind("_") + 1:], s)
    return "<a href = https://www.ccdc.cam.ac.uk/structures/Search?Ccdcid={}&DatabaseToSearch=Published>{}</a>".format(s, s)
def kb_link(s):
    return "<a href = /scsd/{}>{}</a>".format(s, s)

def make_table_2(names, vals):
    tr = [
        '''<p><caption style="text-align:center"> Table 2; Most similar in precalculated databases </caption> </p>
        <table border="1" id="comparison_table"> <thead> <tr style="text-align: center;"> <th>CCDC link</th> <th>scsd link</th>
        <th>&#8730;&Sigma;(S(1)-S(2))<sup>2</sup></th> </tr> </thead> <tbody>'''
    ]
    [
        tr.append(
            "<tr><td>"
            + ccdc_link(x)
            + "</td><td>"
            + kb_link(x)
            + "</td><td>"
            + str(round(y, 3))
            + " &#8491;</td></tr>"
        )
        for x, y in zip(names, vals)
    ]
    tr.append(
        '</tbody></table><div style="text-align:center">  <button onclick="selectElementContents( document.getElementById(\'comparison_table\') );">Copy Table</button> </div>'
    )
    return "\n".join(tr)


def make_table_2_pdb(names, vals):
    tr = [
        '''<p><caption style="text-align:center"> Table 2; Most similar in precalculated databases </caption> </p>
        <table border="1" id="comparison_table"> <thead> <tr style="text-align: center;">
        <th>PDB link</th> <th>scsd link</th> <th>&#8730;&Sigma;(S(1)-S(2))<sup>2</sup></th> </tr> </thead> <tbody>'''
    ]
    [
        tr.append(
            "<tr><td>"
            + pdb_link(x.split("_")[0])
            + "</td><td>"
            + kb_link(x)
            + "</td><td>"
            + str(round(y, 3))
            + " &#8491;</td></tr>"
        )
        for x, y in zip(names, vals)
    ]
    tr.append(
        '</tbody></table><div style="text-align:center">  <button onclick="selectElementContents( document.getElementById(\'comparison_table\') );">Copy Table</button> </div>'
    )
    return "\n".join(tr)


def find_symmetry_arbitrary(ops, pgt, lookup_table):
    # looks up the tables in symmetry_dicts to correspond to the point group, from knowledge of the irreducible representations
    if (ops == []) or (ops is None):
        ops = [list(pgt.keys())[0]]
    return lookup_table.get(
        str(np.prod([pgt.get(x) for x in ops], axis=0).tolist()), "unknown"
    )


def generate_line_thru_points(ats, maxdist=1.75):
    dists = tril(norm(np.sum([np.meshgrid(a, -a) for a in ats.T], axis=1), axis=0), -1)
    pairs = [[a, b] for a, b in zip(*np.where((0.5 < dists) & (dists < maxdist)))]
    try:
        strop = pairs[0]
    except IndexError:
        strop = []
    for x in range(150):
        pairs, strop = generate_superlist(pairs, strop)
        pairs, strop = generate_superlist([x[::-1] for x in pairs], strop)
    return strop


def generate_line_thru_points_plus_theta(ats, thetas, maxdist=1.75):
    pairs = [
        [
            [i, j]
            for i, x in enumerate(ats)
            if (
                (norm(x - y) < maxdist)
                and (i > j)
                and (abs(thetas[i] - thetas[j]) < 180)
            )
        ]
        for j, y in enumerate(ats)
    ]
    pairs = [y for x in pairs for y in x]
    try:
        strop = pairs[0]
    except IndexError:
        strop = []
    for x in range(100):
        pairs, strop = generate_superlist(pairs, strop)
        pairs, strop = generate_superlist([x[::-1] for x in pairs], strop)
    return strop


def plotly_points_joined(traces, tracenames, maxdist=1.7, **kwargs):
    line_thru = generate_line_thru_points(traces[0], maxdist)
    fig = go.Figure()
    for trace, name in zip(traces, tracenames):
        fig.add_trace(
            go.Scatter3d(
                x=trace.T[0],
                y=trace.T[1],
                z=trace.T[2],
                mode="markers",
                name=name,
                legendgroup=name,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=trace[line_thru].T[0],
                y=trace[line_thru].T[1],
                z=trace[line_thru].T[2],
                mode="lines",
                line=dict(color="black"),
                showlegend=False,
                legendgroup=name,
            )
        )
        fig.layout.scene.camera.projection.type = "orthographic"
        fig.layout.scene.aspectmode = "data"
    return fig


def plotly_points_joined_2d(traces, tracenames, maxdist=1.7, **kwargs):
    line_thru = generate_line_thru_points(traces[0], maxdist)
    fig = go.Figure()
    for trace, name in zip(traces, tracenames):
        fig.add_trace(
            go.Scatter(
                x=trace.T[0],
                y=trace.T[1],
                mode="markers",
                marker=dict(size=15),
                name=name,
                legendgroup=name,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trace[line_thru].T[0],
                y=trace[line_thru].T[1],
                mode="lines",
                line=dict(color="black"),
                showlegend=False,
                legendgroup=name,
                **kwargs,
            )
        )
    return fig


# ex. scsd_direct
# stuff for direct - just on my machine. I'll change it.
# this still runs (or doesn't run) on the old version of scsd (i.e. without classes)


def scsd_direct(
    pdb_path,
    model=None,
    ptgr="",
    cscheme="Spectral",
    output_filepath=False,
    basinhopping=False,
):
    if output_filepath:
        pass
    else:
        output_filepath = pdb_path[:-4] + ".html"
    filename = pdb_path.split("/")[-1]
    ats = import_pdb(pdb_path)
    if isinstance(model, str):
        model = model_objs_dict.get(model)
    if isinstance(model, scsd_model):
        ptgr = model.ptgr
    scsd_obj = scsd_matrix(ats, model, ptgr)
    scsd_obj.calc_scsd(basinhopping)

    app = Flask(
        __name__,
        template_folder=abspath(__file__)[:-8] + "/templates/",
        static_folder="static",
    )
    with app.app_context():
        extras = scsd_obj.compare_table(data_path=data_path) + render_template(
            "/scsd/scsd_hidden_raw_data_section.html", raw_data=scsd_obj.raw_data(), table_ident = 'raw_data'
        )
        template = "/scsd/scsd_html_template_v2.html"
        html = render_template(
            template,
            title=filename[:-4],
            headbox="",
            nsd_table=scsd_obj.html_table(n_modes=2),
            mondrian_fig=scsd_obj.mondrian(as_type="buffer", cmap=cscheme),
            plotly_fig=scsd_obj.scsd_plotly(maxdist=scsd_obj.model.maxdist),
            extras=extras,
        )
    html_file = open(output_filepath, "w")
    html_file.write(html)
    html_file.close()
    return scsd_obj


def gen_headbox(data, ptgr, file1nm=None, file2nm=None, df_name=None):
    ptgr2 = symm_typog_html.get(ptgr, ptgr)
    if isinstance(file2nm, str):
        # don't think this is used any more
        return f"SCSD generated from {file1nm} (query) and {file2nm} (model) using {ptgr2} symmetry"
    if isinstance(file1nm, str):
        if data.get("model_name") == "":
            return f"SCSD generated from {file1nm} using {ptgr2} symmetry and no model"
        else:
            return f"SCSD generated from {file1nm} using {ptgr2} symmetry and <a href = '/scsd_model/{data.get('model_name')}'>{data.get('model_name')}</a> pregenerated model"
    else:
        return f"SCSD for refcode {data.get('refcode')} retrieved from the <a href = '/scsd_model/{df_name}'>{df_name}</a> dataset using {ptgr2} symmetry"

def assign_mat_for_symm(model, typo, theta, phi, rotation):
    omega = (2 * pi) / rotation
    refvec = s2c(theta, phi)
    rot_mat = q2r(*s2q(theta, phi, omega))
    trans_atoms = model.copy()[:, :3]

    if typo in ["improperrotation", "mirror"]:
        trans_atoms = trans_atoms - np.outer(2 * np.dot(trans_atoms, refvec), refvec)
    if typo in ["rotation", "improperrotation"]:
        trans_atoms = np.dot(rot_mat, trans_atoms.T).T
    if typo == "inversion":
        trans_atoms = -model[:, :3]

    costmat = costmat_gen(model[:, :3], trans_atoms)
    row_ind, col_ind = linear_sum_assignment(costmat)
    return col_ind.argsort().tolist()


def return_assign_mat(model, point_group_name):
    ops, pgt, ops_order = [
        x.get(point_group_name.capitalize())
        for x in [point_group_dict, pgt_dict, ordered_ops_dict]
    ]
    ops_vals = [operations_dict.get(x) for x in ops_order]
    return {
        z: [assign_mat_for_symm(model, *x) for x in y]
        for z, y in zip(ops_order, ops_vals)
    }


def generate_superlist(pairs, strop):
    for i, pair in enumerate(pairs):
        for j, at in enumerate(strop):
            if at == pair[0]:
                strop.insert(j, pair[0])
                strop.insert(j + 1, pair[1])
                pairs.pop(i)
                return pairs, strop
    return pairs, strop


class scsd_model:
    def __init__(
        self,
        name,
        atom_positions,
        point_group_name,
        assign_mat=None,
        pca=None,
        database_path=None,
        maxdist=1.75,
        mondrian_limits=None,
        smarts=None,
    ):
        self.name = name
        self.ats = atom_positions
        self.ats_3 = self.ats[:, :3].astype(float)
        self.ats_3_round_4 = np.round(self.ats[:, :3].astype(float), 4)
        if len(self.ats[0]) == 4:
            self.ats_round_4 = np.hstack(
                (self.ats_3_round_4, np.atleast_2d(self.ats[:, 3]).T)
            )
        else:
            self.ats_round_4 = self.ats_3_round_4
        self.ptgr = point_group_name.capitalize()
        self.symm = scsd_symmetry(self.ptgr)
        self.pca = pca
        self.assign_mat = assign_mat
        # self.database_path = database_path
        avail_databases = self.database_path = sorted([x for x in os.listdir(data_path) if x.startswith(self.name)])
        if len(avail_databases) > 0:
            self.database_path = avail_databases[-1]
        else:
            self.database_path = database_path
        self.maxdist = maxdist
        self.mondrian_limits = mondrian_limits
        self.ptgr_html = symm_typog_html.get(self.ptgr, self.ptgr)
        self.smarts = smarts

    def gen_assign_mat(self):
        self.assign_mat = return_assign_mat(self.ats_3, self.ptgr)

    def __repr__(self):
        not_Nones = [
            x
            for x, y in zip(
                "name,ats,ptgr,pca,assign_mat,database_path,maxdist,mondrian_limits,smarts".split(
                    ","
                ),
                [
                    self.name,
                    self.ats,
                    self.ptgr,
                    self.pca,
                    self.assign_mat,
                    self.database_path,
                    self.maxdist,
                    self.mondrian_limits,
                    self.smarts,
                ],
            )
            if y is not None
        ]
        return "scsd_model: " + self.name + "; attributes: " + ", ".join(not_Nones)

    def importable(self):
        l1 = [
            f"""model_objs_dict['{self.name}'] = scsd_model('{self.name}', array({str(self.ats_round_4.tolist())}), '{self.ptgr}' """
        ]
        if type(self.pca) is not type(None):
            l1.append(
                ", pca = {"
                + ", ".join(
                    [
                        f'"{k}":array({str(v.round(5).tolist())})'
                        for k, v in self.pca.items()
                    ]
                )
                + "}"
            )
        if type(self.database_path) is not type(None):
            l1.append(f", database_path = '{self.database_path}'")
        if type(self.maxdist) is not type(None):
            l1.append(f", maxdist = {self.maxdist}")
        if type(self.mondrian_limits) is not type(None):
            l1.append(f", mondrian_limits = {self.mondrian_limits}")
        if type(self.smarts) is not type(None):
            l1.append(f", smarts = '{self.smarts}'")
        l1.append(")")
        return "".join(l1)

    def to_dict(self):
        if self.pca is None:
            pca = None
        else:
            pca = {k: v.tolist() for k, v in self.pca.items()}
        return {
            "name": self.name,
            "ats": self.ats.tolist(),
            "ptgr": self.ptgr,
            "assign_mat": self.assign_mat,
            "mondrian_limits": self.mondrian_limits,
            "maxdist": self.maxdist,
            "pca": pca,
            "database_path": self.database_path,
            "smarts": self.smarts,
        }

        # if type(self.pca) is type(None):
        #    return "model_objs_dict['{}'] = scsd_model('{}', array({}), '{}')\n".format(self.name, self.name, str(self.ats_round_4.tolist()),self.ptgr)
        # else:
        #    return f'''model_objs_dict['{self.name}'] = scsd_model('{self.name}', array({str(self.ats_round_4.tolist())}), '{self.ptgr}',
        # pca = {'{'+', '.join([f'"{k}":array({str(v.round(5).tolist())})' for k,v in self.pca.items()])+'}' if type(self.pca) is not None else 'None'},
        # database_path = {"'"+self.database_path+"'" if type(self.database_path) == str else 'None'}, maxdist = {self.maxdist if type(self.maxdist) in [int,float] else 'None'},
        # mondrian_limits = {self.mondrian_limits if type(self.mondrian_limits) is not None else 'None'} )'''

    #
    #  maxdist = {self.maxdist if type(self.maxdist) in [int,float] else 'None'}, mondrian_limits = {self.mondrian_limits if type(self.mondrian_limits) is not None else 'None'} )'''

    def html_table(self):
        return (
            "<table border='1' id = 'html_table'>"
            + "\n".join(
                [
                    "<tr>{}</tr>".format(
                        "".join(["<td>{}</td>".format(value) for value in row])
                    )
                    for row in np.vstack(
                        (
                            np.atleast_2d(
                                ["x", "y", "z", "at"][: len(self.ats_round_4[0])]
                            ),
                            self.ats_round_4.tolist(),
                        )
                    )
                ]
            )
            + "</table>"
        )

    def plotly_plot_model(self, conformation = None):
        line_thru = generate_line_thru_points(self.ats_3, self.maxdist)

        if len(self.ats.T) == 3:
            if isinstance(conformation, np.ndarray):
                a = conformation[line_thru].T
                b = conformation.T
            else:   
                a = self.ats_3[line_thru].T
                b = self.ats_3.T
            fig = go.Figure()
            fig.add_trace(
                go.Scatter3d(
                    x=b[0],
                    y=b[1],
                    z=b[2],
                    mode="markers",
                    name="Model",
                    legendgroup="Model",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=a[0],
                    y=a[1],
                    z=a[2],
                    mode="lines",
                    showlegend=False,
                    legendgroup="Model",
                    line=dict(color="black"),
                )
            )
        if len(self.ats.T) == 4:
            if isinstance(conformation, np.ndarray):
                a, b = conformation[line_thru].T, conformation.T
            else:   
                a, b = self.ats_3[line_thru].T, self.ats.T
            fig = go.Figure()
            for index, atype in enumerate(np.unique(b[3])):
                d = b[:3, np.where(b[3] == atype)[0]].astype(float)
                if atype in atoms_color_dict.keys():
                    fig.add_trace(
                        go.Scatter3d(
                            x=d[0],
                            y=d[1],
                            z=d[2],
                            mode="markers",
                            marker={"color": atoms_color_dict.get(atype, 'pink')},
                            name=atype,
                            legendgroup="Model",
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter3d(
                            x=d[0],
                            y=d[1],
                            z=d[2],
                            mode="markers",
                            name=atype,
                            legendgroup="Model",
                        )
                    )
            fig.add_trace(
                go.Scatter3d(
                    x=a[0],
                    y=a[1],
                    z=a[2],
                    mode="lines",
                    showlegend=False,
                    legendgroup="Model",
                    line=dict(color="black"),
                )
            )
        fig.layout.scene.camera.projection = dict(type="orthographic")
        fig.layout.scene.aspectmode = "data"

        return fig

    def visualize_symm_ops(self, as_type="html", symm_ops=None):  #
        self.symm = scsd_symmetry(self.ptgr)
        ops_verb = [y for x in self.symm.ops_verb_order for y in x]
        if isinstance(symm_ops, str):
            ops_verb = [
                y
                for x in [
                    z
                    for z, a in zip(
                        self.symm.ops_verb_order, self.symm.pgt.get(symm_ops)
                    )
                    if (a == 1)
                ]
                for y in x
            ]
        if isinstance(symm_ops, list):
            if symm_ops[0] == 1:
                ops_temp = [
                    z
                    for i, z in enumerate(self.symm.ops_verb_order)
                    if symm_ops[i] == 1
                ]
            else:
                ops_temp = [
                    z
                    for i, z in enumerate(self.symm.ops_verb_order)
                    if all([self.symm.pgt.get(b)[i] == 1 for b in symm_ops])
                ]
            ops_verb = [y for x in ops_temp for y in x]

        fig = self.plotly_plot_model()
        scale = max(self.ats_3.flatten())

        for op in ops_verb[1:]:
            typo, th, ph, n_rot = op
            if typo.lower() == "rotation":
                x, y, z = s2c(th, ph) * scale * 1.2
                fig.add_trace(
                    go.Scatter3d(
                        x=[-x, x],
                        y=[-y, y],
                        z=[-z, z],
                        marker=dict(symbol="circle-open"),
                        name=str(n_rot) + "-fold rotation",
                    )
                )
            if typo.lower() == "improperrotation":
                x, y, z = s2c(th, ph) * scale * 1.1
                fig.add_trace(
                    go.Scatter3d(
                        x=[-x, x],
                        y=[-y, y],
                        z=[-z, z],
                        marker=dict(symbol="circle-open"),
                        name=str(n_rot) + "-fold improper rotation",
                    )
                )
            if typo.lower() == "mirror":
                x1, y1, z1 = s2c(th, ph + (pi / 2)) * scale
                x2, y2, z2 = np.cross(s2c(th, ph + pi / 2), s2c(th, ph)) * scale
                x3, y3, z3 = x1 + x2, y1 + y2, z1 + z2
                x4, y4, z4 = x1 - x2, y1 - y2, z1 - z2
                x, y, z = (
                    [x3, x4, -x3, -x4, x3],
                    [y3, y4, -y3, -y4, y3],
                    [z3, z4, -z3, -z4, z3],
                )
                sax = np.argmin([norm(v) for v in [x, y, z]])
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        name="Mirror",
                        mode="lines",
                        surfaceaxis=sax,
                        surfacecolor="rgba(150,50,50,0.2)",
                    )
                )
            if typo.lower() == "inversion":
                x, y, z = [0], [0], [0]
                fig.add_trace(go.Scatter3d(x=x, y=y, z=z, name="Inversion"))
        fig.layout.scene.camera.projection = dict(type="orthographic")
        fig.layout.scene.aspectmode = "data"
        if as_type.lower() == "html":
            return to_html(fig)
        elif as_type.lower() == "fig":
            return fig

    def visualize_principal_component(
        self, irrep, mode_number, magnitude=1.0, as_type="fig", color="black"
    ):  # , symm_ops = None
        traces = []

        a0 = self.ats_3
        a1 = self.ats_3 + (
            np.array(self.pca.get(irrep)[mode_number - 1])
            .astype(float)
            .reshape(shape(self.ats_3))
            * magnitude
        )
        for xyz0, xyz1 in zip(a0, a1):
            x0, y0, z0 = xyz0
            x1, y1, z1 = xyz1
            traces.append(
                go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode="lines",
                    line={"color": color},
                    legendgroup=irrep + "(" + str(mode_number) + ")",
                    name=irrep + "(" + str(mode_number) + ")",
                )
            )
        x, y, z = a1.T
        u, v, w = (a1 - a0).T / 5
        traces.append(
            go.Cone(
                x=x,
                y=y,
                z=z,
                u=u,
                v=v,
                w=w,
                colorscale=[color, color],
                showscale=False,
                legendgroup=irrep + "(" + str(mode_number) + ")",
            )
        )
        if as_type == "traces":
            return traces
        fig = self.plotly_plot_model()
        [fig.add_traces(x) for x in traces]
        fig.layout.scene.camera.projection = dict(type="orthographic")
        fig.layout.scene.aspectmode = "data"
        fig.update_layout(showlegend=False)

        if as_type.lower() == "fig":
            return fig
        elif as_type.lower() == "html":
            return to_html(fig)

    def visualize_symm_and_pc(self, as_type="html", symm_ops=None):  #
        fig = self.visualize_symm_ops(as_type="fig")

        if type(self.pca) is not type(None):
            scale_pca = np.floor((max(self.ats_3.flatten()) ** 2) / 4)
            line_thru = generate_line_thru_points(self.ats_3, self.maxdist)

            for k, v in self.pca.items():
                if len(v) > 0:
                    ats = (
                        self.ats_3
                        + np.array(v[0]).reshape(shape(self.ats_3)) * scale_pca
                    )
                    x, y, z = ats[line_thru].T
                    tracename = irrep_typog_html.get(k, k) + " (1)"
                    fig.add_trace(
                        go.Scatter3d(
                            x=x, y=y, z=z, name=tracename, visible="legendonly"
                        )
                    )
                if len(v) > 1:
                    ats = (
                        self.ats_3
                        + np.array(v[1]).reshape(shape(self.ats_3)) * scale_pca
                    )
                    x, y, z = ats[line_thru].T
                    tracename = irrep_typog_html.get(k, k) + " (2)"
                    fig.add_trace(
                        go.Scatter3d(
                            x=x, y=y, z=z, name=tracename, visible="legendonly"
                        )
                    )

        fig.layout.scene.camera.projection = dict(type="orthographic")
        fig.layout.scene.aspectmode = "data"
        if as_type.lower() == "html":
            return to_html(fig)
        elif as_type.lower() == "fig":
            return fig

    def headbox(self, filename):
        return "SCSD model generated from {} using {} symmetry, available as <i>{}</i>".format(
            filename, symm_typog_html.get(self.ptgr, self.ptgr), self.name
        )

    def chem_formula(self):
        if len(self.ats[0]) == 4:
            uq = np.unique(self.ats[:, 3], return_counts=True)
            return "".join([f"{a}<sub>{str(b)}</sub>" for a, b in zip(*uq)])
        else:
            return ""
        
    def generate_conformation(self, mode_magnitudes={}):
        conformation = np.array(self.ats_3).astype(float)
        for key, values in mode_magnitudes.items():
            for ix, mag in enumerate(list(values)):
                conformation = conformation + (self.pca.get(key,0)[ix] * mag).reshape(np.shape(conformation))
        if len(self.ats[0]) == 4:
            return np.hstack((conformation, self.ats[:,3:]))
        else:
            return conformation

    def visualize_conformation(self, mode_magnitudes={}): 
        conformation = self.generate_conformation(mode_magnitudes)
        return self.plotly_plot_model(conformation=conformation)


from pathlib import Path
import os
import json


def scsd_model_from_dict(source):
    if source.get("pca", None) is not None:
        pca = {k: np.array(v) for k, v in source.get("pca").items()}
    else:
        pca = None

    model = scsd_model(
        name=source.get("name", None),
        atom_positions=np.array(source.get("ats", None)),
        point_group_name=source.get("ptgr", None),
        pca=pca,
        maxdist=source.get("maxdist", None),
        mondrian_limits=source.get("mondrian_limits", None),
        database_path=source.get("database_path", None),
        smarts=source.get("smarts", None),
        assign_mat=source.get("assign_mat", None),
    )
    return model


fpath = Path(os.path.dirname(os.path.realpath(__file__)))
with open(fpath / "./scsd_models.json", "r") as models_json:
    model_objs_dict = {
        k: scsd_model_from_dict(v) for k, v in json.load(models_json).items()
    }
