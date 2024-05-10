#
# This script can be used for any purpose without limitation subject to the
# conditions at http://www.ccdc.cam.ac.uk/Community/Pages/Licences/v2.aspx
#
# This permission notice and the following statement of attribution must be
# included in all copies or substantial portions of this script.
#
# 2023-04-25 Created by Chris Kingsbury, the Cambridge Crystallographic Data Centre
# ORCID 0000-0002-4694-5566
#
# scsd - Symmetry-Coordinate Structural Decomposition for molecules
# written by Dr. Christopher J. Kingsbury, Trinity College Dublin, with Prof. Dr. Mathias O. Senge
# cjkingsbury@gmail.com / www.kingsbury.id.au
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
#
#

from os.path import abspath
from numpy import array
import argparse
from ccdc.io import MoleculeReader

from pandas import DataFrame
from scsd.scsd import scsd_matrix
from scsd.scsd import model_objs_dict, scsd_model

def run_scsd_single_cl():
    parser = argparse.ArgumentParser()
    parser.add_argument('idents', type=str, nargs = '+',
                        help = 'the files/refcodes that will be analysed')
    parser.add_argument('-m','--model', type = str, default = '',
                        help = 'The name of a model in scsd_models.py')
    parser.add_argument('-p','--ptgr', type = str, default = '',
                        help = 'idealised point group (e.g. "C2v"), inherited from model')
    parser.add_argument('-b','--basinhopping', action = 'store_true',
                        help = 'add basinhopping algorithm to calculation (avoids local minima, increases time)')
    parser.add_argument('-g','--by_graph', action = 'store_true',
                        help = 'perform calculation using networkx.algorithms.isomorphism.GraphMatcher (faster)')
    parser.add_argument('-o','--output', type = str,
                        help = 'output location / filetype (csv, pkl, html)')

    args = parser.parse_args()

    output_dicts = []

    if ('.' not in ''.join(args.idents)):
        csd_reader = MoleculeReader('CSD')

    for ident in args.idents:
        if ('.' in ident):
            source = MoleculeReader(ident)
        else:
            source = csd_reader.molecule(ident.upper())
            source.assign_bond_types(which='unknown')

        #dfs_path = settings.get('dfs_path', '')

        ats = array([[*x.coordinates,x.atomic_symbol] for x in source.atoms])
        model = model_objs_dict.get(args.model,None)

        if isinstance(model, scsd_model):
            ptgr = model.ptgr
        else:
            ptgr = args.ptgr

        scsd_obj = scsd_matrix(ats, model, ptgr)
        scsd_obj.calc_scsd(args.basinhopping, by_graph = args.by_graph)
        output = scsd_obj.simple_scsd()
        output_dicts.append(output)

    df = DataFrame(output_dicts)

    if args.output:
        if args.output.endswith('.csv'):
            df.to_csv(args.output)
        elif args.output.endswith('.html'):
            df.to_html(args.output)
        elif args.output.endswith('.pkl'):
            df.to_pickle(args.output)
    else:
        print(df)

if __name__ == '__main__':
    run_scsd_single_cl()
