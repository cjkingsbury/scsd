#
# This script can be used for any purpose without limitation subject to the
# conditions at http://www.ccdc.cam.ac.uk/Community/Pages/Licences/v2.aspx
#
# This permission notice and the following statement of attribution must be
# included in all copies or substantial portions of this script.
#
# 2023-04-25 Created by Chris Kingsbury,
# Cambridge Crystallographic Data Centre
# ORCID 0000-0002-4694-5566
#
# scsd analysis for the dissymmetric distortion of chemical systems.
#
#
from scsd import scsd
from scsd import scsd_models_user

from flask import Flask, render_template
import os
from pathlib import Path
from numpy import array, mean
from ccdc.utilities import ApplicationInterface
import ccdc.search

import make_smarts
from datetime import date

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULT_MODEL_SETTINGS = {
    "basinhopping": True,
    "by_graph": False,
    "use_smarts":True,
    "return_model":True,
}
DEFAULT_COLLECTION_SETTINGS = {
    "basinhopping": True,
    "by_graph": False,
    "exclude_structures": None,
    "pca_limit": 2
}

from scsd import scsd_model_mercury

def collection_from_ccdc(model, settings = DEFAULT_COLLECTION_SETTINGS, interface = ApplicationInterface(parse_commandline=False)):
    interface = ApplicationInterface(parse_commandline=False)
    coll = scsd.scsd_collection(model)
    print(coll.model.smarts)

    substructure_search = ccdc.search.SubstructureSearch()
    substructure_search.add_substructure(
        ccdc.search.SMARTSSubstructure(model.smarts))
    substructure_search.add_centroid('CENT1', *[(0,x) for x in range(scsd_model_mercury.smarts_count(model.smarts))])
    hits = substructure_search.search(database = "CSD")
    hits_filt = [hit for hit in hits if len(hit.centroid_atoms('CENT1')) > 1]

    hitlist = []
    for ix,hit in enumerate(hits_filt):
        try:
            hitlist.append([hit.identifier, [[*atom.coordinates, atom.atomic_symbol] for atom in hit.centroid_atoms('CENT1')]])
        except IndexError:
            pass

    coll.gen_simple_df(hitlist, by_graph = settings["by_graph"], bhop = settings["basinhopping"], verbose = True)
    coll.simple_df
    coll.gen_pca(settings["pca_limit"])
    coll.model.pca = coll.pca
    coll.gen_complex_df()
    coll.write_df()
    return coll


def scsd_collection_mercury(settings = DEFAULT_COLLECTION_SETTINGS):

    interface = ApplicationInterface(parse_commandline=False)

    model = scsd_model_mercury.scsd_model_mercury(DEFAULT_MODEL_SETTINGS, interface = interface)

    interface.update_progress("Calculating collection...")
    coll = collection_from_ccdc(model, settings, interface = interface)

    tstamp = str(date.today()).replace("-", "")
    df = coll.complex_df
    link = "<a href = '/scsd/{x}'>{x}</a>"
    irs = coll.model.pca.keys()

    app = Flask(
        __name__,
        template_folder=Path(os.path.dirname(os.path.realpath(__file__)))
        / "./templates/",
        static_folder="static",
    )
    with app.app_context():
        extras = ''.join([render_template("/scsd/scsd_hidden_raw_data_section.html",
                                          raw_data="\n".join(["#" + model.name + " " + tstamp, model.importable()]),
                                            table_ident='raw_data'),
                         "<br>", ", ".join([link.format(x=refcode) for refcode in df["name"].values]),
                          "<br>", ", ".join([coll.pca_kdeplot(x, as_type="html") for x in irs])])

        html = render_template(
            "/scsd/scsd_model_report.html",
            title=model.name,
            headbox=model.headbox(f"<a href = '/scsd_data/{model.name}'>{model.name}</a>"),
            html_table=model.html_table(),
            plotly_fig=model.visualize_symm_and_pc(),
            extras=extras,
        )

    interface = ApplicationInterface(parse_commandline=False)

    with interface.html_report(title=f'scsd model for {model.name}') as report:
        report.write(html)

    return None


if __name__ == "__main__":
    scsd_collection_mercury()
