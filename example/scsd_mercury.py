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
from numpy import array
from ccdc.utilities import ApplicationInterface
import ccdc.search

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

default_settings = {
    "cscheme": "random",
    "ptgr": "C2v",
    "basinhopping": True,
    "by_graph": False,
    "dfs_path": "",
    "break_after_hits": True,
}


def smarts_count(x):
    from re import split as resplit

    if x.startswith("["):
        return int(
            sum([1 if y.isalpha() else 0
                 for y in "".join(resplit(r"\[|\]", x)[0::2])])
            + x.count("[")
        )
    elif x.count("[") > 0:
        return int(
            sum([1 if y.isalpha() else 0
                 for y in "".join(resplit(r"\[|\]", x)[1::2])])
            + x.count("[")
        )
    else:
        return int(sum([1 for y in x if y.isalpha()]))


def hit_to_html(hit, model, settings=default_settings):
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.abspath(os.path.dirname(scsd.__file__)), "templates"),
        static_folder="static",
    )

    ats = array(
        [
            array([*x.coordinates, x.atomic_symbol])
            for x in hit.centroid_atoms("CENT1")
        ]
    )
    scsd_obj = scsd.scsd_matrix(ats, model, model.ptgr)
    scsd_obj.calc_scsd(
        settings.get("basinhopping", False),
        by_graph=settings.get("by_graph", True),
    )

    with app.app_context():
        extras = "\n".join(
            (
                scsd_obj.compare_table(data_path=settings.get("dfs_path", "")),
                render_template(
                    "/scsd/scsd_hidden_raw_data_section.html",
                    raw_data=scsd_obj.raw_data(),
                ),
            )
        )
        template = "/scsd/scsd_html_template_v2.html"
        html = render_template(
            template,
            title="",
            headbox=model.headbox(''),
            nsd_table=scsd_obj.html_table(n_modes=2),
            mondrian_fig=scsd_obj.mondrian(
                as_type="buffer", cmap=settings.get("cscheme", "random")
            ),
            plotly_fig=scsd_obj.scsd_plotly(maxdist=scsd_obj.model.maxdist, as_type='html_min'),
            extras=extras,
        )
    return html

model_heirarchy = {'tetrabenzopentacene':['naphthalene', 'anthracene', 'tetracene', 'pentacene'],
                   'coronene':['naphthalene', 'anthracene', 'phenanthrene'],
                   'pentacene':['naphthalene', 'anthracene', 'tetracene'],
                   'tetracene':['naphthalene', 'anthracene'],
                   'anthracene':['naphthalene'],
                   'pyrene':['naphthalene', 'phenanthrene'],
                   'teropyrene':['pyrene','naphthalene', 'phenanthrene'],
                   'porphyrin':['dipyrrin'],
                   "carbamazepine":['dibenzazepine'],
                   }

def scsd_mercury(settings=default_settings):
    interface = ApplicationInterface(parse_commandline=False)
    interface.parse_commandline()
    interface.update_progress(str(settings))
    entry = interface.current_entry
    crystal = entry.crystal
    title = f"scsd for {interface.identifier}"
    skip_models = []

    #    molecule = crystal.molecule
    htmls = []
    scsd.model_objs_dict.update({k:v for k,v in scsd_models_user.model_objs_dict.items() if k not in scsd.model_objs_dict})

    for name, model in scsd.model_objs_dict.items():
        print(name)
        if (model is None) or (model.smarts is None) or (name in skip_models):
            continue

        if name in model_heirarchy.keys():
            [skip_models.append(x) for x in model_heirarchy[name]]

        substructure_search = ccdc.search.SubstructureSearch()
        substructure_search.add_substructure(ccdc.search.SMARTSSubstructure(model.smarts))
        substructure_search.add_centroid("CENT1", *((0, x) for x in range(smarts_count(model.smarts))))
        hits = substructure_search.search(database=crystal, max_hits_per_structure = 20)

        interface.update_progress("trying " + model.name)

        if len(hits) > 0:
            interface.update_progress("found " + model.name)
            for hit in hits:
                htmls.append(hit_to_html(hit, model))

    with interface.html_report(title=title) as report:
        if len(htmls) > 0:
            report.write(htmls)
        else:
            report.write("no structures found")

if __name__ == "__main__":
    scsd_mercury()
