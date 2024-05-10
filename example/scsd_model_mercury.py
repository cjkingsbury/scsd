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


from flask import Flask, render_template
import os
from pathlib import Path
from numpy import array, mean
from ccdc.utilities import ApplicationInterface
import ccdc.search

from scsd import make_smarts
from datetime import date
from scsd import scsd
from scsd import scsd_models_user
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULT_SETTINGS = {
    "basinhopping": True,
    "by_graph": False,
    "use_smarts": True,
    "return_model": False,
}


def smarts_count(x):
    from re import split as resplit

    if x.startswith("["):
        return int(
            sum([1 if y.isalpha() else 0 for y in "".join(resplit(r"\[|\]", x)[0::2])])
            + x.count("[")
        )
    elif x.count("[") > 0:
        return int(
            sum([1 if y.isalpha() else 0 for y in "".join(resplit(r"\[|\]", x)[1::2])])
            + x.count("[")
        )
    else:
        return int(sum([1 for y in x if y.isalpha()]))


def scsd_model_mercury(
    settings=DEFAULT_SETTINGS, interface=ApplicationInterface(parse_commandline=False)
):
    interface.parse_commandline()

    entry = interface.current_entry
    crystal = entry.crystal

    molecule = crystal.molecule
    molecule.assign_bond_types(which="unknown")

    if len(interface.selected_atoms) == 0:
        with interface.html_report(title="FAILURE") as report:
            report.write("Select more than zero atoms, idiot")
        return None

    ats_labels = [x.label for x in interface.selected_atoms]
    frag = molecule.copy()
    not_in_frag = [x for x in frag.atoms if (x.label not in ats_labels) or x.is_metal]
    frag.remove_atoms(not_in_frag)
    frag.add_hydrogens("all")
    smiles = frag.smiles.split(".")[0]

    symmetry = ccdc.descriptors.MolecularDescriptors.point_group_analysis(frag)[1]
    if symmetry not in ("C2v", "C2h", "D2h", "D4h"):
        with interface.html_report(title="FAILURE 2") as report:
            report.write(f"Symmetry identified as {symmetry}, not implemented")
        return None

    if settings.get("use_smarts"):
        smarts = make_smarts.make_smarter(smiles)
    else:
        smarts = smiles

    interface.update_progress("found SMARTS")

    substructure_search = ccdc.search.SubstructureSearch()
    substructure_search.add_substructure(ccdc.search.SMARTSSubstructure(smarts))
    substructure_search.add_centroid(
        "CENT1", *((0, x) for x in range(smarts_count(smarts)))
    )
    hits = substructure_search.search(database=crystal, max_hits_per_structure=20)
    if len(hits) == 0:
        scsd_model_mercury(settings=settings.update({"use_smarts": False}))
    centroid = mean([y.coordinates for y in hits[0].centroid_atoms("CENT1")], axis=0)
    interface.update_progress(f"found fragment, centroid = {centroid}")

    model_ats_in = array(
        [
            [*x.coordinates - centroid, x.atomic_symbol]
            for x in hits[0].centroid_atoms("CENT1")
        ]
    )

    model_name = interface.identifier
    model_ats = scsd.yield_model(
        model_ats_in,
        symmetry,
        bhopping=settings.get("basinhopping", True),
        by_graph=settings.get("by_graph", False),
    )

    interface.update_progress("Model identified, writing...")

    model = scsd.scsd_model(
        model_name,
        model_ats,
        symmetry,
        mondrian_limits=[-1 if len(model_ats) < 30 else 0] * 2,
        smarts=smarts,
    )
    if settings["return_model"]:
        return model

    tstamp = str(date.today()).replace("-", "")
    for_mod_usr = ["\n#" + model_name + " " + tstamp]
    for_mod_usr.append(model.importable())
    user_model_filepath = scsd.data_path / "scsd_models_user.py"

    with open(user_model_filepath, "a") as f2:
        f2.writelines("\n".join(for_mod_usr))

    app = Flask(
        __name__,
        template_folder=Path(os.path.dirname(os.path.realpath(__file__)))
        / "./templates/",
        static_folder="static",
    )
    with app.app_context():
        extras = render_template(
            "/scsd/scsd_hidden_raw_data_section.html", raw_data="\n".join(for_mod_usr)
        )

        html = render_template(
            "/scsd/scsd_model_report.html",
            title="",
            headbox=model.headbox(model_name),
            html_table=model.html_table(),
            plotly_fig=model.visualize_symm_ops(),
            extras=extras,
        )
    with interface.html_report(title=f"scsd model for {model_name}") as report:
        report.write(html)

    return None


if __name__ == "__main__":
    scsd_model_mercury()
