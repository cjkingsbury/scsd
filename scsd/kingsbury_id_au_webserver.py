# This work is licensed under THE ANTI-CAPITALIST SOFTWARE LICENSE (v 1.4) 
# To view a copy of this license, visit https://directory.fsf.org/wiki/License:ANTI-1.4

from os import getcwd
import os
from time import time, ctime
from pandas import read_pickle
from datetime import date
from importlib import reload
from flask import (
    Flask,
    render_template,
    request,
    url_for,
    send_from_directory,
    redirect
)

from numpy import random, sqrt, unique

from . import scsd
from . import scsd_models_user
from .nsd import nsd_obj, write_logfile

try:
    from flask_weasyprint import HTML, render_pdf
except ImportError:
    print("flask_weasyprint not available - pdf output disabled")

lib_folder = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(lib_folder, "templates"),
    static_folder="static",
    static_url_path="",
)

from pathlib import Path

u_folder = os.path.join(lib_folder, "data", "temp")
dfs_path = os.path.join(lib_folder, "data", "scsd")
dfs_path += r"/"
u_folder += r"/"


@app.route("/nsd", methods=["GET"])
def upload_file():
    return render_template(
        "/nsd/nsd_uploader.html",
        nsdlogopath=url_for("static", filename="nsdlogo.png"),
        sengelogopath=url_for("static", filename="sengelogo.jpg"),
    )


@app.route("/nsd_output", methods=["GET", "POST"])
def uploaded_file():
    if request.method == "POST":
        f = request.files["file"]
        data = request.form
        tstamp = str(int(time()))
        f.save(u_folder + f.filename)

        if data.get("IUPAC_numbering"):
            nsd_object = nsd_obj(
                filenm=u_folder + f.filename, calctype="correct_numbering"
            )
        elif data.get("errorbars"):
            nsd_object = nsd_obj(filenm=u_folder + f.filename, calctype="errorbars")
        else:
            nsd_object = nsd_obj(filenm=u_folder + f.filename, calctype="pdb")
        nsd_object.calc_nsd()

        write_logfile(f.filename, nsd_object.nsd_matrix, u_folder[:-5] + "logfile.txt")

        html = render_template(
            "/nsd/nsd_html_template_v5.html",
            pdbpath=f.filename,
            time=ctime(time()),
            nsd=nsd_object.nsd_matrix,
            skeletal_fig=nsd_object.nsd_fig(),
            extras=nsd_object.extras(
                data["validtype"],
                data.get("rounding"),
                data.get("keith"),
                cmap=data.get("cmap"),
            ),
        )

        if data["outtype"] == "html":
            return html
        elif data["outtype"] == "pdf":
            return render_pdf(
                HTML(string=html), download_filename=f.filename[:-4] + "_nsd.pdf"
            )
            # return redirect(url_for('nsd'))


@app.route("/nsd_ccdc", methods=["GET"])
def upload_file_ccdc():
    return render_template("/nsd/nsd_ccdc_uploader.html")


fail_html = "Structure not in database (v2020.1) check structure code"


@app.route("/nsd_output_ccdc", methods=["GET", "POST"])
def uploaded_file_ccdc():
    if request.method == "POST":
        data = request.form
        tstamp = str(int(time()))
        try:
            nsd_object = nsd_obj(refcode=data.get("NAME_str").upper(), calctype="ccdc")
            nsd_object.calc_nsd()
        except IndexError:
            return fail_html

        html = render_template(
            "/nsd/nsd_html_template_v5.html",
            pdbpath=data.get("NAME_str").upper(),
            time=ctime(time()),
            nsd=nsd_object.nsd_matrix,
            skeletal_fig=nsd_object.nsd_fig(),
            extras=nsd_object.extras(
                data["validtype"],
                data.get("rounding"),
                data.get("keith"),
                cmap=data.get("cmap"),
            ),
        )
        if data["outtype"] == "html":
            return html
        elif data["outtype"] == "pdf":
            return render_pdf(
                HTML(string=html),
                download_filename=data.get("NAME_str").upper() + "_nsd.pdf",
            )
            # return redirect(url_for('nsd'))


@app.route("/nsd/<refcode>", methods=["GET", "POST"])
def database_direct_ccdc(refcode):
    try:
        nsd_object = nsd_obj(refcode=refcode.upper(), calctype="ccdc")
        nsd_object.calc_nsd()
    except IndexError:
        return fail_html

    extras = "\n".join(
        [
            nsd_object.alert_section(),
            render_template(
                "/nsd/nsd_bdba_table_section.html",
                vals=nsd_object.bdba_extract_array(True, False),
            ),
            render_template(
                "/nsd/nsd_verbose_table_section.html", t=nsd_object.verbose_output()
            ),
            render_template(
                "/nsd/nsd_mondrian_section.html",
                mondrian_fig=nsd_object.mondrian(cmap="random"),
            ),
        ]
    )

    html = render_template(
        "/nsd/nsd_html_template_v5.html",
        pdbpath=refcode.upper(),
        time=ctime(time()),
        nsd=nsd_object.nsd_matrix,
        skeletal_fig=nsd_object.nsd_fig(),
        extras=nsd_object.extras("large", True, False, "random"),
    )
    return html


@app.route("/scsd", methods=["GET"])
def scsd_in():
    if __name__ == "__main__":
        return render_template("/scsd/scsd_uploader_localtest.html")
    else:
        return render_template("/scsd/scsd_uploader.html")


@app.route("/scsd_output", methods=["GET", "POST"])
def scsd_out():
    if request.method == "POST":
        f, data = request.files["file"], request.form
        f.save(u_folder + f.filename)

        model = scsd.model_objs_dict.get(
            data.get("model_name"),
            scsd_models_user.model_objs_dict.get(data.get("model_name"), None),
        )

        if isinstance(model, scsd.scsd_model):
            ptgr, maxdist = model.ptgr, model.maxdist
        else:
            ptgr, maxdist = data.get("point_group"), 2.2

        ats = scsd.import_pdb(u_folder + f.filename)
        scsd_obj = scsd.scsd_matrix(ats, model, ptgr)
        scsd_obj.calc_scsd(
            data.get("basinhopping") == "True", by_graph=data.get("by_graph") == "True"
        )
        extras = scsd_obj.compare_table(data_path=dfs_path) + render_template(
            "/scsd/scsd_hidden_raw_data_section.html",
            raw_data=scsd_obj.raw_data(),
            table_ident="raw_data",
        )

        template = "/scsd/scsd_html_template_v2.html"
        html = render_template(
            template,
            title=f.filename[:-4],
            headbox=scsd.gen_headbox(data, ptgr, f.filename),
            nsd_table=scsd_obj.html_table(n_modes=2),
            mondrian_fig=scsd_obj.mondrian(
                as_type="buffer", cmap=data.get("cscheme_text", "Spectral")
            ),
            plotly_fig=scsd_obj.scsd_plotly(maxdist=maxdist),
            extras=extras,
        )
        return html


@app.route("/scsd_ccdc", methods=["GET"])
def scsd_ccdc_in():
    return render_template("/scsd/scsd_uploader_ccdc.html")


@app.route("/scsd_ccdc_redir", methods=["GET", "POST"])
def scsd_ccdc_redir():
    if request.method == "POST":
        return redirect("/scsd/" + request.form.get("refcode").upper())


@app.route("/scsd/<refcode>", methods=["GET"])
def scsd_ccdc_out(refcode):
    combined_df = read_pickle(dfs_path + "combined_df.pkl")
    try:
        df_name = combined_df[combined_df["name"].isin([refcode, refcode.upper()])][
            "df_name"
        ].values[0]
    except IndexError:
        return (
            refcode
            + " "
            + refcode.upper()
            + ": Refcode not found in precomputed databases - use ./scsd"
        )
    model = scsd.model_objs_dict.get(df_name, False)
    if isinstance(model, bool):
        return "Model cannot be found for " + refcode + " + " + df_name
    try:
        df = read_pickle(dfs_path + model.database_path)
    except FileNotFoundError:
        return "Database not on this server - contact Chris Kingsbury at ckingsbury@ccdc.cam.ac.uk for data"
    # scsd_obj = scsd.scsd_matrix(df[df['NAME'].isin([refcode, refcode.upper()])]['coords_matrix'].values[0], model)
    dfrow = df[df["name"].isin([refcode, refcode.upper()])]
    if len(dfrow) == 2:
        dfrow = df[df["name"] == refcode]
    scsd_obj = scsd.scsd_matrix(dfrow["coords"].values[0], model)
    scsd_obj.calc_scsd(False, bypass=True)
    if "nearest" in dfrow.columns:
        extras = scsd_obj.compare_table(
            data_path=dfs_path,
            bypass=True,
            nearest_dict={x: sqrt(y) for x, y in dfrow["nearest"].values[0].items()},
        )
    else:
        extras = scsd_obj.compare_table(data_path=dfs_path)
    extras = extras + render_template(
        "/scsd/scsd_hidden_raw_data_section.html",
        raw_data=scsd_obj.raw_data(),
        table_ident="raw_data",
    )
    data = {"model_name": model.name, "refcode": refcode}
    template = "/scsd//scsd_html_template_v2.html"
    html = render_template(
        template,
        title=refcode,
        headbox=scsd.gen_headbox(data, model.ptgr, df_name=df_name),
        nsd_table=scsd_obj.html_table(n_modes=2),
        mondrian_fig=scsd_obj.mondrian(
            as_type="buffer", cmap=random.choice(scsd.good_cmaps)
        ),
        plotly_fig=scsd_obj.scsd_plotly(as_type="html_min", maxdist=model.maxdist),
        extras=extras,
    )
    return html


@app.route("/scsd_recalc/<refcode>", methods=["GET"])
def scsd_ccdc_recalc(refcode):
    combined_df = read_pickle(dfs_path + "combined_df.pkl")
    try:
        df_name = combined_df[combined_df["name"].isin([refcode, refcode.upper()])][
            "df_name"
        ].values[0]
    except IndexError:
        return (
            refcode
            + " "
            + refcode.upper()
            + ": Refcode not found in precomputed databases - use ./scsd"
        )
    model = scsd.model_objs_dict.get(df_name, False)
    if isinstance(model, bool):
        return "Model cannot be found for " + refcode + " + " + df_name
    try:
        df = read_pickle(dfs_path + model.database_path)
    except FileNotFoundError:
        return "Database not on this server - contact Chris Kingsbury at ckingsbury@ccdc.cam.ac.uk for data"
    # scsd_obj = scsd.scsd_matrix(df[df['NAME'].isin([refcode, refcode.upper()])]['coords_matrix'].values[0], model)
    dfrow = df[df["name"].isin([refcode, refcode.upper()])]
    scsd_obj = scsd.scsd_matrix(dfrow["coords"].values[0], model)
    scsd_obj.calc_scsd(True)
    if "nearest" in dfrow.columns:
        extras = scsd_obj.compare_table(
            data_path=dfs_path,
            bypass=True,
            nearest_dict={x: sqrt(y) for x, y in dfrow["nearest"].values[0].items()},
        )
    else:
        extras = scsd_obj.compare_table(data_path=dfs_path)
    extras = extras + render_template(
        "/scsd/scsd_hidden_raw_data_section.html",
        raw_data=scsd_obj.raw_data(),
        table_ident="raw_data",
    )
    data = {"model_name": model.name, "refcode": refcode}
    template = "/scsd//scsd_html_template_v2.html"
    html = render_template(
        template,
        title=refcode,
        headbox=scsd.gen_headbox(data, model.ptgr, df_name=df_name),
        nsd_table=scsd_obj.html_table(n_modes=2),
        mondrian_fig=scsd_obj.mondrian(
            as_type="buffer", cmap=random.choice(scsd.good_cmaps)
        ),
        plotly_fig=scsd_obj.scsd_plotly(as_type="html_min", maxdist=model.maxdist),
        extras=extras,
    )
    return html


@app.route("/scsd_random", methods=["GET"])
def scsd_ccdc_random():
    combined_df = read_pickle(dfs_path + "combined_df.pkl")
    refcode = random.choice(combined_df["name"].values)
    try:
        df_name = combined_df[combined_df["name"].isin([refcode, refcode.upper()])][
            "df_name"
        ].values[0]
    except IndexError:
        return (
            refcode
            + " "
            + refcode.upper()
            + ": Refcode not found in precomputed databases - use ./scsd"
        )
    print(refcode, df_name)
    model = scsd.model_objs_dict.get(df_name, False)
    if isinstance(model, bool):
        return "Model cannot be found for " + refcode + " + " + df_name
    try:
        df = read_pickle(dfs_path + model.database_path)
    except FileNotFoundError:
        return "Database not on this server - contact Chris Kingsbury at ckingsbury@ccdc.cam.ac.uk for data"
    dfrow = df[df["name"].isin([refcode, refcode.upper()])]
    scsd_obj = scsd.scsd_matrix(dfrow["coords"].values[0], model)
    scsd_obj.calc_scsd(False, bypass=True)
    if "nearest" in dfrow.columns:
        extras = scsd_obj.compare_table(
            data_path=dfs_path, bypass=True, nearest_dict=dfrow["nearest"].values[0]
        )
    else:
        extras = scsd_obj.compare_table(data_path=dfs_path)
    extras = extras + render_template(
        "/scsd/scsd_hidden_raw_data_section.html",
        raw_data=scsd_obj.raw_data(),
        table_ident="raw_data",
    )
    data = {"model_name": model.name, "refcode": refcode}
    template = "/scsd//scsd_html_template_v2.html"
    html = render_template(
        template,
        title=refcode,
        headbox=scsd.gen_headbox(data, model.ptgr, df_name=df_name),
        nsd_table=scsd_obj.html_table(n_modes=2),
        mondrian_fig=scsd_obj.mondrian(
            as_type="buffer", cmap=random.choice(scsd.good_cmaps)
        ),
        plotly_fig=scsd_obj.scsd_plotly(as_type="html_min", maxdist=model.maxdist),
        extras=extras,
    )
    return html


@app.route("/scsd_new_model", methods=["GET"])
def scsd_mod_in():
    if __name__ == "__main__":
        return render_template("/scsd/scsd_uploader_new_model_local.html")
    else:
        return render_template("/scsd/scsd_uploader_new_model.html")


@app.route("/scsd_new_model_output", methods=["GET", "POST"])
def scsd_mod_out():
    if request.method == "POST":
        f, data = request.files["file"], request.form
        f.save(u_folder + f.filename)

        model_name = str(data.get("model_name"))
        if model_name in scsd.model_objs_dict.keys():
            return "Cannot overwrite this model name. Please choose a different name"
        model_ats_in = scsd.import_pdb(u_folder + f.filename)
        model_ats = scsd.yield_model(
            model_ats_in, data.get("point_group"), bhopping=data.get("basinhopping")
        )
        model = scsd.scsd_model(
            model_name,
            model_ats,
            data.get("point_group"),
            maxdist=float(data.get("maxdist")),
            mondrian_limits=[float(data.get("mondrian_limits"))] * 2,
        )

        tstamp = str(date.today()).replace("-", "")
        for_mod_usr = ["\n#" + f.filename + " " + model_name + " " + tstamp]
        for_mod_usr.append(model.importable())

        user_model_filepath = getcwd() + "/scsd_models_user.py"
        f2 = open(user_model_filepath, "a")
        f2.writelines("\n".join(for_mod_usr))
        f2.close()
        reload(scsd_models_user)

        extras = render_template(
            "/scsd/scsd_hidden_raw_data_section.html",
            raw_data="\n".join(for_mod_usr),
            table_ident="raw_data",
        )

        # template = model_templates.get(data.get('model_name'),'/scsd_html_template_v2.html')
        html = render_template(
            "/scsd/scsd_model_report.html",
            title=f.filename[:-4],
            headbox=model.headbox(f.filename),
            html_table=model.html_table(),
            plotly_fig=model.visualize_symm_ops(),
            extras=extras,
        )
        return html


@app.route("/scsd_model/<model_name>", methods=["GET", "POST"])
def scsd_mod_lookup(model_name):
    model = scsd.model_objs_dict.get(
        model_name, scsd_models_user.model_objs_dict.get(model_name, None)
    )
    if isinstance(model, type(None)):
        return "Cannot find model"
    tstamp = str(date.today()).replace("-", "")
    extras = render_template(
        "/scsd/scsd_hidden_raw_data_section.html",
        raw_data="\n".join(["#" + model_name + " " + tstamp, model.importable()]),
        table_ident="raw_data",
    )

    if model.database_path is not None:
        try:
            df = read_pickle(dfs_path + model.database_path)
            link = "<a href = '/scsd/{x}'>{x}</a>"
            extras = (
                extras
                + "<br>"
                + ", ".join([link.format(x=refcode) for refcode in df["name"].values])
            )
        except FileNotFoundError:
            extras = (
                extras
                + "<br> Database not on this server - contact Chris Kingsbury at ckingsbury@ccdc.cam.ac.uk for data"
            )
    html = render_template(
        "/scsd/scsd_model_report.html",
        title=model_name,
        headbox=model.headbox("<i>precomputed</i>"),
        html_table=model.html_table(),
        plotly_fig=model.visualize_symm_and_pc(),
        extras=extras,
    )
    return html


@app.route("/scsd_model_ext/<model_name>", methods=["GET", "POST"])
def scsd_mod_ext(model_name):
    model = scsd.model_objs_dict.get(
        model_name, scsd_models_user.model_objs_dict.get(model_name, None)
    )
    if isinstance(model, type(None)):
        return "Cannot find model"
    tstamp = str(date.today()).replace("-", "")
    extras = render_template(
        "/scsd/scsd_hidden_raw_data_section.html",
        raw_data="\n".join(["#" + model_name + " " + tstamp, model.importable()]),
    )

    if model.database_path is not None:
        try:
            coll = scsd.scsd_collection(model_name)
            # try: coll.get_pca_from_model()
            # except:
            coll.gen_pca(2)
            model.pca = coll.pca
            extras = render_template(
                "/scsd/scsd_hidden_raw_data_section.html",
                raw_data="\n".join(
                    ["#" + model_name + " " + tstamp, model.importable()]
                ),
                table_ident="raw_data",
            )

            df = coll.gen_complex_df()
            link = "<a href = '/scsd/{x}'>{x}</a>"
            extras = (
                extras
                + "<br>"
                + ", ".join([link.format(x=refcode) for refcode in df["name"].values])
            )
            try:
                irs = coll.model.pca.keys()
            except:
                irs = coll.model.symm.pgt.keys()
            extras = (
                extras
                + "<br>"
                + ", ".join([coll.pca_kdeplot(x, as_type="html") for x in irs])
            )

        except FileNotFoundError:
            extras = (
                extras
                + "<br> Database not on this server - contact Chris Kingsbury at ckingsbury@ccdc.cam.ac.uk for data"
            )
    html = render_template(
        "/scsd/scsd_model_report.html",
        title=model_name,
        headbox=model.headbox(f"<a href = '/scsd_data/{model_name}'>{model_name}</a>"),
        html_table=model.html_table(),
        plotly_fig=model.visualize_symm_and_pc(),
        extras=extras,
    )
    return html


@app.route("/scsd_data/<model_name>", methods=["GET", "POST"])
def scsd_data(model_name):
    model = scsd.model_objs_dict.get(
        model_name, scsd_models_user.model_objs_dict.get(model_name, None)
    )
    try:
        return send_from_directory(dfs_path, model.database_path, as_attachment=True)
    except FileNotFoundError:
        return "Database not on this server - contact Chris Kingsbury at ckingsbury@ccdc.cam.ac.uk for data"


@app.route("/scsd_models_table", methods=["GET"])
def scsd_mod_all():

    model_names = list(scsd.model_objs_dict.keys()) + list(
        unique(list(scsd_models_user.model_objs_dict.keys()))
    )
    table = [
        "<th> name(link)  </th><th> chem formula </th><th>point group</th><th> data_path </th>"
    ]
    for name in model_names:
        model = scsd.model_objs_dict.get(
            name, scsd_models_user.model_objs_dict.get(name, None)
        )

        table.append(
            "</td><td>".join(
                [
                    f"<td><a href=/scsd_model/{name}>{name}</a>",
                    model.chem_formula(),
                    model.ptgr_html,
                    str(model.database_path) + "</td>",
                ]
            )
        )
    html = render_template(
        "/scsd/scsd_models_table.html",
        table="<table id='modstab'><tr>" + "</tr>\n<tr>".join(table) + "</tr></table>",
    )

    return html


@app.route("/coord_table_ref_str_mat")
def return_table():
    return send_from_directory("", "/coord_table_ref_str_mat.html")


@app.route("/bdba_defs")
def return_bdba_defs():
    return send_from_directory("", "/bdba_defs.html")


@app.route("/nsd_common_issues")
def return_common_issues():
    return send_from_directory("", "/nsd_common_issues.html")


@app.route("/alert_defs")
def return_alert_defs():
    return send_from_directory("", "/alert_defs.html")


@app.route("/database_structure_refcodes")
def return_db_refcodes():
    return send_from_directory("", "/database_structure_refcodes.html")


@app.route("/")
def return_s1():
    return render_template("/kingsbury/Profile.html")


@app.route("/profile")
def return_s2():
    return render_template("/kingsbury/Profile.html")


@app.route("/research")
def return_s3():
    return render_template("/kingsbury/Research.html")


@app.route("/publications")
def return_s4():
    return render_template("/kingsbury/Publications.html")


@app.route("/contact")
def return_s5():
    return render_template("/kingsbury/Contact.html")


@app.route("/data")
def return_s6():
    return render_template("/kingsbury/data.html")


def start_server():
    import webbrowser

    webbrowser.open_new("http://localhost:5050/scsd_random")
    app.run(host="0.0.0.0", port=5050)


if __name__ == "__main__":
    start_server()
