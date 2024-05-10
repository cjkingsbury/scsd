#
# This script can be used for any purpose without limitation subject to the
# conditions at http://www.ccdc.cam.ac.uk/Community/Pages/Licences/v2.aspx
#
# This permission notice and the following statement of attribution must be
# included in all copies or substantial portions of this script.
#
# 2023-08-18 Created by Chris Kingsbury, the Cambridge Crystallographic Data Centre
# ORCID 0000-0002-4694-5566
#
#
from ccdc.utilities import ApplicationInterface
from re import split as resplit, compile as recompile, escape as reescape

default_settings = {}
#
replacements = {
    "-": "",
    "Cl": "[Cl]~",
    "c": "[C,c]~",
    "C": "[C,c]~",
    "n": "[N,n]~",
    "N": "[N,n]~",
    "B": "[B]~",
    "P": "[P]~",
    "S": "[S]~",
    "O": "[O]~",
    "F": "[F]~",
    "[Si]": "[Si]~",
    "21": "2~1",
    "[cH2+]": "[C,c]~",
}

repl_2 = {f"~{str(x)}": f"{str(x)}~" for x in range(1, 10)}
repl_3 = {
    "~)": ")~",
    "~=": "~",
    "~(=": "(~",
    "~(~": "(~",
    ")[": ")~[",
    "~(": "(~",
    "(~O)O": "(~O)~O~",
    "[N]": "[N,n]",
    "[cH]": "[C,c]",
    "[cH2]": "[C,c]",
    "~#": "~",
    "~~2": "",
    "=": "~",
    "~~": "~",
    "2~2": "~2",
}

button_js = """<script type="text/javascript">
    function selectElementContents(el) {
        var body = document.body, range, sel;
        if (document.createRange && window.getSelection) {
            range = document.createRange();
            sel = window.getSelection();
            sel.removeAllRanges();
            try {
                range.selectNodeContents(el);
                sel.addRange(range);
            } catch (e) {
                range.selectNode(el);
                sel.addRange(range);
            }
            document.execCommand("copy");

        } else if (body.createTextRange) {
            range = body.createTextRange();
            range.moveToElementText(el);
            range.select();
            range.execCommand("Copy");
        }
    }
</script>"""


def make_smarter(x):
    pattern = recompile("|".join(reescape(k) for k in replacements.keys()))
    if x.startswith("["):
        inside = [f"[{a.rstrip('-').rstrip('+')}]" for a in resplit(r"\[|\]", x)[1::2]]
        outside = resplit(r"\[|\]", x)[2::2] + [""]
        outside_repl = [
            pattern.sub(lambda y: replacements[y.group()], line) for line in outside
        ]
        subtotal = "~".join(f"{a}~{b}" for a, b in zip(inside, outside_repl))
    elif x.count("[") > 0:
        inside = [
            f"[{a.rstrip('-').rstrip('+')}]" for a in resplit(r"\[|\]", x)[1::2]
        ] + [""]
        outside = resplit(r"\[|\]", x)[0::2]
        outside_repl = [
            pattern.sub(lambda y: replacements[y.group()], line) for line in outside
        ]
        subtotal = "~".join(f"{b}~{a}" for a, b in zip(inside, outside_repl))
    else:
        subtotal = pattern.sub(lambda y: replacements[y.group()], x)

    for k, v in repl_2.items():
        if subtotal.count(k) > 2:
            s1s, s2s = subtotal.split(k)[0::2], subtotal.split(k)[1::2]
            if len(s1s) == len(s2s):
                subtotal = "".join([s1 + v + s2 + k + "~" for s1, s2 in zip(s1s, s2s)])
            else:
                subtotal = (
                    "".join([s1 + v + s2 + k + "~" for s1, s2 in zip(s1s, s2s)])
                    + s1s[-1]
                )

        elif subtotal.find(k) > 0:
            a, _, c = subtotal.partition(k)
            subtotal = a + v + c
    for k, v in repl_3.items():
        if subtotal.find(k) > -1:
            subtotal = v.join(subtotal.split(k))

    return subtotal.rstrip("~")


def gen_smarts(settings=default_settings):
    interface = ApplicationInterface(parse_commandline=False)
    interface.parse_commandline()
    entry = interface.current_entry
    crystal = entry.crystal
    molecule = crystal.molecule
    molecule.assign_bond_types(which="unknown")

    if len(interface.selected_atoms) > 0:
        ats_labels = [x.label for x in interface.selected_atoms]
        frag = molecule.copy()
        not_in_frag = [
            x for x in frag.atoms if (x.label not in ats_labels) or x.is_metal
        ]
        frag.remove_atoms(not_in_frag)
        frag.add_hydrogens("all")
        smiles = frag.smiles.split(".")[0]
    else:
        molecule.add_hydrogens("all")
        smiles = molecule.smiles

    with interface.html_report(title=f"smiles for {interface.identifier}") as report:
        report.write(button_js)
        report.write(
            f'<p id="smiles">{smiles}</p>'
            """<button onclick="selectElementContents( document.getElementById('smiles') );">Copy SMILES</button>"""
        )
        report.write(
            f'<br><p id="smarts">{str(make_smarter(smiles))}</p>'
            """<button onclick="selectElementContents( document.getElementById('smarts') );">Copy SMARTS</button>"""
        )


if __name__ == "__main__":
    gen_smarts()
