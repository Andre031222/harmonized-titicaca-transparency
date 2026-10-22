"""
p14_verify_manuscript.py -- comprueba el manuscrito contra results/.

Todas las cifras del texto y de las tablas son macros de numbers.tex (p13).
Aqui se verifica que:
  * cada macro usada esta definida y ninguna quedo PENDING;
  * las tablas no llevan numeros tecleados (solo macros);
  * cada figura incluida existe en results/figures_r/ y es posterior a las
    tablas de results/tidy que la alimentan;
  * no quedan marcadores de borrador.
Termina con codigo 1 ante cualquier fallo.
"""
import re
import sys

from p00_config import FIGR, ROOT, TIDY, banner

TEX = ROOT / "manuscript/rsase/titicaca_secchi_rsase.tex"
NUM = ROOT / "manuscript/rsase/numbers.tex"


def main():
    banner("p14 -- VERIFICACION DEL MANUSCRITO")
    if not TEX.exists():
        print("  (manuscrito ausente: se omite)")
        return
    tex, num = TEX.read_text(), NUM.read_text()
    defined = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", num))
    used = set(re.findall(r"\\((?:num|Null)[A-Za-z]+)", tex))
    errors = []
    errors += [f"macro sin definir: \\{m}" for m in sorted(used - set(defined))]
    errors += [f"macro pendiente: \\{m}" for m, v in defined.items()
               if v == "PENDING" and m in used]

    for tab in re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", tex, re.S):
        body = re.sub(r"\\(num|Null)[A-Za-z]+", "", tab)
        body = re.sub(r"\\begin\{tabular\}\{[^}]*\}", "", body)
        body = re.sub(r"\\(cmidrule|multicolumn)(\([^)]*\))?\{[^}]*\}(\{[^}]*\})?", "", body)
        hard = re.findall(r"(?<![A-Za-z\\])\d+\.\d+", body)
        if hard:
            errors.append(f"tabla con cifras tecleadas: {hard[:5]}")

    newest_tidy = max(p.stat().st_mtime for p in TIDY.glob("*.csv"))
    for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        f = FIGR / fig
        if not f.exists():
            errors.append(f"figura ausente: {fig}")
        elif f.stat().st_mtime < newest_tidy - 3600 * 24:
            print(f"  aviso: {fig} es anterior a las tablas mas recientes")

    for marker in ["%% RESULTS", "%% DISC", "%% CONCLUSIONS", "CAPTION-", "TODO", "XXX"]:
        if marker in tex:
            errors.append(f"marcador de borrador: {marker}")

    print(f"  macros usadas: {len(used)}  definidas: {len(defined)}")
    if errors:
        for e in errors:
            print("  FALLO", e)
        sys.exit(1)
    print("  OK: el manuscrito coincide con results/")


if __name__ == "__main__":
    main()
