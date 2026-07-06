"""
Provjera duplikata i kombinacija koje fale za odredjenu osobu.

Koristenje:
    python tests/check_annotations.py --person 1
    python tests/check_annotations.py --person 3
    python tests/check_annotations.py --all
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ANNOTATIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "annotations.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

HANDS = ["left", "right"]
HAND_SIDES = ["dorsal", "palm"]
BANKNOTE_SIDES = ["single_number", "double_number"]
LIGHTINGS = ["L1", "L2", "L3", "L4", "L5", "L6"]
DENOMINATIONS = [5, 10, 20, 50, 100]

EXPECTED_COUNT = len(HANDS) * len(HAND_SIDES) * len(BANKNOTE_SIDES) * len(LIGHTINGS) * len(DENOMINATIONS)


def combo_key(a: dict) -> tuple:
    return (a["hand"], a["hand_side"], a["banknote_side"], a["lighting_id"], a["denomination"])


def analyze_person(annotations: list[dict], person_id: int) -> dict:
    entries = [a for a in annotations if a["person_id"] == person_id]
    if not entries:
        return {"person_id": person_id, "total": 0, "missing": [], "duplicates": [], "unique": 0}

    counts = Counter(combo_key(a) for a in entries)

    missing = []
    duplicates = []

    for h in HANDS:
        for hs in HAND_SIDES:
            for bs in BANKNOTE_SIDES:
                for l in LIGHTINGS:
                    for d in DENOMINATIONS:
                        combo = (h, hs, bs, l, d)
                        cnt = counts.get(combo, 0)
                        if cnt == 0:
                            missing.append(combo)
                        elif cnt > 1:
                            imgs = [a["image"] for a in entries if combo_key(a) == combo]
                            duplicates.append((combo, imgs))

    return {
        "person_id": person_id,
        "total": len(entries),
        "unique": len(counts),
        "missing": missing,
        "duplicates": duplicates,
    }


def generate_md(result: dict) -> str:
    pid = result["person_id"]
    lines = [
        f"# Annotations Info — Osoba {pid} (person_id={pid})",
        "",
        f"**Ukupno anotacija:** {result['total']}",
        f"**Ocekivano jedinstvenih kombinacija:** {EXPECTED_COUNT}"
        f" ({len(HANDS)} ruke x {len(HAND_SIDES)} strane x {len(BANKNOTE_SIDES)} strane novcanice"
        f" x {len(LIGHTINGS)} osvetljenja x {len(DENOMINATIONS)} apoena)",
        f"**Pronadjeno jedinstvenih kombinacija:** {result['unique']}",
        "",
        "Kljuc jedinstvenosti: `(hand, hand_side, banknote_side, lighting_id, denomination)`",
        "",
        "---",
        "",
    ]

    missing = result["missing"]
    lines.append(f"## Kombinacije koje fale ({len(missing)})")
    lines.append("")
    if missing:
        lines.append("| # | Hand | Hand Side | Banknote Side | Lighting | Denomination |")
        lines.append("|---|------|-----------|---------------|----------|-------------|")
        for i, m in enumerate(missing, 1):
            lines.append(f"| {i} | {m[0]} | {m[1]} | {m[2]} | {m[3]} | {m[4]}€ |")
    else:
        lines.append("Nema.")
    lines.append("")
    lines.append("---")
    lines.append("")

    dupes = result["duplicates"]
    lines.append(f"## Duplikati ({len(dupes)})")
    lines.append("")
    if dupes:
        lines.append("Kombinacije koje se pojavljuju vise od jednom:")
        lines.append("")
        lines.append("| # | Hand | Hand Side | Banknote Side | Lighting | Denomination | Slike |")
        lines.append("|---|------|-----------|---------------|----------|-------------|----------|")
        for i, (combo, imgs) in enumerate(dupes, 1):
            imgs_str = ", ".join(imgs)
            lines.append(f"| {i} | {combo[0]} | {combo[1]} | {combo[2]} | {combo[3]} | {combo[4]}€ | {imgs_str} |")
    else:
        lines.append("Nema.")
    lines.append("")
    lines.append("---")
    lines.append("")

    n_missing = len(missing)
    n_dupes = len(dupes)
    lines.append("## Sazetak")
    lines.append("")
    lines.append(f"- {n_missing} kombinacija fali")
    lines.append(f"- {n_dupes} duplikata (kombinacija s vise od 1 slike)")
    expected_total = EXPECTED_COUNT - n_missing + n_dupes
    lines.append(f"- Racunica: {EXPECTED_COUNT} ocekivanih - {n_missing} fali + {n_dupes} duplikata = {expected_total} anotacija ({result['total']} stvarno)")
    if expected_total != result["total"]:
        lines.append(f"- **UPOZORENJE:** racunica se ne poklapa! Razlika: {result['total'] - expected_total}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Provjera anotacija po osobi")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--person", type=int, help="person_id za provjeru")
    group.add_argument("--all", action="store_true", help="provjeri sve osobe")
    args = parser.parse_args()

    with open(ANNOTATIONS_PATH) as f:
        annotations = json.load(f)

    person_ids = sorted(set(a["person_id"] for a in annotations)) if args.all else [args.person]

    for pid in person_ids:
        result = analyze_person(annotations, pid)

        if result["total"] == 0:
            print(f"Osoba {pid}: nema anotacija u {ANNOTATIONS_PATH.name}")
            continue

        md = generate_md(result)
        out_path = OUTPUT_DIR / f"annotations_info_person{pid}.md"
        out_path.write_text(md, encoding="utf-8")

        status = "OK" if not result["missing"] and not result["duplicates"] else "PROBLEMI"
        print(f"Osoba {pid}: {result['total']} anotacija, "
              f"{len(result['missing'])} fali, {len(result['duplicates'])} duplikata "
              f"[{status}] -> {out_path.name}")


if __name__ == "__main__":
    main()
