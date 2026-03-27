#!/usr/bin/env python3
"""
Parse per-country QKD simulation reports and generate LaTeX summary tables.

This script scans a directory containing one simulation-report file per country,
extracts selected numerical results from the plain-text reports, and writes two
LaTeX outputs:

1. a standard ``table`` environment stored in ``eums_summary_table.tex``,
2. a ``longtable`` version stored in ``eums_summary_table_longtable.tex``.

Purpose
-------
The script is intended as a post-processing step after the simulation workflow
has already produced country-level report files. It does not perform any
simulation itself. Its only task is to:
- locate the report files,
- parse predefined metrics from the report text,
- collect the results into structured dictionaries,
- and render those results as LaTeX tabular output.

Expected input structure
------------------------
The script expects a root directory with one subdirectory per country. Each
country directory is expected to contain a file named:

    qkd_simulation_report.txt

For example:

    EUMS_QKD_Network_Results/
        Austria/qkd_simulation_report.txt
        Belgium/qkd_simulation_report.txt
        ...
        Sweden/qkd_simulation_report.txt

Parsed quantities
-----------------
From each report, the script attempts to recover:
- country name,
- number of QKD endpoints,
- number of trusted repeater nodes,
- degree-distribution counts,
- first-run total fiber length,
- first-run hop statistics after trusted repeater splitting,
- Monte Carlo summary statistics for:
  - total fiber length,
  - mean hop length after TRN split,
  - maximum hop length after TRN split.

Output interpretation
---------------------
For each country, the generated LaTeX table contains:
- an ``estimate`` row, which uses the Monte Carlo mean values,
- a ``std`` row, which contains the corresponding standard deviations.

Formatting conventions
----------------------
- Missing parsed values are rendered as ``--``.
- Integer and floating-point formatting are centralized in helper functions.
- LaTeX special characters in state names are escaped before output.

Notes
-----
The code below is kept functionally identical to the provided version. Only
documentation and explanatory comments are added.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# Configuration
# ============================================================

# Root directory that contains the per-country simulation result folders.
# Each country folder is expected to contain a file named
# ``qkd_simulation_report.txt``.
INPUT_ROOT = Path("EUMS_QKD_Network_Results")

# Main output file for the standard LaTeX table environment.
OUTPUT_TEX = Path("eums_summary_table.tex")


# ============================================================
# Helpers
# ============================================================

def latex_escape(text: str) -> str:
    """
    Escape LaTeX-sensitive characters in plain text.

    This is used primarily for country names or other labels that are inserted
    directly into the generated LaTeX table. Characters with special meaning in
    LaTeX are replaced by safe escaped representations.

    Parameters
    ----------
    text:
        Input string to be escaped.

    Returns
    -------
    str
        LaTeX-safe string.
    """
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(repl.get(ch, ch))
    return "".join(out)


def fmt_int(x: Optional[int]) -> str:
    """
    Format an optional integer for LaTeX table output.

    Parameters
    ----------
    x:
        Integer value or ``None``.

    Returns
    -------
    str
        Decimal string if the value exists, otherwise ``"--"``.
    """
    if x is None:
        return "--"
    return f"{x:d}"


def fmt_float(x: Optional[float], ndigits: int = 3) -> str:
    """
    Format an optional floating-point number for LaTeX table output.

    Parameters
    ----------
    x:
        Floating-point value or ``None``.
    ndigits:
        Number of digits after the decimal point.

    Returns
    -------
    str
        Formatted decimal string if the value exists, otherwise ``"--"``.
    """
    if x is None:
        return "--"
    return f"{x:.{ndigits}f}"


def parse_first_match(pattern: str, text: str, flags: int = 0) -> Optional[str]:
    """
    Return the first captured regex group for a pattern, if present.

    Parameters
    ----------
    pattern:
        Regular expression containing at least one capturing group.
    text:
        Input text in which the pattern is searched.
    flags:
        Optional regular-expression flags.

    Returns
    -------
    Optional[str]
        First captured group if a match exists, otherwise ``None``.
    """
    m = re.search(pattern, text, flags)
    return m.group(1) if m else None


def parse_int(pattern: str, text: str, flags: int = 0) -> Optional[int]:
    """
    Parse an integer from the first captured regex group.

    Commas are removed before conversion, which allows parsing values such as
    ``"1,000"``.

    Parameters
    ----------
    pattern:
        Regular expression containing one capturing group.
    text:
        Input text to search.
    flags:
        Optional regular-expression flags.

    Returns
    -------
    Optional[int]
        Parsed integer, or ``None`` if no match is found.
    """
    s = parse_first_match(pattern, text, flags)
    if s is None:
        return None
    return int(s.replace(",", "").strip())


def parse_float(pattern: str, text: str, flags: int = 0) -> Optional[float]:
    """
    Parse a floating-point number from the first captured regex group.

    Commas are removed before conversion, which allows parsing values such as
    ``"1,234.567"``.

    Parameters
    ----------
    pattern:
        Regular expression containing one capturing group.
    text:
        Input text to search.
    flags:
        Optional regular-expression flags.

    Returns
    -------
    Optional[float]
        Parsed float, or ``None`` if no match is found.
    """
    s = parse_first_match(pattern, text, flags)
    if s is None:
        return None
    return float(s.replace(",", "").strip())


def parse_metric_block(metric_title: str, text: str) -> Dict[str, Optional[float]]:
    """
    Parse a metric summary block from the report text.

    The function looks for report sections of the form:

    Total fiber length (km)
      raw:      mean ± std = 123.456 ± 7.890   (stderr 0.249)
                min / max  = ...
      adjusted: mean ± std = 185.184 ± 11.835   (stderr 0.374)   [x1.500]
                min / max  = ...

    It extracts the raw and adjusted mean, standard deviation, and standard
    error values.

    Parameters
    ----------
    metric_title:
        Exact section title to search for.
    text:
        Full report text.

    Returns
    -------
    Dict[str, Optional[float]]
        Dictionary with the keys:
        ``raw_mean``, ``raw_std``, ``raw_stderr``,
        ``adj_mean``, ``adj_std``, ``adj_stderr``.

        If the block cannot be found, all values remain ``None``.
    """
    out = {
        "raw_mean": None,
        "raw_std": None,
        "raw_stderr": None,
        "adj_mean": None,
        "adj_std": None,
        "adj_stderr": None,
    }

    block_pat = (
        re.escape(metric_title)
        + r"\s*"
        + r"raw:\s*mean ± std =\s*([0-9eE+.\-]+)\s*±\s*([0-9eE+.\-]+)\s*\(stderr\s*([0-9eE+.\-]+)\)"
        + r".*?"
        + r"adjusted:\s*mean ± std =\s*([0-9eE+.\-]+)\s*±\s*([0-9eE+.\-]+)\s*\(stderr\s*([0-9eE+.\-]+)\)"
    )

    m = re.search(block_pat, text, flags=re.DOTALL)
    if not m:
        return out

    out["raw_mean"] = float(m.group(1))
    out["raw_std"] = float(m.group(2))
    out["raw_stderr"] = float(m.group(3))
    out["adj_mean"] = float(m.group(4))
    out["adj_std"] = float(m.group(5))
    out["adj_stderr"] = float(m.group(6))
    return out


def parse_degree_dist(text: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Parse the degree-distribution line from a report.

    The expected line format is:

    Target degree dist:       k2=100, k3=150, k4=0, k5=0

    Parameters
    ----------
    text:
        Full report text.

    Returns
    -------
    Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]
        Parsed values ``(k2, k3, k4, k5)``.
        If the line is not found, all entries are ``None``.
    """
    m = re.search(
        r"Target degree dist:\s*k2\s*=\s*(\d+)\s*,\s*k3\s*=\s*(\d+)\s*,\s*k4\s*=\s*(\d+)\s*,\s*k5\s*=\s*(\d+)",
        text
    )
    if not m:
        return None, None, None, None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


# ============================================================
# Report parsing
# ============================================================

def parse_report(report_path: Path) -> Dict:
    """
    Parse a single country simulation report.

    The function extracts selected scalar values and metric summaries from the
    plain-text report and returns them in a structured dictionary used later
    for LaTeX table generation.

    Parameters
    ----------
    report_path:
        Path to a ``qkd_simulation_report.txt`` file.

    Returns
    -------
    Dict
        Parsed metric dictionary for one country.
    """
    text = report_path.read_text(encoding="utf-8")

    country = parse_first_match(r"^(.*?)\s+QKD Network Simulation Report", text, flags=re.MULTILINE)
    if country is None:
        country = report_path.parent.name

    n_endpoints = parse_int(r"QKD endpoints total:\s*([0-9,]+)", text)
    n_trns = parse_int(r"TRNs \(placed on edges\):\s*([0-9,]+)", text)
    k2, k3, k4, k5 = parse_degree_dist(text)

    first_run_fiber = parse_float(r"Total fiber \(km\):\s*([0-9eE+.\-]+)", text)

    first_run_hop_match = re.search(
        r"Hop stats after TRN split \(km\): min/mean/max =\s*([0-9eE+.\-]+)\s*/\s*([0-9eE+.\-]+)\s*/\s*([0-9eE+.\-]+)",
        text
    )
    if first_run_hop_match:
        first_run_hop_min = float(first_run_hop_match.group(1))
        first_run_hop_mean = float(first_run_hop_match.group(2))
        first_run_hop_max = float(first_run_hop_match.group(3))
    else:
        first_run_hop_min = None
        first_run_hop_mean = None
        first_run_hop_max = None

    fiber_stats = parse_metric_block("Total fiber length (km)", text)
    mean_hop_stats = parse_metric_block("Mean hop length after TRN split (km)", text)
    max_hop_stats = parse_metric_block("Max hop length after TRN split (km)", text)

    return {
        "country": country,
        "report_path": str(report_path),
        "n_endpoints": n_endpoints,
        "n_trns": n_trns,
        "k2": k2,
        "k3": k3,
        "k4": k4,
        "k5": k5,
        "first_run_fiber": first_run_fiber,
        "first_run_hop_min": first_run_hop_min,
        "first_run_hop_mean": first_run_hop_mean,
        "first_run_hop_max": first_run_hop_max,
        "fiber_raw_mean": fiber_stats["raw_mean"],
        "fiber_raw_std": fiber_stats["raw_std"],
        "fiber_raw_stderr": fiber_stats["raw_stderr"],
        "fiber_adj_mean": fiber_stats["adj_mean"],
        "fiber_adj_std": fiber_stats["adj_std"],
        "fiber_adj_stderr": fiber_stats["adj_stderr"],
        "meanhop_raw_mean": mean_hop_stats["raw_mean"],
        "meanhop_raw_std": mean_hop_stats["raw_std"],
        "meanhop_raw_stderr": mean_hop_stats["raw_stderr"],
        "meanhop_adj_mean": mean_hop_stats["adj_mean"],
        "meanhop_adj_std": mean_hop_stats["adj_std"],
        "meanhop_adj_stderr": mean_hop_stats["adj_stderr"],
        "maxhop_raw_mean": max_hop_stats["raw_mean"],
        "maxhop_raw_std": max_hop_stats["raw_std"],
        "maxhop_raw_stderr": max_hop_stats["raw_stderr"],
        "maxhop_adj_mean": max_hop_stats["adj_mean"],
        "maxhop_adj_std": max_hop_stats["adj_std"],
        "maxhop_adj_stderr": max_hop_stats["adj_stderr"],
    }


def load_all_reports(root: Path) -> List[Dict]:
    """
    Load and parse all available report files below the input root.

    The function searches one directory level below ``root`` for files named
    ``qkd_simulation_report.txt``, parses each report, and returns the results
    sorted by country name.

    Parameters
    ----------
    root:
        Root directory containing country subdirectories.

    Returns
    -------
    List[Dict]
        List of parsed report dictionaries.
    """
    reports = sorted(root.glob("*/qkd_simulation_report.txt"))
    rows: List[Dict] = []

    for rp in reports:
        try:
            rows.append(parse_report(rp))
        except Exception as e:
            print(f"[WARN] Failed to parse {rp}: {e!r}")

    rows.sort(key=lambda d: d["country"])
    return rows


# ============================================================
# LaTeX table generation
# ============================================================

def build_latex_table(rows: List[Dict]) -> str:
    """
    Build a standard LaTeX ``table`` environment from parsed report rows.

    For each country, the function emits:
    - one row with estimated values based on Monte Carlo means,
    - one row with the corresponding standard deviations.

    Parameters
    ----------
    rows:
        Parsed report dictionaries.

    Returns
    -------
    str
        Complete LaTeX table source.
    """
    lines: List[str] = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{llrrrrrrrrrr}")
    lines.append(r"\hline")
    lines.append(
        r"State & Row "
        r"& $N_{\mathrm{EP}}$ "
        r"& $N_{\mathrm{TRN}}$ "
        r"& $k_2$ "
        r"& $k_3$ "
        r"& $\overline{h}$ raw "
        r"& $\overline{h}$ adj "
        r"& $h_{\max}$ raw "
        r"& $h_{\max}$ adj "
        r"& $L_{\mathrm{grid}}$ raw "
        r"& $L_{\mathrm{grid}}$ adj \\"
    )
    lines.append(r"\hline")

    for row in rows:
        state = latex_escape(str(row["country"]))

        # Main estimate row containing the Monte Carlo mean values and the
        # static configuration counts.
        lines.append(
            " & ".join(
                [
                    state,
                    "estimate",
                    fmt_int(row["n_endpoints"]),
                    fmt_int(row["n_trns"]),
                    fmt_int(row["k2"]),
                    fmt_int(row["k3"]),
                    fmt_float(row["meanhop_raw_mean"]),
                    fmt_float(row["meanhop_adj_mean"]),
                    fmt_float(row["maxhop_raw_mean"]),
                    fmt_float(row["maxhop_adj_mean"]),
                    fmt_float(row["fiber_raw_mean"]),
                    fmt_float(row["fiber_adj_mean"]),
                ]
            )
            + r" \\"
        )

        # Standard deviation row for the same country. Structural counts are not
        # repeated here and are replaced by ``--``.
        lines.append(
            " & ".join(
                [
                    "",
                    "std",
                    "--",
                    "--",
                    "--",
                    "--",
                    fmt_float(row["meanhop_raw_std"]),
                    fmt_float(row["meanhop_adj_std"]),
                    fmt_float(row["maxhop_raw_std"]),
                    fmt_float(row["maxhop_adj_std"]),
                    fmt_float(row["fiber_raw_std"]),
                    fmt_float(row["fiber_adj_std"]),
                ]
            )
            + r" \\"
        )

        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Summary of QKD simulation results per EU member state. "
        r"For each state, the first row reports the estimate and the second row reports the corresponding standard deviation from the Monte Carlo summary. "
        r"$N_{\mathrm{EP}}$ denotes the number of QKD endpoints, $N_{\mathrm{TRN}}$ the number of trusted repeater nodes, "
        r"$k_2$ and $k_3$ the degree-class counts, $\overline{h}$ the mean hop length after TRN splitting, "
        r"$h_{\max}$ the maximum hop length after TRN splitting, and $L_{\mathrm{grid}}$ the total fiber/grid length. "
        r"“raw” refers to the direct Monte Carlo result and “adj” to the value after application of the detour factor.}"
    )
    lines.append(r"\label{tab:eums_qkd_summary}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


# ============================================================
# Optional wide-table version
# ============================================================

def build_latex_table_longtable(rows: List[Dict]) -> str:
    """
    Build a LaTeX ``longtable`` version of the summary table.

    This variant is suitable for documents in which the summary may span
    multiple pages.

    Parameters
    ----------
    rows:
        Parsed report dictionaries.

    Returns
    -------
    str
        Complete LaTeX longtable source.
    """
    lines: List[str] = []

    lines.append(r"\begin{longtable}{llrrrrrrrrrr}")
    lines.append(r"\caption{Summary of QKD simulation results per EU member state. Estimates are followed by a second row containing the corresponding standard deviations.}\label{tab:eums_qkd_summary_long} \\")
    lines.append(r"\hline")
    lines.append(
        r"State & Row "
        r"& $N_{\mathrm{EP}}$ "
        r"& $N_{\mathrm{TRN}}$ "
        r"& $k_2$ "
        r"& $k_3$ "
        r"& $\overline{h}$ raw "
        r"& $\overline{h}$ adj "
        r"& $h_{\max}$ raw "
        r"& $h_{\max}$ adj "
        r"& $L_{\mathrm{grid}}$ raw "
        r"& $L_{\mathrm{grid}}$ adj \\"
    )
    lines.append(r"\hline")
    lines.append(r"\endfirsthead")

    # Repeated header for subsequent longtable pages.
    lines.append(r"\hline")
    lines.append(
        r"State & Row "
        r"& $N_{\mathrm{EP}}$ "
        r"& $N_{\mathrm{TRN}}$ "
        r"& $k_2$ "
        r"& $k_3$ "
        r"& $\overline{h}$ raw "
        r"& $\overline{h}$ adj "
        r"& $h_{\max}$ raw "
        r"& $h_{\max}$ adj "
        r"& $L_{\mathrm{grid}}$ raw "
        r"& $L_{\mathrm{grid}}$ adj \\"
    )
    lines.append(r"\hline")
    lines.append(r"\endhead")

    for row in rows:
        state = latex_escape(str(row["country"]))

        lines.append(
            " & ".join(
                [
                    state,
                    "estimate",
                    fmt_int(row["n_endpoints"]),
                    fmt_int(row["n_trns"]),
                    fmt_int(row["k2"]),
                    fmt_int(row["k3"]),
                    fmt_float(row["meanhop_raw_mean"]),
                    fmt_float(row["meanhop_adj_mean"]),
                    fmt_float(row["maxhop_raw_mean"]),
                    fmt_float(row["maxhop_adj_mean"]),
                    fmt_float(row["fiber_raw_mean"]),
                    fmt_float(row["fiber_adj_mean"]),
                ]
            )
            + r" \\"
        )

        lines.append(
            " & ".join(
                [
                    "",
                    "std",
                    "--",
                    "--",
                    "--",
                    "--",
                    fmt_float(row["meanhop_raw_std"]),
                    fmt_float(row["meanhop_adj_std"]),
                    fmt_float(row["maxhop_raw_std"]),
                    fmt_float(row["maxhop_adj_std"]),
                    fmt_float(row["fiber_raw_std"]),
                    fmt_float(row["fiber_adj_std"]),
                ]
            )
            + r" \\"
        )

        lines.append(r"\hline")

    lines.append(r"\end{longtable}")

    return "\n".join(lines) + "\n"


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Run the full report-parsing and LaTeX-export workflow.

    The function:
    1. validates the input directory,
    2. loads all available report files,
    3. builds the standard table and longtable LaTeX outputs,
    4. writes both files to disk,
    5. prints a short completion summary.
    """
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input folder does not exist: {INPUT_ROOT.resolve()}")

    rows = load_all_reports(INPUT_ROOT)
    if not rows:
        raise RuntimeError(f"No report files found under {INPUT_ROOT.resolve()}")

    tex = build_latex_table(rows)
    OUTPUT_TEX.write_text(tex, encoding="utf-8")

    # Optional second export using the longtable environment for wider or
    # multi-page document layouts.
    OUTPUT_TEX.with_name(OUTPUT_TEX.stem + "_longtable.tex").write_text(
        build_latex_table_longtable(rows),
        encoding="utf-8",
    )

    print(f"[done] Parsed {len(rows)} reports")
    print(f"[out] Wrote table -> {OUTPUT_TEX.resolve()}")
    print(f"[out] Wrote longtable -> {OUTPUT_TEX.with_name(OUTPUT_TEX.stem + '_longtable.tex').resolve()}")


if __name__ == "__main__":
    main()