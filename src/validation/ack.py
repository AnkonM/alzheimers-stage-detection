"""OASIS-1 acknowledgement, citation and results marker. Single source for every generator.

tools/oasis_guard.py keeps a standalone copy (it must run with the standard library only); a test asserts the
two are identical. Any generated document that contains OASIS results carries MARKER, the acknowledgement and
the DOI, and the guard refuses a file that has the marker without both.
"""
ACK_TEXT = ("Data were provided by OASIS-1: Cross-Sectional: Principal Investigators: D. Marcus, R. Buckner, "
            "J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, "
            "U24 RR021382")
DOI = "https://doi.org/10.1162/jocn.2007.19.9.1498"
CITATION = "Marcus et al., J Cogn Neurosci 2007;19(9):1498-1507, " + DOI
MARKER = "OASIS" + "_RESULTS_MARKER"


def notice() -> str:
    return f"{MARKER}: {ACK_TEXT}. Citation: {CITATION}"
