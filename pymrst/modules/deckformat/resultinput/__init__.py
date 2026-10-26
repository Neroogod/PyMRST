"""Reading ECLIPSE *output* (MRST `deckformat/resultinput`).

A deck says what to simulate; these files say what a simulator produced - and
for a field model the EGRID/INIT pair is also the only place the simulator's
own processed grid and transmissibilities exist.
"""

from pymrst.modules.deckformat.resultinput.read_eclipse_output_file_un_fmt import (
    readEclipseOutputFileUnFmt, read_eclipse_output_file_un_fmt,
)
from pymrst.modules.deckformat.resultinput.read_eclipse_summary_un_fmt import (
    readEclipseSummaryUnFmt, read_eclipse_summary_un_fmt,
)
from pymrst.modules.deckformat.resultinput.init_grid_from_eclipse_output import (
    initGridFromEclipseOutput, init_grid_from_eclipse_output,
)

__all__ = ["initGridFromEclipseOutput", "init_grid_from_eclipse_output",
           "readEclipseOutputFileUnFmt", "read_eclipse_output_file_un_fmt",
           "readEclipseSummaryUnFmt", "read_eclipse_summary_un_fmt"]
