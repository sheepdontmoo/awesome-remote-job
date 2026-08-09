"""Makes `python3 -m tradelab ...` work without an install step.

The package has no setup.py, no entry point script and no dependencies, so
this file is the whole distribution story: clone the tree, `cd` into it, run
the module. Anything more would be one more thing between a person and the
backtest that talks them out of the strategy.

All logic lives in `cli.main`, which returns an exit code rather than calling
`sys.exit` itself, so it can be driven from a test or another program.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
