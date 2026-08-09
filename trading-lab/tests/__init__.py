"""The test suite, as a package.

It is a package rather than a loose directory for one practical reason:
`python3 -m unittest discover -s tests -t .` - the command in the README -
requires the start directory to be importable from the top level. Making it a
package also keeps `tradelab` importable from the repository root without any
path manipulation inside the test files, so nothing here depends on an install
step or on a `sys.path` hack that could accidentally pick up a different copy
of the package.

Everything is stdlib `unittest`. No pytest, no fixtures library, no plugins -
the same rule the package itself follows, for the same reason: a test suite you
cannot run on a bare interpreter is a test suite that eventually stops being
run.
"""

from __future__ import annotations
