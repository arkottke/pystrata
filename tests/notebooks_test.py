# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
# Copyright (C) Albert Kottke, 2013-2019
import pathlib

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

root = pathlib.Path(__file__).parent / "../examples"
fpaths = sorted(root.glob("*.ipynb"))


def idfn(val):
    if isinstance(val, pathlib.Path):
        return val.name


@pytest.mark.parametrize("fpath", fpaths, ids=idfn)
def test_notebook(fpath):
    """Execute each notebook, without modifying it."""
    # "python3" is the kernel that ipykernel provides for the interpreter running
    # the tests, so it does not depend on the kernels installed by the user or on
    # the kernelspec stored in the notebook. The executed copy is discarded.
    # https://nbclient.readthedocs.io/en/latest/client.html
    with fpath.open(encoding="utf-8") as fp:
        nb = nbformat.read(fp, as_version=4)

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(fpath.parent)}},
    )
    try:
        client.execute()
    except CellExecutionError as err:
        pytest.fail(f'Error executing the notebook "{fpath.name}":\n{err}')
