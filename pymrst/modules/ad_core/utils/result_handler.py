"""Store and retrieve simulation results (MRST `ad-core/utils/ResultHandler.m`).

A long run produces one state per report step, and holding them all in memory
is the thing that stops a field-scale schedule finishing. MRST's answer is a
store that behaves roughly like a list but can keep its contents on disk,
retrieve a step by index or by simulation time, and resume from whatever it
already has.

Three decisions carried over from MRST, and one forced by Python:

* **Indices are the ids MRST writes into filenames**, so step 1 is
  `state1.pkl` and `handler[1]` returns it. Renumbering to Python's 0-based
  convention would mean the index in code and the number in the filename
  disagree, which is exactly the confusion this class exists to avoid on a
  restart.
* **Memory and disk are independent.** A handler may do both, either, or
  neither; `store_in_memory=False, write_to_disk=True` is the configuration
  that makes a long run possible, and it is not the default because a silent
  write to disk is worse than an explicit one.
* **Absence and emptiness are different.** `get_valid_ids` reports what is
  actually there, with gaps preserved - a schedule that failed at step 7 has
  ids 1..6, and reporting 1..7 with a hole would be a different claim.

The Python-forced one: MRST saves `.mat`, which holds arbitrary MATLAB values.
The nearest equivalent here is `pickle`, and that carries a trust boundary
MATLAB's does not - unpickling executes code. These files are written by this
program for its own use, so that is acceptable, but it is stated rather than
assumed, and `read_from_file` refuses a path outside the handler's own
directory.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


class ResultHandler:
    """A list-like store of per-step results, in memory and/or on disk."""

    def __init__(
        self,
        data_directory: Optional[str | Path] = None,
        data_folder: str = "results",
        data_prefix: str = "state",
        write_to_disk: bool = False,
        store_in_memory: bool = True,
        extension: str = ".pkl",
        cleardir: bool = False,
        verbose: bool = False,
    ):
        self.data_directory = Path(data_directory) if data_directory else Path.cwd()
        self.data_folder = data_folder
        self.data_prefix = data_prefix
        self.write_to_disk = bool(write_to_disk)
        self.store_in_memory = bool(store_in_memory)
        self.extension = extension
        self.verbose = bool(verbose)
        self.data: dict[int, Any] = {}

        if self.write_to_disk:
            self.get_data_path().mkdir(parents=True, exist_ok=True)
            if cleardir:
                for path in self.get_data_path().glob(
                        f"{self.data_prefix}*{self.extension}"):
                    path.unlink()

    # -- paths --------------------------------------------------------------

    def get_extension(self, wildcard: bool = False) -> str:
        return f"*{self.extension}" if wildcard else self.extension

    getExtension = get_extension

    def get_data_path(self, index: Optional[int] = None) -> Path:
        """The store's directory, or the file one step lives in."""
        base = self.data_directory/self.data_folder
        if index is None:
            return base
        if not isinstance(index, (int,)) or isinstance(index, bool):
            raise TypeError(f"index must be a single integer, got {index!r}")
        return base/f"{self.data_prefix}{index}{self.extension}"

    getDataPath = get_data_path

    # -- disk ---------------------------------------------------------------

    def write_to_file(self, data: Any, index: int) -> "ResultHandler":
        path = self.get_data_path(index)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(data, stream)
        if self.verbose:
            print(f"writing data to {path}")
        return self

    writeToFile = write_to_file

    def read_from_file(self, indices: int | Sequence[int]) -> List[Any]:
        """Read one or several steps back from disk.

        Refuses a path that escapes the handler's own directory: the index
        reaches a filename, and a caller passing something unexpected should
        get an error rather than a read from somewhere else.
        """
        if isinstance(indices, int):
            indices = [indices]
        out = []
        root = self.get_data_path().resolve()
        for index in indices:
            path = self.get_data_path(index).resolve()
            if root not in path.parents:
                raise ValueError(
                    f"{path} is outside the handler's directory {root}")
            if not path.exists():
                raise KeyError(f"no stored result for index {index}")
            with path.open("rb") as stream:
                out.append(pickle.load(stream))
        return out

    readFromFile = read_from_file

    def delete_file(self, index: int) -> "ResultHandler":
        path = self.get_data_path(index)
        if path.exists():
            path.unlink()
            if self.verbose:
                print(f"deleting data at {path}")
        return self

    deleteFile = delete_file

    # -- what is stored -----------------------------------------------------

    def get_valid_ids(self) -> List[int]:
        """Every index that actually holds a result, ascending.

        Gaps are preserved. A schedule that failed at step 7 has ids 1..6, and
        reporting a contiguous range would claim something that is not true.
        """
        if self.write_to_disk:
            pattern = re.compile(
                rf"^{re.escape(self.data_prefix)}(\d+){re.escape(self.extension)}$")
            directory = self.get_data_path()
            if not directory.is_dir():
                return []
            found = [int(m.group(1)) for m in
                     (pattern.match(p.name) for p in directory.iterdir())
                     if m]
            return sorted(found)
        return sorted(self.data)

    getValidIds = get_valid_ids

    def numel_data(self) -> int:
        ids = self.get_valid_ids()
        return max(ids) if ids else 0

    numelData = numel_data

    def size_data(self, dim: Optional[int] = None):
        """MRST's `sizeData`: `(n, 1)`, or one of those two."""
        shape = (self.numel_data(), 1)
        return shape if dim is None else shape[dim - 1]

    sizeData = size_data

    def __len__(self) -> int:
        return len(self.get_valid_ids())

    def __iter__(self) -> Iterator[Any]:
        for index in self.get_valid_ids():
            yield self[index]

    # -- element access -----------------------------------------------------

    def get_data(self, indices: Optional[Sequence[int]] = None) -> List[Any]:
        if indices is None:
            indices = self.get_valid_ids()
        return [self[i] for i in indices]

    getData = get_data

    def __getitem__(self, index: int) -> Any:
        if self.store_in_memory and index in self.data:
            return self.data[index]
        if self.write_to_disk:
            return self.read_from_file(index)[0]
        raise KeyError(f"no stored result for index {index}")

    def __setitem__(self, index: int, value: Any) -> None:
        if not self.store_in_memory and not self.write_to_disk:
            raise ValueError(
                "this handler neither stores in memory nor writes to disk, so "
                "the value would be discarded; set store_in_memory or "
                "write_to_disk")
        if self.store_in_memory:
            self.data[index] = value
        if self.write_to_disk:
            self.write_to_file(value, index)

    def __delitem__(self, index: int) -> None:
        self.data.pop(index, None)
        if self.write_to_disk:
            self.delete_file(index)

    def __contains__(self, index: int) -> bool:
        return index in self.get_valid_ids()

    def reset(self) -> "ResultHandler":
        """Drop everything, in memory and on disk."""
        for index in list(self.get_valid_ids()):
            del self[index]
        self.data.clear()
        return self

    def reset_data(self, subs: Optional[Sequence[int] | np.ndarray] = None) -> None:
        """MRST `resetData`: drop everything, or the ids (or mask) in `subs`.

        Without `subs`, memory is emptied and every `<dataPrefix>*<extension>`
        file deleted, as MRST deletes `<dataPrefix>*.mat`. With `subs`, those
        ids are dropped from memory and their files deleted; a logical mask
        selects the ids `find(subs)`.
        """
        if subs is None:
            self.data.clear()
            if self.write_to_disk:
                for path in self.get_data_path().glob(
                        f"{self.data_prefix}{self.get_extension(True)}"):
                    path.unlink()
            return
        subs = np.atleast_1d(np.asarray(subs))
        if subs.dtype == bool:
            subs = np.flatnonzero(subs) + 1
        for index in subs.astype(int).tolist():
            if self.store_in_memory:
                self.data.pop(index, None)
            if self.write_to_disk:
                self.delete_file(index)

    resetData = reset_data

    # -- searching ----------------------------------------------------------

    def get_first_by_function(self, predicate: Callable[[Any], bool],
                              reverse: bool = False) -> Tuple[Any, Optional[int]]:
        """The first stored result satisfying `predicate`, and its index.

        `(None, None)` when nothing matches, rather than raising: a caller
        asking "is there a step like this" wants an answer, not an exception.
        """
        indices = self.get_valid_ids()
        if reverse:
            indices = indices[::-1]
        for index in indices:
            candidate = self[index]
            if predicate(candidate):
                return candidate, index
        return None, None

    getFirstByFunction = get_first_by_function

    def get_by_time(self, time: float, tol: float = 1e-3,
                    reverse: bool = False) -> Tuple[Any, Optional[int]]:
        """The first result whose `time` is within `tol` of `time`."""
        return self.get_first_by_function(
            lambda state: abs(_time_of(state) - time) < tol, reverse)

    getByTime = get_by_time

    def get_closest_by_time(self, time: float, reverse: bool = False):
        """The stored result nearest `time`. Returns `(state, index, dist)`.

        Distinct from `get_by_time`, which can legitimately find nothing: this
        always answers, and the returned distance is what tells the caller
        whether the answer is any good.
        """
        best = (None, None, float("inf"))
        indices = self.get_valid_ids()
        if reverse:
            indices = indices[::-1]
        for index in indices:
            candidate = self[index]
            distance = abs(_time_of(candidate) - time)
            if distance < best[2]:
                best = (candidate, index, distance)
        return best

    getClosestByTime = get_closest_by_time

    def get_restart(self) -> Tuple[Any, int]:
        """`(last stored state, the step to resume at)`.

        `(None, 1)` on an empty store, so a caller can start a fresh run and a
        resumed one through the same code path - which is the point of the
        method.
        """
        ids = self.get_valid_ids()
        if not ids:
            return None, 1
        last = max(ids)
        return self[last], last + 1

    getRestart = get_restart

    def __repr__(self) -> str:
        where = []
        if self.store_in_memory:
            where.append("memory")
        if self.write_to_disk:
            where.append(str(self.get_data_path()))
        return (f"ResultHandler({len(self)} result(s) in "
                f"{' + '.join(where) or 'nothing'})")


def _time_of(state: Any) -> float:
    """The simulation time of a stored result, however it is shaped."""
    if isinstance(state, dict):
        if "time" not in state:
            raise KeyError("stored result has no 'time' field to search on")
        return float(state["time"])
    time = getattr(state, "time", None)
    if time is None:
        raise KeyError("stored result has no 'time' field to search on")
    return float(time)


__all__ = ["ResultHandler"]


if __name__ == "__main__":
    import tempfile

    def test_memory_round_trip():
        handler = ResultHandler()
        handler[1] = {"time": 0.0, "pressure": 1.0}
        handler[2] = {"time": 10.0, "pressure": 2.0}
        assert handler[2]["pressure"] == 2.0
        assert handler.get_valid_ids() == [1, 2]
        print("ok test_memory_round_trip")

    def test_disk_round_trip():
        with tempfile.TemporaryDirectory() as tmp:
            handler = ResultHandler(tmp, write_to_disk=True,
                                    store_in_memory=False)
            handler[1] = {"time": 0.0, "x": [1, 2, 3]}
            fresh = ResultHandler(tmp, write_to_disk=True,
                                  store_in_memory=False)
            assert fresh[1]["x"] == [1, 2, 3]
            assert fresh.get_valid_ids() == [1]
        print("ok test_disk_round_trip")

    def test_restart_from_an_empty_store():
        state, step = ResultHandler().get_restart()
        assert state is None and step == 1
        print("ok test_restart_from_an_empty_store")

    def test_reset_data_all_or_some():
        with tempfile.TemporaryDirectory() as tmp:
            handler = ResultHandler(tmp, write_to_disk=True,
                                    store_in_memory=False)
            for index in range(1, 5):
                handler[index] = {"time": float(index)}
            handler.reset_data([2, 3])
            assert handler.get_valid_ids() == [1, 4]
            handler.reset_data(np.array([True, False, False, False]))
            assert handler.get_valid_ids() == [4]
            handler.reset_data()
            assert handler.get_valid_ids() == []
        memory = ResultHandler()
        memory[1], memory[2] = {"time": 0.0}, {"time": 1.0}
        memory.reset_data([1])
        assert memory.get_valid_ids() == [2]
        print("ok test_reset_data_all_or_some")

    test_memory_round_trip()
    test_disk_round_trip()
    test_reset_data_all_or_some()
    test_restart_from_an_empty_store()
    print("\nAll tests passed!")
