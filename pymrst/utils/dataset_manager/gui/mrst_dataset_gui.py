"""Interactive dataset catalog (MRST ``gui/mrstDatasetGUI.m``).

The Python port exposes a testable catalog model and a terminal UI. It can
filter, inspect, and install datasets without depending on MATLAB graphics.
"""

from dataclasses import asdict

from pymrst.utils.dataset_manager.download_dataset import download_dataset
from pymrst.utils.dataset_manager.get_available_datasets import get_available_datasets


class DatasetBrowser:
    def refresh(self):
        info, present = get_available_datasets()
        self.rows = [(item, installed) for item, installed in zip(info, present)]
        return self

    def __init__(self):
        self.rows = []
        self.refresh()

    def filter(self, text="", installed=None):
        text = text.lower().strip()
        return [(item, present) for item, present in self.rows
                if (installed is None or present is installed)
                and (not text or text in item.name.lower()
                     or text in item.description.lower()
                     or text in item.modelType.lower())]

    def details(self, name):
        for item, present in self.rows:
            if item.name.lower() == name.lower():
                return {**asdict(item), "present": present}
        raise KeyError(name)

    def install(self, name, ask_first=True):
        result = download_dataset(name, ask_first)
        self.refresh()
        return result

    def render(self, text=""):
        rows = self.filter(text)
        lines = [f"{'Dataset':28} {'Installed':9} {'Cells':>10}  Model"]
        for item, installed in rows:
            cells = "-" if item.cells is None else str(item.cells)
            lines.append(f"{item.name:28} {('yes' if installed else 'no'):9} "
                         f"{cells:>10}  {item.modelType}")
        return "\n".join(lines)


def mrst_dataset_gui(search=""):
    browser = DatasetBrowser()
    print(browser.render(search))
    return browser


mrstDatasetGUI = mrst_dataset_gui

