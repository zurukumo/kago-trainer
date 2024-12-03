from __future__ import annotations

import os
import re
from typing import Generator


class HaihuLoader:
    root_dir: str

    __slots__ = ("root_dir",)

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir

    def __iter__(self) -> Generator[HaihuItem, None, None]:
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith(".xml"):
                    yield HaihuItem(os.path.join(dirpath, filename))

    def __len__(self) -> int:
        count = 0
        for _, _, filenames in os.walk(self.root_dir):
            count += len([filename for filename in filenames if filename.endswith(".xml")])
        return count


class HaihuItem:
    id: str
    tags: list[tuple[str, dict[str, str]]]

    __slots__ = ("id", "tags")

    def __init__(self, filepath: str) -> None:
        self.id = os.path.basename(filepath).replace(".xml", "")
        with open(filepath, "r") as fp:
            content = fp.read()
            self.tags = HaihuItem.parse_xml(content)

    @classmethod
    def parse_xml(self, xml: str) -> list[tuple[str, dict[str, str]]]:
        tags = []
        for elem, attr in re.findall(r"<(.+?)[ /](.*?)/?>", xml):
            attr = dict(re.findall(r'\s?(.+?)="(.*?)"', attr))
            tags.append((elem, attr))
        return tags
