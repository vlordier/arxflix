""" This module contains the JSONExporter class, which is responsible for exporting RichContent objects to a JSON file. """

import json
from pathlib import Path
from typing import List

from backend.src.models import RichContent


class JSONExporter:
    """This class is responsible for exporting RichContent objects to a JSON file.

    Attributes:
        output_path (Path): The path to the output JSON file.
    """


def export_rich_content_json(
    rich_contents: List[RichContent], output_path: Path
) -> None:
    """Export RichContent objects to a JSON file.

    Args:
        rich_contents (List[RichContent]): A list of RichContent objects to export.
        output_path (Path): The path to the output JSON file.
    """
    rich_content_dicts = [
        {
            "type": content.__class__.__name__.lower(),
            "content": content.content,
            "start": content.start,
            "end": content.end,
        }
        for content in rich_contents
    ]
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(rich_content_dicts, file, indent=4)
