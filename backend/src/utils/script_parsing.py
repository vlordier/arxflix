""" This module contains the ScriptParser class, which is responsible for parsing the script and extracting the content. """

import logging
from typing import List, Union

from backend.src.models import Equation, Figure, Headline, RichContent, Text

logger = logging.getLogger(__name__)


class ScriptParser:
    def __init__(self, script: str):
        self.script = script

    def parse(self) -> List[Union[RichContent, Text, Figure, Equation, Headline]]:
        """Parses the script and returns a list of RichContent objects.

        Returns:
            List[Union[RichContent, Text, Figure, Equation, Headline]]: A list of RichContent objects.
        """
        lines = self.script.split("\n")
        content: List[Union[RichContent, Text, Figure, Equation, Headline]] = []
        content_mapping = {
            "\\Figure:": Figure,
            "\\Text:": Text,
            "\\Equation:": Equation,
            "\\Headline:": Headline,
        }

        for line in lines:
            if line.strip() == "":
                continue
            for prefix, cls in content_mapping.items():
                if line.startswith(prefix):
                    content_value = line[len(prefix) :].strip()
                    if cls in [Figure, Equation]:
                        content.append(
                            cls(
                                content=content_value,
                                start=0.0,
                                end=0.5,
                                audio=None,
                                captions=None,
                            )
                        )
                    else:
                        content.append(cls(content=content_value))
                    break
            else:
                logger.warning("Unknown line: %s", line)

        return content


def parse_script(
    script: str,
) -> List[Union[RichContent, Text, Figure, Equation, Headline]]:
    parser = ScriptParser(script)
    return parser.parse()
