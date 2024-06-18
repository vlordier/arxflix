from typing import List, Union

from backend.src.models import RichContent, Text


class ScriptParser:
    def __init__(self, script: str):
        self.script = script

    def parse(self) -> List[Union[RichContent, Text]]:
        # Placeholder implementation for parsing the script
        # This should be replaced with the actual parsing logic
        return []


def parse_script(script: str) -> List[Union[RichContent, Text]]:
    parser = ScriptParser(script)
    return parser.parse()
