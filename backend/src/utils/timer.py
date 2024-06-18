from typing import List, Union

from backend.src.models import RichContent, Text


class ContentTimer:
    """This class is responsible for filling the start and end times of the rich content objects in the script."""

    def fill_time(
        self, script_contents: List[Union[RichContent, Text]]
    ) -> List[Union[RichContent, Text]]:
        """This method fills the start and end times of the rich content objects in the script.

        Args:
            script_contents (List[Union[RichContent, Text]]): A list of rich content and text objects.

        Returns:
            List[Union[RichContent, Text]]: A list of rich content and text objects with start and end times filled.
        """
        index = 0
        while index < len(script_contents):
            current_rich_content_group = []
            while index < len(script_contents) and not isinstance(
                script_contents[index], Text
            ):
                current_rich_content_group.append(script_contents[index])
                index += 1

            if index >= len(script_contents):
                break

            next_text_group = []
            while index < len(script_contents) and isinstance(
                script_contents[index], Text
            ):
                next_text_group.append(script_contents[index])
                index += 1

            if not next_text_group:
                break

            if current_rich_content_group and next_text_group:
                total_duration = (next_text_group[-1].end or 0.0) - (
                    next_text_group[0].start or 0.0
                )
                duration_per_rich_content = total_duration / (
                    len(current_rich_content_group) + 1
                )
                offset = next_text_group[0].start or 0.0
                for _i, rich_content in current_rich_content_group:
                    rich_content.start = offset
                    rich_content.end = offset + duration_per_rich_content
                    offset += duration_per_rich_content

        return script_contents
