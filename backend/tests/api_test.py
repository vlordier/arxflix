""" Testing API endpoints """

import pathlib

import pytest
from fastapi.testclient import TestClient

# Importing the FastAPI instance from main.py
from main import api

client = TestClient(api)


def test_generate_paper_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test generating paper endpoint.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture to mock the process_article function.
    """

    def mock_process_article(url: str) -> str:
        """Mock the process_article function to return a fixed value

        Args:
            url (str): The URL.

        Returns:
            str: The processed paper content.
        """
        return "Processed paper content"

    monkeypatch.setattr("main.process_article", mock_process_article)

    response = client.get("/generate_paper/", params={"url": "http://example.com"})
    assert response.status_code == 200
    assert response.json() == "Processed paper content"


def test_generate_script_endpoint_with_content(
    monkeypatch: pytest.MonkeyPatch, sample_paper: str
) -> None:
    """
    Test generating script endpoint with content.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture to mock the process_script function.
        sample_paper (str): The sample paper content.
    """

    # Mock the process_script function to return a fixed value
    def mock_process_script(paper_content: str, paper_url: str) -> str:
        """Mock the process_script function to return a fixed value

        Args:
            paper_content (str): The paper content.
            paper_url (str): The paper URL.

        Returns:
            str: The generated script content.
        """

        return "Generated script content"

    monkeypatch.setattr("main.process_script", mock_process_script)

    response = client.post(
        "/generate_script/", json={"paper": sample_paper, "use_path": False}
    )
    assert response.status_code == 200
    assert response.json() == "Generated script content"


def test_generate_script_endpoint_with_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """
    Test generating script endpoint with path.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture to mock the process_script function.
        tmp_path (pathlib.Path): Temporary path for the paper file.
    """
    paper_file = tmp_path / "sample_paper.txt"
    paper_file.write_text("This is the paper content from file.")

    def mock_process_script(paper_content: str, paper_url: str) -> str:
        """Mock the process_script function to return a fixed value

        Args:
            paper_content (str): The paper content.
            paper_url (str): The paper URL.

        Returns:
            str: The generated script content.
        """
        return "Generated script content"

    monkeypatch.setattr("main.process_script", mock_process_script)

    response = client.post(
        "/generate_script/", json={"paper": str(paper_file), "use_path": True}
    )
    assert response.status_code == 200
    assert response.json() == "Generated script content"


def test_generate_assets_endpoint_with_content(
    monkeypatch: pytest.MonkeyPatch, sample_script: str
) -> None:
    """
    Test generating assets endpoint with content.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture to mock the generate_assets functions.
        sample_script (str): The sample script content.
    """

    def mock_generate_audio_and_caption(script_contents: list, temp_dir: str) -> list:
        """Mock the generate_audio_and_caption function to return a fixed value

        Args:
            script_contents (list): The script contents.
            temp_dir (str): The temporary directory.

        Returns:
            list: The generated audio and caption.
        """
        return script_contents

    def mock_export_mp3(text_content_list: list, mp3_output: str) -> None:
        """
        Mock the export_mp3 function to return a fixed value.

        Args:
            text_content_list (list): The text content list.
            mp3_output (str): The MP3 output file.
        """
        pass

    def mock_export_srt(mp3_output: str, srt_output: str) -> None:
        """
        Mock the export_srt function to return a fixed value.

        Args:
            mp3_output (str): The MP3 output file.
            srt_output (str): The SRT output file.
        """

        pass

    def mock_export_rich_content_json(
        rich_content_list: list, rich_output: str
    ) -> None:
        """
        Mock the export_rich_content_json function to return a fixed value.

        Args:
            rich_content_list (list): The rich content list.
            rich_output (str): The rich content output file.
        """

        pass

    def mock_parse_script(script_content: str) -> list:
        """Mock the parse_script function to return a fixed value
        Args:
            script_content (str): The script content.

        Returns:
            list: The parsed script.
        """

        return []

    def mock_fill_rich_content_time(script_contents: list) -> list:
        """
        Mock the fill_rich_content_time function to return a fixed value.

        Args:
            script_contents (list): The script contents.

        Returns:
            list: The filled rich content time.
        """

        return []

    monkeypatch.setattr(
        "main.generate_audio_and_caption", mock_generate_audio_and_caption
    )
    monkeypatch.setattr("main.export_mp3", mock_export_mp3)
    monkeypatch.setattr("main.export_srt", mock_export_srt)
    monkeypatch.setattr("main.export_rich_content_json", mock_export_rich_content_json)
    monkeypatch.setattr("main.parse_script", mock_parse_script)
    monkeypatch.setattr("main.fill_rich_content_time", mock_fill_rich_content_time)

    response = client.post(
        "/generate_assets/", json={"script": sample_script, "use_path": False}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), float)


def test_generate_assets_endpoint_with_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """
    Test generating assets endpoint with path.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture to mock the generate_assets functions.
        tmp_path (pathlib.Path): Temporary path for the script file.
    """
    script_file = tmp_path / "sample_script.txt"
    script_file.write_text("This is the script content from file.")

    def mock_generate_audio_and_caption(script_contents: list, temp_dir: str) -> list:
        """Mock the generate_audio_and_caption function to return a fixed value

        Args:
            script_contents (list): The script contents.
            temp_dir (str): The temporary directory.

        Returns:
            list: The generated audio and caption.
        """
        return script_contents

    def mock_export_mp3(text_content_list: list, mp3_output: str) -> None:
        """Mock the export_mp3 function to return a fixed value

        Args:
            text_content_list (list): The text content list.
            mp3_output (str): The MP3 output file.
        """

        pass

    def mock_export_srt(mp3_output: str, srt_output: str) -> None:
        """Mock the export_srt function to return a fixed value

        Args:
            mp3_output (str): The MP3 output file.
            srt_output (str): The SRT output file.
        """

        pass

    def mock_export_rich_content_json(
        rich_content_list: list, rich_output: str
    ) -> None:
        """Mock the export_rich_content_json function to return a fixed value

        Args:
            rich_content_list (list): The rich content list.
            rich_output (str): The rich content output file.
        """
        pass

    def mock_parse_script(script_content: str) -> list:
        """Mock the parse_script function to return a fixed value

        Args:
            script_content (str): The script content.

        Returns:
            list: The parsed script.
        """

        return []

    def mock_fill_rich_content_time(script_contents: list) -> list:
        """Mock the fill_rich_content_time function to return a fixed value

        Args:
            script_contents (list): The script contents.

        Returns:
            list: The filled rich content time.
        """

        return []

    monkeypatch.setattr(
        "main.generate_audio_and_caption", mock_generate_audio_and_caption
    )
    monkeypatch.setattr("main.export_mp3", mock_export_mp3)
    monkeypatch.setattr("main.export_srt", mock_export_srt)
    monkeypatch.setattr("main.export_rich_content_json", mock_export_rich_content_json)
    monkeypatch.setattr("main.parse_script", mock_parse_script)
    monkeypatch.setattr("main.fill_rich_content_time", mock_fill_rich_content_time)

    response = client.post(
        "/generate_assets/", json={"script": str(script_file), "use_path": True}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), float)


def test_generate_video_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """
    Test generating video endpoint.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture to mock the process_video function.
        tmp_path (pathlib.Path): Temporary path for the output file.
    """
    output_file = tmp_path / "output.mp4"

    def mock_process_video(output_path: str, composition_props: dict) -> None:
        """Mock the process_video function to return a fixed value

        Args:
            output_path (str): The output file path.
            composition_props (dict): The composition properties.
        """

        pass

    monkeypatch.setattr("main.process_video", mock_process_video)

    response = client.post("/generate_video/", json={"output_path": str(output_file)})
    assert response.status_code == 200
