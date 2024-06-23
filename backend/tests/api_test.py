""" Test module for the API endpoints """

from fastapi.testclient import TestClient
from src.main import app, generate_assets_api, generate_paper, generate_script_api

client = TestClient(app)


# Mocking dependencies
def mock_process_article(url: str) -> str:
    return f"Processed content of {url}"


def mock_process_script(paper_content: str, paper_url: str) -> str:
    return f"Generated script from paper content: {paper_content}"


def mock_generate_audio_and_caption(script_contents, temp_dir=None):
    return script_contents


def mock_export_mp3(text_content_list, mp3_output):
    pass


def mock_export_srt(mp3_output, srt_output):
    pass


def mock_export_rich_content_json(rich_content_list, rich_output):
    pass


# Patching the functions in the test module
app.dependency_overrides[generate_paper] = mock_process_article
app.dependency_overrides[generate_script_api] = mock_process_script
app.dependency_overrides[generate_assets_api] = (
    mock_generate_audio_and_caption,
    mock_export_mp3,
    mock_export_srt,
    mock_export_rich_content_json,
)


def test_generate_paper() -> None:
    response = client.get(
        "/generate_paper/", params={"url": "http://example.com/paper"}
    )
    assert response.status_code == 200
    assert response.json() == "Processed content of http://example.com/paper"


def test_generate_script_api(script_input) -> None:
    response = client.post("/generate_script/", json=script_input.dict())
    assert response.status_code == 200
    assert (
        response.json()
        == "Generated script from paper content: This is a sample paper content"
    )


def test_generate_assets_api(assets_input):
    response = client.post("/generate_assets/", json=assets_input.dict())
    assert response.status_code == 200
    assert response.json() == 0.0  # Mocked function returns duration as 0


# Reset the dependency overrides after tests
app.dependency_overrides = {}
