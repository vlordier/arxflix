""" Testing API endpoints """
import pytest
from fastapi.testclient import TestClient

# importing the FastAPI instance from main.py
from main import api  

client = TestClient(api)

def test_generate_paper_endpoint(monkeypatch):
    def mock_process_article(url: str):
        return "Processed paper content"
    
    monkeypatch.setattr("main.process_article", mock_process_article)

    response = client.get("/generate_paper/", params={"url": "http://example.com"})
    assert response.status_code == 200
    assert response.json() == "Processed paper content"

def test_generate_script_endpoint_with_content(monkeypatch, sample_paper):
    def mock_process_script(paper_content, paper_url):
        return "Generated script content"

    monkeypatch.setattr("main.process_script", mock_process_script)
    
    response = client.post("/generate_script/", json={"paper": sample_paper, "use_path": False})
    assert response.status_code == 200
    assert response.json() == "Generated script content"

def test_generate_script_endpoint_with_path(monkeypatch, tmp_path):
    paper_file = tmp_path / "sample_paper.txt"
    paper_file.write_text("This is the paper content from file.")

    def mock_process_script(paper_content, paper_url):
        return "Generated script content"

    monkeypatch.setattr("main.process_script", mock_process_script)
    
    response = client.post("/generate_script/", json={"paper": str(paper_file), "use_path": True})
    assert response.status_code == 200
    assert response.json() == "Generated script content"

def test_generate_assets_endpoint_with_content(monkeypatch, sample_script):
    def mock_generate_audio_and_caption(script_contents, temp_dir):
        return script_contents

    def mock_export_mp3(text_content_list, mp3_output):
        pass

    def mock_export_srt(mp3_output, srt_output):
        pass

    def mock_export_rich_content_json(rich_content_list, rich_output):
        pass

    def mock_parse_script(script_content):
        return []

    def mock_fill_rich_content_time(script_contents):
        return []

    monkeypatch.setattr("main.generate_audio_and_caption", mock_generate_audio_and_caption)
    monkeypatch.setattr("main.export_mp3", mock_export_mp3)
    monkeypatch.setattr("main.export_srt", mock_export_srt)
    monkeypatch.setattr("main.export_rich_content_json", mock_export_rich_content_json)
    monkeypatch.setattr("main.parse_script", mock_parse_script)
    monkeypatch.setattr("main.fill_rich_content_time", mock_fill_rich_content_time)

    response = client.post(
        "/generate_assets/",
        json={"script": sample_script, "use_path": False}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), float)

def test_generate_assets_endpoint_with_path(monkeypatch, tmp_path):
    script_file = tmp_path / "sample_script.txt"
    script_file.write_text("This is the script content from file.")

    def mock_generate_audio_and_caption(script_contents, temp_dir):
        return script_contents

    def mock_export_mp3(text_content_list, mp3_output):
        pass

    def mock_export_srt(mp3_output, srt_output):
        pass

    def mock_export_rich_content_json(rich_content_list, rich_output):
        pass

    def mock_parse_script(script_content):
        return []

    def mock_fill_rich_content_time(script_contents):
        return []

    monkeypatch.setattr("main.generate_audio_and_caption", mock_generate_audio_and_caption)
    monkeypatch.setattr("main.export_mp3", mock_export_mp3)
    monkeypatch.setattr("main.export_srt", mock_export_srt)
    monkeypatch.setattr("main.export_rich_content_json", mock_export_rich_content_json)
    monkeypatch.setattr("main.parse_script", mock_parse_script)
    monkeypatch.setattr("main.fill_rich_content_time", mock_fill_rich_content_time)

    response = client.post(
        "/generate_assets/",
        json={"script": str(script_file), "use_path": True}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), float)

def test_generate_video_endpoint(monkeypatch, tmp_path):
    output_file = tmp_path / "output.mp4"
    
    def mock_process_video(output_path, composition_props):
        pass

    monkeypatch.setattr("main.process_video", mock_process_video)

    response = client.post("/generate_video/", json={"output_path": str(output_file)})
    assert response.status_code == 200
