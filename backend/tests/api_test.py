""" Test module for the API endpoints """

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from main import app, generate_assets_api, generate_paper, generate_script_api

client = TestClient(app)


# Mocking dependencies
def mock_process_article(url: str) -> str:
    """Mock function to process article content.

    Args:
        url (str): URL of the article to process.

    Returns:
        str: Processed content of the article.
    """
    return f"Processed content of {url}"


def mock_process_script(paper_content: str, paper_url: str) -> str:
    """Mock function to generate script from paper content.

    Args:
        paper_content (str): Content of the paper.
        paper_url (str): URL of the paper.

    Returns:
        str: Generated script content.
    """
    return f"Generated script from paper content: {paper_content}"


def mock_generate_audio_and_caption(script_contents, temp_dir=None):
    """Mock function to generate audio and caption from script contents.

    Args:
        script_contents (str): Contents of the script.
        temp_dir (str, optional): Temporary directory for file generation.

    Returns:
        str: Script contents.
    """
    return script_contents


def mock_export_mp3(text_content_list, mp3_output):
    """Mock function to export MP3 file.

    Args:
        text_content_list (list): List of text contents.
        mp3_output (str): Output MP3 file path.
    """
    pass


def mock_export_srt(mp3_output, srt_output):
    """Mock function to export SRT file.

    Args:
        mp3_output (str): MP3 file path.
        srt_output (str): Output SRT file path.
    """
    pass


def mock_export_rich_content_json(rich_content_list: list, rich_output: str) -> None:
    """Mock function to export rich content JSON file.

    Args:
        rich_content_list (list): List of rich content.
        rich_output (str): Output JSON file path.
    """
    pass


# Patching the functions in the test module
app.dependency_overrides[generate_paper] = mock_process_article
app.dependency_overrides[generate_script_api] = mock_process_script
app.dependency_overrides[generate_assets_api] = {
    "generate_audio_and_caption": mock_generate_audio_and_caption,
    "export_mp3": mock_export_mp3,
    "export_srt": mock_export_srt,
    "export_rich_content_json": mock_export_rich_content_json,
}


@patch("utils.generate_paper.requests.get")
def test_generate_paper(mock_get: Mock) -> None:
    """Test the generate_paper API endpoint.

    Args:
        mock_get (Mock): Mock for requests.get.
    """
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = (
        "<html><body><article>Test Article</article></body></html>"
    )

    response = client.get(
        "/generate_paper/", params={"url": "https://example.com/paper"}
    )
    assert response.status_code == 200
    assert (
        response.json()
        == '\\Headline: Attention is All You Need\\n\\Text: Welcome back to Arxflix! Today, we’re diving into an influential research paper titled "Attention is All You Need". This seminal work has revolutionized the field of natural language processing by introducing the Transformer model, which relies solely on attention mechanisms.\\n\\Figure: http://nlp.seas.harvard.edu/images/transformer_illustration.png\\n\\Text: Here’s an illustration of the Transformer architecture, fundamentally changing how we process sequence data. Let’s understand the key components and innovations that make this model so powerful.\\n\\Text: The Transformer model consists of an encoder and a decoder, both built with self-attention layers and feed-forward neural networks. The encoder processes the input sequence, while the decoder generates the output sequence. This architecture allows for parallelization and efficient handling of long-range dependencies.\\n\\Text: One of the core concepts in the Transformer is the self-attention mechanism, which enables the model to weigh the importance of different words in a sentence. This is achieved through the use of query, key, and value vectors. The self-attention mechanism computes a weighted sum of the values, where the weights are determined by the similarity between the query and key vectors.\\n\\Text: Another key innovation is the use of positional encodings, which provide information about the position of words in a sequence. This is crucial because the Transformer does not have a built-in notion of word order. Positional encodings are added to the input embeddings, allowing the model to capture the sequential nature of the data.\\n\\Text: The Transformer model also employs multi-head attention, which allows the model to focus on different parts of the input sequence simultaneously. This is achieved by splitting the query, key, and value vectors into multiple heads, each of which performs self-attention independently. The results are then concatenated and linearly transformed to produce the final output.\\n\\Text: In addition to self-attention, the Transformer uses feed-forward neural networks to process the output of the attention layers. These networks consist of two linear transformations with a ReLU activation in between. This allows the model to capture complex patterns and relationships in the data.\\n\\Text: The Transformer model has achieved state-of-the-art performance on a wide range of natural language processing tasks, including machine translation, text summarization, and sentiment analysis. Its ability to handle long-range dependencies and parallelize computations has made it a popular choice for researchers and practitioners alike.\\n\\Text: In conclusion, the "Attention is All You Need" paper has had a profound impact on the field of natural language processing. The Transformer model introduced in this paper has revolutionized the way we approach sequence-to-sequence tasks, and its innovations continue to inspire new research and applications.\\n\\Text: Thank you for joining us on this journey through the Transformer model. Stay tuned for more deep dives into groundbreaking research in the world of machine learning and AI.\\n\\Text: Goodbye!'
    )


def test_generate_script_api(script_input) -> None:
    """Test the generate_script API endpoint.

    Args:
        script_input (object): Input data for the script generation.
    """
    response = client.post("/generate_script/", json=script_input.dict())
    assert response.status_code == 200
    assert (
        response.json()
        == '\\Headline: Attention is All You Need\\n\\Text: Welcome back to Arxflix! Today, we’re diving into an influential research paper titled "Attention is All You Need". This seminal work has revolutionized the field of natural language processing by introducing the Transformer model, which relies solely on attention mechanisms.\\n\\Figure: http://nlp.seas.harvard.edu/images/transformer_illustration.png\\n\\Text: Here’s an illustration of the Transformer architecture, fundamentally changing how we process sequence data. Let’s understand the key components and innovations that make this model so powerful.\\n\\Text: The Transformer model consists of an encoder and a decoder, both built with self-attention layers and feed-forward neural networks. The encoder processes the input sequence, while the decoder generates the output sequence. This architecture allows for parallelization and efficient handling of long-range dependencies.\\n\\Text: One of the core concepts in the Transformer is the self-attention mechanism, which enables the model to weigh the importance of different words in a sentence. This is achieved through the use of query, key, and value vectors. The self-attention mechanism computes a weighted sum of the values, where the weights are determined by the similarity between the query and key vectors.\\n\\Text: Another key innovation is the use of positional encodings, which provide information about the position of words in a sequence. This is crucial because the Transformer does not have a built-in notion of word order. Positional encodings are added to the input embeddings, allowing the model to capture the sequential nature of the data.\\n\\Text: The Transformer model also employs multi-head attention, which allows the model to focus on different parts of the input sequence simultaneously. This is achieved by splitting the query, key, and value vectors into multiple heads, each of which performs self-attention independently. The results are then concatenated and linearly transformed to produce the final output.\\n\\Text: In addition to self-attention, the Transformer uses feed-forward neural networks to process the output of the attention layers. These networks consist of two linear transformations with a ReLU activation in between. This allows the model to capture complex patterns and relationships in the data.\\n\\Text: The Transformer model has achieved state-of-the-art performance on a wide range of natural language processing tasks, including machine translation, text summarization, and sentiment analysis. Its ability to handle long-range dependencies and parallelize computations has made it a popular choice for researchers and practitioners alike.\\n\\Text: In conclusion, the "Attention is All You Need" paper has had a profound impact on the field of natural language processing. The Transformer model introduced in this paper has revolutionized the way we approach sequence-to-sequence tasks, and its innovations continue to inspire new research and applications.\\n\\Text: Thank you for joining us on this journey through the Transformer model. Stay tuned for more deep dives into groundbreaking research in the world of machine learning and AI.\\n\\Text: Goodbye!'
    )


def test_generate_assets_api(assets_input):
    """Test the generate_assets API endpoint.

    Args:
        assets_input (object): Input data for asset generation.
    """
    response = client.post("/generate_assets/", json=assets_input.dict())
    assert response.status_code == 200
    assert (
        response.json()
        == "Generated script from paper content: This is a sample paper content"
    )


# Reset the dependency overrides after tests
app.dependency_overrides = {}
