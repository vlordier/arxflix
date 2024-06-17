from unittest.mock import MagicMock, patch

import pytest
from utils.generate_script import correct_result_link, process_script


@pytest.fixture
def test_script():
    return """
    This is a sample script.
    \\Figure: /figures/sample1.png
    Another line of text.
    \\Figure: /figures/sample2.png
    """

@pytest.fixture
def sample_url():
    return "https://ar5iv.labs.arxiv.org/html/1234.5678"

@pytest.fixture
def corrected_test_script():
    return """
    This is a sample script.
    \\Figure: https://ar5iv.labs.arxiv.org/html/1234.5678/figures/sample1.png
    Another line of text.
    \\Figure: https://ar5iv.labs.arxiv.org/html/1234.5678/figures/sample2.png
    """

def test_correct_result_link(test_script, sample_url, corrected_test_script):
    with patch("requests.head") as mock_head:
        # Mock response for valid image URL
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "image/png"}
        mock_head.return_value = mock_response

        corrected = correct_result_link(test_script, sample_url)
        assert corrected == corrected_test_script


def test_process_script(sample_paper, sample_url, mock_openai_response):
    mock_openai_client, mock_response = mock_openai_response
    generated_script = "\\Headline: Sample Paper Research Overview\n\\Text: Welcome back to Arxflix! Today, we’re exploring an insightful research paper, \"Sample Paper Research Overview.\" This study delved into an advanced topic within the field of deep learning, proposing a novel approach to optimize neural network performance.\n\n\\Headline: Introduction\n\\Text: To start, the authors of this paper address the limitations of current deep learning models, specifically how they struggle with scalability and efficiency. They highlight the necessity for more robust architectures that can handle larger datasets and more complex tasks.\n\n\\Headline: Methodology\n\\Text: The core of their approach involves a multi-layered neural network with enhanced feature extraction capabilities. They introduce a new algorithm that significantly reduces training time while maintaining high accuracy.\n\n\\Headline: Results\n\\Text: The results are promising, showing a marked improvement in both speed and accuracy compared to existing models. The authors provide detailed benchmarks and comparisons to illustrate the effectiveness of their method.\n\n\\Headline: Conclusion\n\\Text: In conclusion, this paper presents a significant advancement in the field of deep learning. The proposed method not only improves performance but also opens new avenues for future research. Stay tuned for more updates and insights from the frontiers of AI!"
    mock_response.choices[0].message.content = generated_script

    with patch("openai.OpenAI", return_value=mock_openai_client):
        result = process_script(sample_paper, sample_url)
        assert result == generated_script

        # Test for ValueError when no result is returned
        mock_response.choices[0].message.content = None
        with pytest.raises(ValueError):
            process_script(sample_paper, sample_url)
        mock_openai_client.chat.completions.create.assert_called_once()

        with pytest.raises(ValueError):
            mock_response.choices[0].message.content = None
            process_script(sample_paper, sample_url)
