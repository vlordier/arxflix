from unittest.mock import MagicMock, patch

import pytest
from utils.generate_script import correct_result_link, process_script


@pytest.fixture
def test_script() -> str:
    """
    Fixture to create a test script.
    """
    return """
    This is a sample script.
    \\Figure: /figures/sample1.png
    Another line of text.
    \\Figure: /figures/sample2.png
    """


@pytest.fixture
def sample_url() -> str:
    """
    Fixture to create a sample URL.
    """
    return "https://ar5iv.labs.arxiv.org/html/1234.5678"


@pytest.fixture
def corrected_test_script() -> str:
    """
    Fixture to create a corrected test script.
    """
    return """
    This is a sample script.
    \\Figure: https://ar5iv.labs.arxiv.org/html/1234.5678/figures/sample1.png
    Another line of text.
    \\Figure: https://ar5iv.labs.arxiv.org/html/1234.5678/figures/sample2.png
    """


def test_correct_result_link(
    test_script: str, sample_url: str, corrected_test_script: str
) -> None:
    """
    Test correcting result link.

    Args:
        test_script (str): The test script.
        sample_url (str): The sample URL.
        corrected_test_script (str): The corrected test script.
    """
    with patch("requests.head") as mock_head:
        # Mock response for valid image URL
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "image/png"}
        mock_head.return_value = mock_response

        corrected = correct_result_link(test_script, sample_url)
        assert corrected.strip() == corrected_test_script.strip()


def test_process_script(
    sample_paper: str,
    sample_url: str,
    mock_openai_response: tuple[MagicMock, MagicMock],
) -> None:
    """
    Test processing script.

    Args:
        sample_paper (str): The sample paper content.
        sample_url (str): The sample URL.
        mock_openai_response (tuple[MagicMock, MagicMock]): The mocked OpenAI response.
    """
    mock_openai_client, mock_response = mock_openai_response
    generated_script = """Understood, please provide the title and content of the research paper you want the script for.

\\Headline: A New Approach to CNN Optimization with Feature Fusion
\\Text: Welcome back to Arxflix! Today, we’re diving into an innovative research paper titled "A New Approach to CNN Optimization with Feature Fusion". This study explores a fresh methodology to enhance the performance of Convolutional Neural Networks (CNNs) by leveraging feature fusion techniques.
\\Figure: https://example.com/cnn_optimization_fusion.png
\\Text: Here’s a visual representation of the proposed CNN optimization framework. The model integrates multiple feature extraction techniques to improve the overall accuracy and computational efficiency.

\\Headline: Introduction to CNN and Feature Fusion
\\Text: Convolutional Neural Networks have become a cornerstone in deep learning, especially in image and video processing tasks. However, there's always room for improvement in how these networks extract and process features.
\\Text: This paper introduces a novel feature fusion module that integrates features from different network layers, aiming to enhance the representational power of the network.

\\Headline: The Core Idea
\\Text: The core idea revolves around combining shallow and deep features within a CNN to capture both low-level and high-level semantic information. This fusion helps in mitigating the loss of spatial details which often occurs in deeper layers.
\\Figure: https://example.com/feature_fusion_diagram.png
\\Text: Here, we see the architecture of the feature fusion module. Shallow layers capture fine-grained details, while deep layers encapsulate abstract concepts. The fusion module blends these features together for a richer output.

\\Headline: Mathematical Formulation
\\Text: Let's delve into the mathematics behind this approach. The feature maps from different layers are fused using a concatenation operation, followed by a 1x1 convolution to reduce dimensionality.
\\Equation: F_fused = Conv([F_shallow, F_deep])
\\Text: In this equation, \\( F_{\\text{fused}} \\) represents the fused feature map, while \\( F_{\text{shallow}} \\) and \\( F_{\text{deep}} \\) are the feature maps from shallow and deep layers, respectively.

\\Headline: Experiment and Results
\\Text: The researchers conducted extensive experiments on popular benchmarks like CIFAR-10 and ImageNet. The results showed significant improvements in accuracy and a reduction in computational load.
\\Figure: https://example.com/experimental_results.png
\\Text: This chart compares the performance of traditional CNNs and the proposed model on the CIFAR-10 dataset. Notice the marked enhancement in accuracy and efficiency.

\\Headline: Conclusion and Future Work
\\Text: In conclusion, the paper presents a compelling case for utilizing feature fusion to optimize CNNs. It opens up new avenues for research in neural network performance tweaks.
\\Text: Future work could explore extending this methodology to other deep learning architectures and applications. The potential for combining this approach with other optimization techniques is also worth investigating.

\\Text: Thanks for watching, and don’t forget to like, share, and subscribe to Arxflix for more deep dives into cutting-edge research. See you next time!

And that wraps up our video! I hope this script serves as a helpful guide."""
    mock_response.choices[0].message.content = generated_script

    with patch("openai.OpenAI", return_value=mock_openai_client):
        result = process_script(sample_paper, sample_url)
        assert result == generated_script

        # Test for ValueError when no result is returned
        mock_response.choices[0].message.content = ""
        with pytest.raises(ValueError):
            process_script(sample_paper, sample_url)
        mock_openai_client.chat.completions.create.assert_called_once()
