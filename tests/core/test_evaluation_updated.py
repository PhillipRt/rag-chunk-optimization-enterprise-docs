"""
Tests for the RAG evaluation module.
"""
import pytest
import os
from unittest.mock import patch, MagicMock

# Mock the protobuf implementation before any Google imports
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from core.evaluation import RagEvaluator


@pytest.fixture
def evaluation_config():
    """Return a configuration for the RagEvaluator."""
    return {
        "llm_model": "deepseek-chat",
        "cache_dir": "cache/test_ragas",
        "metrics": ["nv_accuracy", "answer_correctness", "factual_correctness", "faithfulness", "context_precision", "context_recall"]
    }


@pytest.fixture
def evaluation_data():
    """Return test evaluation data."""
    return {
        "questions": [
            "What is Python programming language?",
            "How does JavaScript relate to web development?"
        ],
        "answers": [
            "Python is a high-level, general-purpose programming language known for its readability.",
            "JavaScript is one of the core technologies of the web, used for client-side interactivity."
        ],
        "contexts": [
            [
                "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.",
                "Python is dynamically-typed and garbage-collected."
            ],
            [
                "JavaScript, often abbreviated as JS, is a programming language that is one of the core technologies of the World Wide Web, alongside HTML and CSS.",
                "98% of websites use JavaScript on the client side for webpage behavior."
            ]
        ],
        "ground_truths": [
            "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability.",
            "JavaScript is one of the core technologies of the World Wide Web, alongside HTML and CSS. It is used for client-side webpage behavior."
        ]
    }


@patch('core.evaluation.LangchainLLMWrapper')
@patch('core.evaluation.LangchainEmbeddingsWrapper')
@patch('core.evaluation.AnswerAccuracy')
@patch('core.evaluation.AnswerCorrectness')
@patch('core.evaluation.FactualCorrectness')
@patch('core.evaluation.Faithfulness')
@patch('core.evaluation.ContextPrecision')
@patch('core.evaluation.ContextRecall')
@patch('core.evaluation.evaluate')
def test_evaluator_setup(mock_evaluate, mock_cr, mock_cp, mock_f, mock_fc, mock_ac, mock_aa,
                        mock_embeddings_wrapper, mock_llm_wrapper, evaluation_config):
    """Test that the evaluator can be set up correctly."""
    # Mock the LLM and embeddings
    mock_llm = MagicMock()
    mock_llm_wrapper.return_value = mock_llm

    mock_embeddings = MagicMock()
    mock_embeddings_wrapper.return_value = mock_embeddings

    # Mock metrics
    mock_aa.return_value = "answer_accuracy_metric"
    mock_ac.return_value = "answer_correctness_metric"
    mock_fc.return_value = "factual_correctness_metric"
    mock_f.return_value = "faithfulness_metric"
    mock_cp.return_value = "context_precision_metric"
    mock_cr.return_value = "context_recall_metric"

    # Create the evaluator
    evaluator = RagEvaluator(evaluation_config)

    # Set up the evaluator (mocking the _setup_llm and _setup_embeddings methods)
    with patch.object(evaluator, '_setup_llm'), patch.object(evaluator, '_setup_embeddings'):
        evaluator.setup()

    # Check that metrics were set up correctly
    assert evaluator.metrics is not None
    assert len(evaluator.metrics) == 6


@patch('core.evaluation.LangchainLLMWrapper')
@patch('core.evaluation.LangchainEmbeddingsWrapper')
@patch('core.evaluation.AnswerAccuracy')
@patch('core.evaluation.AnswerCorrectness')
@patch('core.evaluation.FactualCorrectness')
@patch('core.evaluation.Faithfulness')
@patch('core.evaluation.ContextPrecision')
@patch('core.evaluation.ContextRecall')
@patch('core.evaluation.evaluate')
@patch('core.evaluation.EvaluationDataset')
def test_evaluator_evaluate(mock_dataset, mock_evaluate, mock_cr, mock_cp, mock_f, mock_fc, mock_ac, mock_aa,
                          mock_embeddings_wrapper, mock_llm_wrapper, evaluation_config, evaluation_data):
    """Test that the evaluator can evaluate RAG output."""
    # Mock the evaluation dataset
    mock_dataset_instance = MagicMock()
    mock_dataset.return_value = mock_dataset_instance

    # Mock the evaluate function to return a DataFrame with results
    import pandas as pd
    mock_results = pd.DataFrame({
        "answer_accuracy": [0.7, 0.85],
        "answer_correctness": [0.65, 0.8],
        "factual_correctness": [0.68, 0.82],
        "faithfulness": [0.8, 0.9],
        "context_precision": [0.75, 0.8],
        "context_recall": [0.65, 0.7]
    })
    mock_evaluate.return_value = mock_results

    # Mock metrics
    mock_aa.return_value = "answer_accuracy_metric"
    mock_ac.return_value = "answer_correctness_metric"
    mock_fc.return_value = "factual_correctness_metric"
    mock_f.return_value = "faithfulness_metric"
    mock_cp.return_value = "context_precision_metric"
    mock_cr.return_value = "context_recall_metric"

    # Create the evaluator
    evaluator = RagEvaluator(evaluation_config)

    # Set up the evaluator (mocking the setup method)
    with patch.object(evaluator, 'setup'):
        # Ensure the metrics are set manually since we're mocking setup
        evaluator.metrics = [
            "answer_accuracy_metric",
            "answer_correctness_metric",
            "factual_correctness_metric",
            "faithfulness_metric",
            "context_precision_metric",
            "context_recall_metric"
        ]

        # Evaluate the RAG output
        results = evaluator.evaluate(evaluation_data)

    # Check that the dataset was created correctly
    mock_dataset.assert_called_once()

    # Check that evaluate was called with the correct arguments
    mock_evaluate.assert_called_once()

    # Check the results
    assert "results" in results
    assert "metrics" in results
    assert "average_scores" in results
    assert len(results["average_scores"]) == 6

    # Make sure the average scores are calculated correctly
    assert results["average_scores"]["answer_accuracy"] == 0.775  # (0.7 + 0.85) / 2
    assert results["average_scores"]["answer_correctness"] == 0.725  # (0.65 + 0.8) / 2
    assert results["average_scores"]["factual_correctness"] == 0.75  # (0.68 + 0.82) / 2
    assert results["average_scores"]["faithfulness"] == 0.85  # (0.8 + 0.9) / 2
    assert results["average_scores"]["context_precision"] == 0.775  # (0.75 + 0.8) / 2
    assert results["average_scores"]["context_recall"] == 0.675  # (0.65 + 0.7) / 2


@patch('os.environ.get')
@patch('langchain_openai.chat_models.base.BaseChatOpenAI')
def test_setup_llm_deepseek(mock_base_chat_openai, mock_os_environ_get, evaluation_config):
    """Test that the evaluator can set up a DeepSeek LLM."""
    # Mock the environment variable
    mock_os_environ_get.return_value = "fake-api-key"

    # Mock the OpenAI chat model
    mock_llm = MagicMock()
    mock_base_chat_openai.return_value = mock_llm

    # Create the evaluator
    evaluator = RagEvaluator(evaluation_config)

    # Call the _setup_llm method
    evaluator._setup_llm()

    # Check that the LLM was initialized correctly
    mock_base_chat_openai.assert_called_once_with(
        model="deepseek-chat",
        temperature=0.3,
        max_tokens=8192,
        base_url="https://api.deepseek.com/beta",
        api_key="fake-api-key"
    )
    assert evaluator.eval_llm == mock_llm


@patch('os.environ.get')
@patch('langchain_google_genai.ChatGoogleGenerativeAI')
def test_setup_llm_gemini(mock_chat_google, mock_os_environ_get):
    """Test that the evaluator can set up a Gemini LLM."""
    # Create a config with Gemini
    config = {
        "llm_model": "gemini-1.5-pro",
        "cache_dir": "cache/test_ragas"
    }

    # Mock the Google chat model
    mock_llm = MagicMock()
    mock_chat_google.return_value = mock_llm

    # Create the evaluator
    evaluator = RagEvaluator(config)

    # Call the _setup_llm method
    evaluator._setup_llm()

    # Check that the LLM was initialized correctly
    mock_chat_google.assert_called_once_with(
        model="gemini-1.5-pro",
        temperature=0.2,
        max_output_tokens=8192
    )
    assert evaluator.eval_llm == mock_llm


@pytest.mark.skip(reason="Issue with protobuf implementation mocking")
@patch('vertexai.init')
@patch('langchain_google_vertexai.VertexAIEmbeddings')
@patch('os.environ.get')
def test_setup_embeddings(mock_os_environ_get, mock_vertex_embeddings, mock_vertexai_init):
    """Test that the evaluator can set up embeddings."""
    # Mock the environment variables
    def mock_env_get(var, default=None):
        if var == "GOOGLE_CLOUD_PROJECT_ID":
            return "test-project"
        elif var == "GOOGLE_CLOUD_LOCATION":
            return "us-central1"
        return default

    mock_os_environ_get.side_effect = mock_env_get

    # Mock the embeddings
    mock_embeddings = MagicMock()
    mock_vertex_embeddings.return_value = mock_embeddings

    # Create the evaluator
    evaluator = RagEvaluator({"llm_model": "gemini-1.5-pro"})

    # Call the _setup_embeddings method
    evaluator._setup_embeddings()

    # Check that VertexAI was initialized correctly
    mock_vertexai_init.assert_called_once_with(project="test-project", location="us-central1")

    # Check that the embeddings were initialized correctly
    mock_vertex_embeddings.assert_called_once_with(
        model_name="text-embedding-005",
        request_parallelism=1
    )
    assert evaluator.embeddings == mock_embeddings
