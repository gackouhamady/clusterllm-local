from unittest.mock import MagicMock, patch

import pytest

from clusterllm.llm_client.ollama import OllamaClient


@patch("clusterllm.llm_client.ollama.requests.post")
def test_ollama_generate_returns_response(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "ok"}
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    client = OllamaClient(model_tag="llama3:8b-instruct-q4_K_M")
    out = client.generate("hello")

    assert out == "ok"
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert "/api/generate" in args[0]
    assert kwargs["json"]["model"] == "llama3:8b-instruct-q4_K_M"
    assert kwargs["json"]["prompt"] == "hello"
    assert kwargs["json"]["stream"] is False


def test_unknown_model_key_raises():
    with pytest.raises(ValueError):
        OllamaClient(model_key="does_not_exist")


def test_model_key_resolves_to_tag():
    client = OllamaClient(model_key="gemma_q4km")
    assert client.model == "gemma:7b-instruct-q4_K_M"
