from unittest.mock import MagicMock, patch

from clusterllm.llm_client.ollama import OllamaClient
from clusterllm.llm_client.local_completion import delayed_completion, post_process


@patch("clusterllm.llm_client.ollama.requests.post")
def test_delayed_completion_and_post_process(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "Yes"}
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    client = OllamaClient(model_key="llama3_q4km")

    completion, err = delayed_completion(
        client=client,
        prompt="Answer Yes or No only",
        delay_in_seconds=0.0,
        max_trials=1,
    )

    assert err is None
    content, parsed = post_process(completion)
    assert content == "Yes"
    assert parsed == ["Yes"]
