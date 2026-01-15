import pytest
from unittest.mock import MagicMock, patch
from openai import APIConnectionError
from loom.client import LoomClient, LoomResponse

# --- Fixtures ---
@pytest.fixture
def mock_openai_client():
    """Mocks the internal OpenAI client to avoid making real network calls during unit tests."""
    with patch("loom.client.OpenAI") as MockClient:
        yield MockClient.return_value

@pytest.fixture
def loom(mock_openai_client):
    """Returns a LoomClient instance with a mocked backend."""
    return LoomClient(host="http://fake-url", model_name="test-model")

# --- Unit Tests (Logic & Contracts) ---

def test_preconditions_raise_value_error(loom):
    """
    Verifies that the client enforces Design by Contract pre-conditions.
    """
    # Test 1: Empty System Prompt
    with pytest.raises(ValueError, match="system_prompt cannot be empty"):
        loom.generate(system_prompt="", user_prompt="Hello")

    # Test 2: Empty User Prompt
    with pytest.raises(ValueError, match="user_prompt cannot be empty"):
        loom.generate(system_prompt="Valid", user_prompt="   ")

    # Test 3: Temperature out of bounds
    with pytest.raises(ValueError, match="temperature -1.0 is out of range"):
        loom.generate(system_prompt="Valid", user_prompt="Valid", temperature=-1.0)

def test_successful_generation_structure(loom, mock_openai_client):
    """
    Verifies that a successful API response is correctly parsed into a LoomResponse object.
    """
    # Setup the mock response to look like a real OpenAI response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Threadrippers are powerful."
    mock_response.choices[0].finish_reason = "stop"

    # This prevents MagicMock from auto-creating a child mock when accessed.
    mock_response.choices[0].message.reasoning_content = None

    mock_response.choices[0].message.model_extra = {}

    # Configure attributes directly because the client now accesses them directly
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model = "test-model"
    
    # Configure the client to return this mock
    mock_openai_client.chat.completions.create.return_value = mock_response

    # Execute
    result = loom.generate("System", "User")

    # Assert Post-conditions
    assert isinstance(result, LoomResponse)
    assert result.content == "Threadrippers are powerful."
    assert result.reasoning is None
    assert result.token_usage["total_tokens"] == 15
    assert result.finish_reason == "stop"

def test_generation_with_reasoning(loom, mock_openai_client):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "42"
    mock_response.choices[0].finish_reason = "stop"
    # Simulate vLLM returning the reasoning field
    mock_response.choices[0].message.reasoning_content = "Thinking about the meaning of life..." 
    
    # (Set other required usage/model fields as above...)
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model = "gpt-oss-120b"

    mock_openai_client.chat.completions.create.return_value = mock_response

    result = loom.generate("System", "User")
    
    assert result.reasoning == "Thinking about the meaning of life..."

def test_connection_error_handling(loom, mock_openai_client):
    """
    Verifies that network errors are raised correctly.
    """
    # Simulate the server being down
    mock_openai_client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())

    with pytest.raises(APIConnectionError):
        loom.generate("System", "User")

# --- Integration Test (Real Network) ---

@pytest.mark.integration
def test_real_connection_to_lenny():
    """
    Hits the actual Threadripper server. 
    Requires the server to be running on 192.168.1.6:8080.
    """
    # Use the default host defined in your class (Lenny's IP)
    client = LoomClient() 
    
    try:
        response = client.generate(
            system_prompt="Health Check",
            user_prompt="Reply with the single word 'OK'.",
            temperature=0.1,
            max_tokens=100
        )
        
        # Verify we got a real response
        print(f"\nIntegration Response: {response.content}")
        assert isinstance(response, LoomResponse)
        assert len(response.content) > 0
        # assert response.model_used == "gpt-120b" # Should match your systemd alias

    except APIConnectionError:
        pytest.fail("Could not connect to Lenny. Is the systemd service running?")
