# Test the /v1/tokenize endpoint

import sys, os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, logging
import litellm
from litellm.proxy.proxy_server import tokenize
from litellm._logging import verbose_proxy_logger

verbose_proxy_logger.setLevel(level=logging.DEBUG)

from litellm.proxy._types import TokenizeRequest
from litellm import Router


@pytest.mark.asyncio
async def test_tokenize_with_text():
    """
    Test /v1/tokenize endpoint with text input
    - Should return tokens, token_strings, total_tokens, and model
    """
    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    response = await tokenize(
        request=TokenizeRequest(
            model="gpt-4",
            text="Hello world!",
        )
    )

    print("response: ", response)

    # Validate response structure
    assert hasattr(response, "tokens")
    assert hasattr(response, "token_strings")
    assert hasattr(response, "total_tokens")
    assert hasattr(response, "model")

    # Validate response data
    assert isinstance(response.tokens, list)
    assert isinstance(response.token_strings, list)
    assert len(response.tokens) > 0
    assert len(response.token_strings) > 0
    assert len(response.tokens) == len(response.token_strings)
    assert response.total_tokens == len(response.tokens)
    assert response.model == "gpt-4"


@pytest.mark.asyncio
async def test_tokenize_with_messages():
    """
    Test /v1/tokenize endpoint with messages input
    - Should convert messages to text and tokenize
    """
    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    response = await tokenize(
        request=TokenizeRequest(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        )
    )

    print("response: ", response)

    # Validate response structure
    assert isinstance(response.tokens, list)
    assert isinstance(response.token_strings, list)
    assert len(response.tokens) > 0
    assert len(response.token_strings) > 0
    assert response.total_tokens == len(response.tokens)
    assert response.model == "gpt-3.5-turbo"


@pytest.mark.asyncio
async def test_tokenize_without_text_or_messages():
    """
    Test /v1/tokenize endpoint without text or messages
    - Should raise HTTPException with 400 status code
    """
    from fastapi import HTTPException

    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    with pytest.raises(HTTPException) as exc_info:
        await tokenize(
            request=TokenizeRequest(
                model="gpt-4",
            )
        )

    assert exc_info.value.status_code == 400
    assert "Either 'text' or 'messages' must be provided" in str(exc_info.value.detail)
