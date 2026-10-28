import os
import re
import time
from typing import Callable, Optional

import requests
from dotenv import load_dotenv

from .prompts.ar import (
    RISK_DETECTION_PROMPT as AR_RISK_PROMPT,
)
from .prompts.ar import (
    SUMMARY_PROMPT as AR_SUMMARY_PROMPT,
)
from .prompts.en import (
    RISK_DETECTION_PROMPT as EN_RISK_PROMPT,
)
from .prompts.en import (
    SUMMARY_PROMPT as EN_SUMMARY_PROMPT,
)
from schemas import ContractAnalysis

from config import ENV_FILE

load_dotenv(ENV_FILE)

# --------------------------------------------------
# Settings: the final model is Qwen3-8B only
# --------------------------------------------------

MODEL_NAME = "Qwen3-8B"

URL_VARIABLE = "QWEN8_API_URL"
KEY_VARIABLE = "QWEN8_API_KEY"

REQUEST_TIMEOUT = 240

# Connection retry (network problems only)
CONNECTION_ATTEMPTS = 3
WAIT_SECONDS = 3
RETRY_STATUS_CODES = {429, 502, 503, 504}


# --------------------------------------------------
# Prompt building
# --------------------------------------------------


def build_summary_prompt(
    contract_text: str,
    language: str = "en",
) -> str:
    """Build a contract summary prompt in Arabic or English."""

    if not contract_text or not contract_text.strip():
        raise ValueError("contract_text cannot be empty")

    if language == "ar":
        template = AR_SUMMARY_PROMPT
    elif language == "en":
        template = EN_SUMMARY_PROMPT
    else:
        raise ValueError("language must be 'ar' or 'en'")

    return template.format(contract_text=contract_text)


def build_risk_prompt(
    contract_text: str,
    legal_context: str,
    language: str = "en",
) -> str:
    """Build a risk detection prompt using contract text and legal context."""

    if not contract_text or not contract_text.strip():
        raise ValueError("contract_text cannot be empty")

    if language == "ar":
        template = AR_RISK_PROMPT
    elif language == "en":
        template = EN_RISK_PROMPT
    else:
        raise ValueError("language must be 'ar' or 'en'")

    return template.format(
        contract_text=contract_text,
        legal_context=legal_context,
    )


# --------------------------------------------------
# Talking to the Qwen3-8B server (Kaggle + ngrok)
# --------------------------------------------------


def _read_env(name: str) -> str:
    """Read an env variable and remove spaces and quotes."""

    value = os.getenv(name) or ""
    return value.strip().strip('"').strip("'")


def get_generate_url() -> str:
    """Return the full /generate URL (works with or without /generate in .env)."""

    url = _read_env(URL_VARIABLE)

    if not url:
        raise ValueError(f"{URL_VARIABLE} is missing from .env")

    url = url.rstrip("/")

    if not url.endswith("/generate"):
        url += "/generate"

    return url


def _post_with_connection_retry(url: str, headers: dict, payload: dict):
    """
    POST to the server. Retry ONLY for connection drops and temporary
    server errors. Do not retry if the model already answered, and do
    not retry a long read timeout (the same contract would be sent twice).
    """

    last_error = None

    for attempt in range(1, CONNECTION_ATTEMPTS + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code in RETRY_STATUS_CODES:
                last_error = RuntimeError(
                    f"HTTP {response.status_code} from {response.url} | "
                    f"Server said: {response.text[:200]}"
                )
            else:
                return response

        except (requests.ConnectionError, requests.ConnectTimeout) as error:
            last_error = error

        if attempt < CONNECTION_ATTEMPTS:
            time.sleep(WAIT_SECONDS * attempt)

    raise RuntimeError(
        f"Model server not reachable after {CONNECTION_ATTEMPTS} "
        f"attempts. Last error: {last_error}"
    )


# response from the server
def _extract_text(data) -> str:
    """Find the generated text in the server response."""

    if isinstance(data, str) and data.strip():
        return data.strip()

    if isinstance(data, dict):
        for key in ("text", "response", "generated_text", "output"):
            value = data.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        choices = data.get("choices")

        if isinstance(choices, list) and choices:
            first = choices[0]

            if isinstance(first, dict):
                message = first.get("message", {})

                if isinstance(message, dict):
                    content = message.get("content")

                    if isinstance(content, str) and content.strip():
                        return content.strip()

                content = first.get("text")

                if isinstance(content, str) and content.strip():
                    return content.strip()

    raise ValueError(
        f"Could not find generated text in API response: {str(data)[:300]}"
    )


def generate_response(prompt: str) -> str:
    """Send a prompt to Qwen3-8B and return the raw text."""

    api_key = _read_env(KEY_VARIABLE)

    if not api_key:
        raise ValueError(f"{KEY_VARIABLE} is missing from .env")

    response = _post_with_connection_retry(
        get_generate_url(),
        headers={
            "x-api-key": api_key,
            # Skips the ngrok free-plan warning page.
            "ngrok-skip-browser-warning": "true",
        },
        payload={"prompt": prompt},
    )

    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code} from {response.url} | "
            f"Server said: {response.text[:300]}"
        )

    return _extract_text(response.json())


def get_generate_fn() -> Callable[[str], str]:
    """Return the function that talks to the final model (Qwen3-8B)."""

    return generate_response


# --------------------------------------------------
# Parsing and validation
# --------------------------------------------------


def clean_json_text(text: str) -> str:
    """Remove <think> blocks, Markdown fences, and text around the JSON."""

    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end > start:
        text = text[start : end + 1]

    return text


def parse_analysis_response(
    response_text: str,
) -> ContractAnalysis:
    """Parse and validate the LLM's JSON response."""

    if not response_text or not response_text.strip():
        raise ValueError("The LLM response is empty.")

    try:
        return ContractAnalysis.model_validate_json(clean_json_text(response_text))
    except ValueError as error:
        raise ValueError(f"Invalid ContractAnalysis JSON: {error}") from error


# --------------------------------------------------
# Main function used by the app
# --------------------------------------------------


def analyze_contract(
    contract_text: str,
    legal_context: str,
    language: str = "en",
    max_retries: int = 2,
    generate_fn: Optional[Callable[[str], str]] = None,
) -> ContractAnalysis:
    """
    Analyze a contract with Qwen3-8B and return a validated result.

    The LLM receives the contract text and legal context only.
    BERT results are not part of this function.

    generate_fn is optional. Leave it empty in the app. It exists only
    so tests can pass a fake model.

    max_retries is for INVALID model output (bad JSON or schema).
    Connection problems are handled inside generate_response.

    Raises ValueError if no valid result after max_retries + 1 attempts.
    The caller must catch it and show a clear message to the user.
    """

    if not contract_text or not contract_text.strip():
        raise ValueError("contract_text cannot be empty")

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")

    if generate_fn is None:
        generate_fn = generate_response

    prompt = build_risk_prompt(
        contract_text=contract_text,
        legal_context=legal_context,
        language=language,
    )

    last_error = None

    for attempt in range(max_retries + 1):
        current_prompt = prompt

        if attempt > 0:
            short_error = str(last_error)[:500]

            if language == "ar":
                correction = (
                    "\n\nأعد إنشاء النتيجة بصيغة JSON صالحة فقط، "
                    "ومتوافقة مع الحقول المطلوبة. "
                    "لا تضف Markdown أو نصًا خارج JSON.\n"
                    f"خطأ التحقق: {short_error}"
                )
            else:
                correction = (
                    "\n\nReturn corrected, valid JSON only, matching "
                    "the required schema. Do not add Markdown or extra text.\n"
                    f"Validation error: {short_error}"
                )

            current_prompt += correction

        response_text = generate_fn(current_prompt)

        try:
            return parse_analysis_response(response_text)
        except ValueError as error:
            last_error = error

    raise ValueError(
        f"Could not get valid contract analysis after "
        f"{max_retries + 1} attempts. Last error: {last_error}"
    )


def summarize_contract(
    contract_text: str,
    language: str = "en",
    generate_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """Generate a contract summary in Arabic or English."""

    if not contract_text or not contract_text.strip():
        raise ValueError("contract_text cannot be empty")

    if generate_fn is None:
        generate_fn = generate_response

    prompt = build_summary_prompt(
        contract_text=contract_text,
        language=language,
    )

    summary = generate_fn(prompt)

    if not summary or not summary.strip():
        raise ValueError("The LLM returned an empty summary.")

    return summary.strip()
