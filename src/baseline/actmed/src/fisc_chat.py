"""Minimal OpenAI-compatible chat client used by the FISC ACTMED adapter."""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class CompatibleChat:
    """Call an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, model: str, retries: int = 6):
        self.model = model
        self.base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
        self.api_keys = tuple(
            value
            for name, value in sorted(os.environ.items())
            if name == "OPENAI_API_KEY" or name.startswith("OPENAI_API_KEY")
        )
        if not self.api_keys:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        self.retries = retries
        self.calls = 0

    def __call__(self, user_prompt, model_name=None, temperature=1, top_p=0.95):
        payload = json.dumps({
            "model": model_name or self.model,
            "messages": [{"role": "user", "content": str(user_prompt)}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 128,
        }).encode("utf-8")

        class PreservePostRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, old, fp, code, msg, headers, newurl):
                old_url, new_url = urlsplit(old.full_url), urlsplit(newurl)
                if (old_url.scheme, old_url.netloc) != (new_url.scheme, new_url.netloc):
                    raise urllib.error.HTTPError(
                        old.full_url, code, "refusing cross-host API-key redirect", headers, fp
                    )
                return urllib.request.Request(
                    newurl, data=old.data, headers=dict(old.headers), method=old.get_method()
                )

        opener = urllib.request.build_opener(PreservePostRedirect)
        for attempt in range(self.retries):
            self.calls += 1
            api_key = self.api_keys[attempt % len(self.api_keys)]
            request = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=payload,
                method="POST",
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                },
            )
            try:
                with opener.open(request, timeout=180) as response:
                    body = json.load(response)
                content = body.get("choices", [{}])[0].get("message", {}).get("content")
                if not content:
                    raise RuntimeError("API response is missing choices[0].message.content")
                return content
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                retryable = error.code in {401, 403, 429, 500, 502, 503, 504, 524}
                if not retryable or attempt == self.retries - 1:
                    raise RuntimeError(f"API HTTP {error.code}: {detail[:800]}") from error
            except (TimeoutError, urllib.error.URLError) as error:
                if attempt == self.retries - 1:
                    raise RuntimeError(f"API request failed: {error}") from error
            time.sleep(min(2 ** attempt, 20) + random.random())
        raise RuntimeError("API request failed")
