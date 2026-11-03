#!/usr/bin/env python3
"""Refresh the hosted Motion Glass model catalog from live provider data.

Standard library only (urllib/json/argparse) so it runs in a bare GitHub Actions
job and on developer machines without installing anything.

What one run does
-----------------
1. `GET https://api.anthropic.com/v1/models` (x-api-key + anthropic-version) and
   `GET https://api.openai.com/v1/models` (Bearer). Provider list failures are
   non-fatal: that provider's models are left untouched for the run.
2. Admits new chat-capable models from those lists (see ADMISSION POLICY below),
   adding them with conservative metadata and `lifecycle: "active"`.
3. Merges public OpenAI pricing from https://icoretech.github.io/openai-json-pricing/pricing.json
   when reachable. Prices are only ever copied from that source — an unknown
   price is `null`, never a guess.
4. Drift detection: a catalog model missing from its provider's list for
   `--grace-runs` consecutive successful refreshes (default 2) is marked
   `retired` and gets an "Automation:" deprecation record. Models already
   retired by automation are reactivated if the provider lists them again.
5. Writes a deterministic catalog (stable ordering, fixed key order, a single
   `generatedAt`). If the content is unchanged, the file is left byte-identical
   (`generatedAt` only changes when something else did) so the workflow does not
   create noise commits.
6. `--check` reports whether the file would change (exit 1 if so). `--offline`
   skips all network access and is used by PR CI; combined with `--check` it
   verifies the committed file is a fixed point of this writer.

Automation state (drift counters) lives in an optional top-level `_automation`
object inside catalog.json; the app's Codable models ignore unknown keys, and the
block is removed entirely once there is nothing to track. No separate state file.

ADMISSION POLICY
----------------
Only chat models are added to the picker. Anthropic: `claude-{haiku,sonnet,opus,fable}-N[-M]`
with major version >= 5. OpenAI: `gpt-N[.M][-suffix]` with major >= 5 and
`oN[-suffix]` with major >= 3, excluding non-chat families (codex, pro, search,
image, audio, realtime, transcribe, tts, embedding, moderation, sora, whisper,
instruct) and dated snapshot ids. `--admit-all` relaxes only the version floor.
Existing catalog entries are never removed by this filter.

Environment: ANTHROPIC_API_KEY / OPENAI_API_KEY (never printed or written). One
missing key is tolerated (that provider is skipped with a warning); both missing
is an error unless `--offline` is used.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Import the sibling validator whether this file is run as a script from
# `scripts/` (public catalog repo) or `catalog/scripts/` (app repo).
try:
    from validate_catalog import validate_catalog_document
except ImportError:  # pragma: no cover - only hit in unusual embedders
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validate_catalog import validate_catalog_document

SCHEMA_VERSION = 1
DEFAULT_ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
DEFAULT_OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
DEFAULT_PRICING_URL = "https://icoretech.github.io/openai-json-pricing/pricing.json"
ANTHROPIC_VERSION = "2023-06-01"
USER_AGENT = "motion-glass-catalog-refresh/1.0 (+https://github.com/icoretech/motion-glass-catalog)"
HTTP_TIMEOUT_SECONDS = 25
HTTP_ATTEMPTS = 2
ANTHROPIC_PAGE_LIMIT = 100
ANTHROPIC_MAX_PAGES = 20
AUTO_DEPRECATION_PREFIX = "Automation: "

DETAILS_PENDING = "New \u2014 pricing pending"
DETAILS_ADDED = "Added automatically from the provider model list."

PROVIDER_ORDER = ("anthropic", "openai")
ROOT_KEY_ORDER = ("schemaVersion", "generatedAt", "sourceURL", "providers", "deprecations", "_automation")
PROVIDER_KEY_ORDER = ("provider", "displayName", "apiBaseURL", "docsURL", "models")
MODEL_KEY_ORDER = (
    "provider",
    "id",
    "displayName",
    "contextWindow",
    "maxOutputTokens",
    "pricing",
    "lifecycle",
    "supportsStructuredJSON",
    "supportsVision",
    "supportsStreaming",
    "isRecommended",
    "details",
    "releasedAt",
    "replacementID",
)
PRICING_KEY_ORDER = ("inputPerMTok", "outputPerMTok", "cacheReadPerMTok", "cacheWritePerMTok", "longContext")
LONG_CONTEXT_KEY_ORDER = ("inputTokensAbove", "inputPerMTok", "outputPerMTok", "cacheReadPerMTok", "cacheWritePerMTok")
DEPRECATION_KEY_ORDER = ("provider", "modelID", "replacedBy", "message", "retiresAt")

# Conservative capability/limit defaults for models discovered from a provider
# list (the provider lists do not publish context/output limits or capabilities).
NEW_MODEL_DEFAULTS = {
    "anthropic": {"contextWindow": 200_000, "maxOutputTokens": 32_000},
    "openai": {"contextWindow": 128_000, "maxOutputTokens": 32_000},
}

# Admission policy (see module docstring).
ANTHROPIC_ID_RE = re.compile(r"^claude-(haiku|sonnet|opus|fable)-(\d+)(?:-(\d+))?$")
OPENAI_GPT_ID_RE = re.compile(r"^gpt-(\d+)(?:\.(\d+))?(?:-([a-z0-9.]+(?:-[a-z0-9.]+)*))?$")
OPENAI_O_ID_RE = re.compile(r"^o(\d+)(?:-[a-z0-9.]+)*$")
DATED_SNAPSHOT_RE = re.compile(r"(?:-\d{8}|-\d{4}-\d{2}-\d{2})$")
NON_CHAT_TOKENS = frozenset(
    {
        "codex",
        "pro",
        "search",
        "image",
        "audio",
        "realtime",
        "transcribe",
        "tts",
        "embedding",
        "moderation",
        "sora",
        "whisper",
        "davinci",
        "babbage",
        "instruct",
    }
)
MIN_MAJOR_VERSION = {"anthropic": 5, "openai": 5}
# o-series reasoning models are admitted only from a future major generation;
# the 2025 line (o3/o4) is legacy and out of scope for a creative Director picker.
MIN_MAJOR_VERSION_O_SERIES = 5
# Chat-product aliases are redundant with the underlying model families.
NON_ADMITTED_SUFFIXES = ("-chat-latest",)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_from_unix(value: Any) -> str | None:
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return None


def is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def ordered_mapping(source: dict[str, Any], key_order: tuple[str, ...]) -> dict[str, Any]:
    """Fixed key order for known fields, then remaining keys sorted (forward compat)."""
    ordered: dict[str, Any] = {}
    for key in key_order:
        if key in source:
            ordered[key] = source[key]
    for key in sorted(source):
        if key not in ordered:
            ordered[key] = source[key]
    return ordered


def canonicalize(catalog: dict[str, Any]) -> dict[str, Any]:
    """Return the catalog with the writer's canonical ordering (values untouched)."""
    providers = catalog.get("providers")
    canonical_providers: list[Any] = []
    if isinstance(providers, list):
        known: list[dict[str, Any]] = []
        unknown: list[dict[str, Any]] = []
        for provider_entry in providers:
            if not isinstance(provider_entry, dict):
                known.append(provider_entry)  # type: ignore[arg-type]
                continue
            provider_id = provider_entry.get("provider")
            if provider_id in PROVIDER_ORDER:
                known.append(provider_entry)
            else:
                unknown.append(provider_entry)
        known.sort(key=lambda entry: PROVIDER_ORDER.index(entry["provider"]))
        unknown.sort(key=lambda entry: str(entry.get("provider")))
        for provider_entry in known + unknown:
            if not isinstance(provider_entry, dict):
                canonical_providers.append(provider_entry)
                continue
            models = provider_entry.get("models")
            canonical_models: list[Any] = []
            if isinstance(models, list):
                for model in models:
                    canonical_models.append(canonicalize_model(model) if isinstance(model, dict) else model)
            provider_copy = dict(provider_entry)
            provider_copy["models"] = canonical_models
            canonical_providers.append(ordered_mapping(provider_copy, PROVIDER_KEY_ORDER))

    deprecations = catalog.get("deprecations")
    canonical_deprecations: list[Any] = []
    if isinstance(deprecations, list):
        entries = [entry for entry in deprecations if isinstance(entry, dict)]
        entries.sort(key=lambda entry: (str(entry.get("provider")), str(entry.get("modelID"))))
        canonical_deprecations = [ordered_mapping(entry, DEPRECATION_KEY_ORDER) for entry in entries]

    automation = catalog.get("_automation")
    canonical_automation: dict[str, Any] | None = None
    if isinstance(automation, dict):
        models = automation.get("models")
        if isinstance(models, dict) and models:
            canonical_models_state = {
                key: ordered_mapping(value, ("missingRuns",))
                for key, value in sorted(models.items())
                if isinstance(value, dict)
            }
            canonical_automation = {
                "schemaVersion": automation.get("schemaVersion", 1),
                "models": canonical_models_state,
            }

    root = dict(catalog)
    root["providers"] = canonical_providers
    root["deprecations"] = canonical_deprecations
    if canonical_automation is not None:
        root["_automation"] = canonical_automation
    else:
        root.pop("_automation", None)
    return ordered_mapping(root, ROOT_KEY_ORDER)


def canonicalize_model(model: dict[str, Any]) -> dict[str, Any]:
    copy_model = dict(model)
    pricing = copy_model.get("pricing")
    if isinstance(pricing, dict):
        pricing_copy = dict(pricing)
        long_context = pricing_copy.get("longContext")
        if isinstance(long_context, dict):
            pricing_copy["longContext"] = ordered_mapping(long_context, LONG_CONTEXT_KEY_ORDER)
        copy_model["pricing"] = ordered_mapping(pricing_copy, PRICING_KEY_ORDER)
    return ordered_mapping(copy_model, MODEL_KEY_ORDER)


def content_without_timestamp(catalog: dict[str, Any]) -> dict[str, Any]:
    canonical = canonicalize(catalog)
    canonical.pop("generatedAt", None)
    return canonical


def render_catalog(catalog: dict[str, Any]) -> str:
    return json.dumps(canonicalize(catalog), indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def default_catalog_path() -> Path:
    # Works both in the public catalog repo (scripts/refresh_catalog.py ->
    # catalog.json next to scripts/) and in the app repo (catalog/scripts/ ->
    # catalog/catalog.json).
    return Path(__file__).resolve().parent.parent / "catalog.json"


# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------


def prettify_display_name(provider: str, model_id: str) -> str:
    """Human-readable name. Mirrors CatalogMerge.prettyDisplayName in the app."""
    parts = model_id.split("-")
    if provider == "anthropic":
        words = ["Claude"]
        for part in parts[1:]:
            words.append(part.capitalize() if not part[0].isdigit() else part)
        return " ".join(words)

    first = parts[0]
    if first.lower() == "gpt":
        head = "GPT"
    elif re.fullmatch(r"o\d+", first):
        head = first
    elif first.lower() in {"api", "tts", "stt", "llm", "hd"}:
        head = first.upper()
    else:
        head = first.capitalize()
    words = [head]
    for index, part in enumerate(parts[1:]):
        word = part if part[0].isdigit() else part.capitalize()
        if index == 0 and part[0].isdigit() and head in {"GPT", "O"}:
            words[-1] += "-" + part
        else:
            words.append(word)
    return " ".join(words)


# ---------------------------------------------------------------------------
# Admission policy
# ---------------------------------------------------------------------------


def is_admitted(provider: str, model_id: str, admit_all: bool = False) -> bool:
    if not isinstance(model_id, str) or not model_id:
        return False
    if DATED_SNAPSHOT_RE.search(model_id):
        return False
    if model_id.endswith(NON_ADMITTED_SUFFIXES):
        return False
    tokens = set(re.split(r"[-.]", model_id))
    if tokens & NON_CHAT_TOKENS:
        return False

    if provider == "anthropic":
        match = ANTHROPIC_ID_RE.match(model_id)
        if not match:
            return False
        major = int(match.group(2))
        return admit_all or major >= MIN_MAJOR_VERSION["anthropic"]

    if provider == "openai":
        gpt_match = OPENAI_GPT_ID_RE.match(model_id)
        if gpt_match:
            major = int(gpt_match.group(1))
            return admit_all or major >= MIN_MAJOR_VERSION["openai"]
        o_match = OPENAI_O_ID_RE.match(model_id)
        if o_match:
            major = int(o_match.group(1))
            return admit_all or major >= MIN_MAJOR_VERSION_O_SERIES
    return False


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def http_get_json(
    url: str,
    headers: dict[str, str],
    label: str,
    warnings: list[str],
) -> Any | None:
    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    request_headers.update(headers)
    last_error = "unknown error"
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        request = urllib.request.Request(url, headers=request_headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            last_error = f"HTTP {error.code}"
            # Auth and permission errors will not fix themselves; don't retry.
            if error.code in (401, 403):
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as error:
            last_error = str(error)
        if attempt < HTTP_ATTEMPTS:
            time.sleep(1.5 * attempt)
    warnings.append(f"{label}: request failed ({last_error})")
    return None


def fetch_anthropic_models(
    api_key: str,
    base_url: str,
    warnings: list[str],
) -> list[dict[str, Any]] | None:
    models: list[dict[str, Any]] = []
    after_id: str | None = None
    for _ in range(ANTHROPIC_MAX_PAGES):
        query = {"limit": str(ANTHROPIC_PAGE_LIMIT)}
        if after_id:
            query["after_id"] = after_id
        url = f"{base_url}?{urllib.parse.urlencode(query)}"
        payload = http_get_json(
            url,
            {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
            "anthropic /v1/models",
            warnings,
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            return None
        for item in payload["data"]:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            models.append(
                {
                    "id": item["id"],
                    "display_name": item.get("display_name"),
                    "created_at": item.get("created_at"),
                }
            )
        if not payload.get("has_more"):
            break
        after_id = payload.get("last_id")
        if not after_id:
            break
    return models


def fetch_openai_models(
    api_key: str,
    base_url: str,
    warnings: list[str],
) -> list[dict[str, Any]] | None:
    payload = http_get_json(base_url, {"Authorization": f"Bearer {api_key}"}, "openai /v1/models", warnings)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return None
    models: list[dict[str, Any]] = []
    for item in payload["data"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        models.append({"id": item["id"], "created": item.get("created")})
    return models


def fetch_pricing_source(url: str, warnings: list[str]) -> dict[str, Any] | None:
    payload = http_get_json(url, {}, "openai pricing catalog", warnings)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        return None
    return payload


# ---------------------------------------------------------------------------
# Pricing merge (never invent a price)
# ---------------------------------------------------------------------------


def _rate(block: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = block.get(key)
        if is_positive_number(value):
            return float(value)
    return None


def pricing_from_public_entry(entry: Any) -> tuple[dict[str, Any] | None, str]:
    """Map one openai-json-pricing model record to a Motion Glass ModelPricing.

    Only values literally present in the public pricing document are used; any
    unknown bucket is omitted (null), never derived. Returns
    `(pricing, long_context_state)` where state is "set" (source published a
    long-context tier), "none" (source says the model has no long-context tier)
    or "unknown" (source did not say).
    """
    if not isinstance(entry, dict):
        return None, "unknown"
    if entry.get("pricing_type") != "per_1m_tokens":
        return None, "unknown"
    standard = entry.get("prices", {})
    if not isinstance(standard, dict):
        return None, "unknown"
    standard = standard.get("standard", {})
    if not isinstance(standard, dict):
        return None, "unknown"
    short = standard.get("default")
    if not isinstance(short, dict):
        short = standard.get("short_context")
    if not isinstance(short, dict) or short.get("available") is False:
        return None, "unknown"

    input_rate = _rate(short, "input")
    output_rate = _rate(short, "output")
    if input_rate is None or output_rate is None:
        return None, "unknown"

    pricing: dict[str, Any] = {"inputPerMTok": input_rate, "outputPerMTok": output_rate}
    cache_read = _rate(short, "cached_input")
    if cache_read is not None:
        pricing["cacheReadPerMTok"] = cache_read
    # Prefer the 30-minute cache-write rate (the app's default cache lifetime);
    # fall back to the 12-hour rate only if 30m is not published.
    cache_write = _rate(short, "cache_write_30m", "cache_write_12h")
    if cache_write is not None:
        pricing["cacheWritePerMTok"] = cache_write

    context_pricing = entry.get("context_pricing", {})
    tiers = context_pricing.get("tiers", {}) if isinstance(context_pricing, dict) else {}
    standard_tier = tiers.get("standard", {}) if isinstance(tiers, dict) else {}
    cutoff = standard_tier.get("long_context_input_tokens_gt") if isinstance(standard_tier, dict) else None
    long_block = standard.get("long_context")
    if standard_tier.get("status") == "uniform":
        return pricing, "none"
    if (
        standard_tier.get("status") == "known"
        and is_positive_int(cutoff)
        and isinstance(long_block, dict)
        and long_block.get("available") is not False
    ):
        long_input = _rate(long_block, "input")
        long_output = _rate(long_block, "output")
        if long_input is not None and long_output is not None:
            long_context: dict[str, Any] = {
                "inputTokensAbove": cutoff,
                "inputPerMTok": long_input,
                "outputPerMTok": long_output,
            }
            long_cache_read = _rate(long_block, "cached_input")
            if long_cache_read is not None:
                long_context["cacheReadPerMTok"] = long_cache_read
            long_cache_write = _rate(long_block, "cache_write_30m", "cache_write_12h")
            if long_cache_write is not None:
                long_context["cacheWritePerMTok"] = long_cache_write
            pricing["longContext"] = long_context
            return pricing, "set"
    return pricing, "unknown"


# ---------------------------------------------------------------------------
# Model records
# ---------------------------------------------------------------------------


def remote_release_date(provider: str, remote: dict[str, Any]) -> str | None:
    if provider == "anthropic":
        created_at = remote.get("created_at")
        if isinstance(created_at, str) and re.match(r"^\d{4}-\d{2}-\d{2}T", created_at):
            return created_at.replace("+00:00", "Z")
        return None
    return iso_from_unix(remote.get("created"))


def make_new_model(provider: str, remote: dict[str, Any]) -> dict[str, Any]:
    model_id = remote["id"]
    defaults = NEW_MODEL_DEFAULTS[provider]
    model: dict[str, Any] = {
        "provider": provider,
        "id": model_id,
        "displayName": prettify_display_name(provider, model_id),
        "contextWindow": defaults["contextWindow"],
        "maxOutputTokens": defaults["maxOutputTokens"],
        "pricing": None,
        "lifecycle": "active",
        "supportsStructuredJSON": False,
        "supportsVision": False,
        "supportsStreaming": True,
        "isRecommended": False,
        "details": DETAILS_PENDING,
    }
    released_at = remote_release_date(provider, remote)
    if released_at:
        model["releasedAt"] = released_at
    return model


def find_automation_deprecation(
    deprecations: list[Any], provider: str, model_id: str
) -> dict[str, Any] | None:
    for entry in deprecations:
        if (
            isinstance(entry, dict)
            and entry.get("provider") == provider
            and entry.get("modelID") == model_id
            and isinstance(entry.get("message"), str)
            and entry["message"].startswith(AUTO_DEPRECATION_PREFIX)
        ):
            return entry
    return None


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


class RefreshSummary:
    def __init__(self) -> None:
        self.added: list[str] = []
        self.reactivated: list[str] = []
        self.retired: list[str] = []
        self.pending_drift: list[str] = []
        self.pricing_updated: list[str] = []
        self.warnings: list[str] = []

    @property
    def changed(self) -> bool:
        return bool(
            self.added or self.reactivated or self.retired or self.pending_drift or self.pricing_updated
        )


def refresh_catalog(
    existing: dict[str, Any],
    *,
    offline: bool,
    grace_runs: int,
    admit_all: bool,
    anthropic_models_url: str,
    openai_models_url: str,
    pricing_url: str,
    merge_pricing: bool,
    now: str,
    api_keys: dict[str, str],
) -> tuple[dict[str, Any], RefreshSummary]:
    catalog = copy.deepcopy(existing)
    summary = RefreshSummary()
    warnings = summary.warnings

    providers: list[dict[str, Any]] = [
        provider for provider in catalog.get("providers", []) if isinstance(provider, dict)
    ]
    if not providers:
        raise ValueError("catalog has no providers to refresh")

    remote_models: dict[str, list[dict[str, Any]]] = {}
    provider_failures: dict[str, str] = {}

    if offline:
        # No network at all. The refresh is a canonical-form check only.
        pass
    else:
        for provider_entry in providers:
            provider = provider_entry.get("provider")
            if provider not in ("anthropic", "openai"):
                continue
            key = api_keys.get(provider)
            if not key:
                warnings.append(f"{provider}: no API key configured; skipping provider list")
                provider_failures[provider] = "no API key"
                continue
            if provider == "anthropic":
                fetched = fetch_anthropic_models(key, anthropic_models_url, warnings)
            else:
                fetched = fetch_openai_models(key, openai_models_url, warnings)
            if fetched is None:
                provider_failures[provider] = "request failed"
            else:
                remote_models[provider] = fetched

    state = catalog.get("_automation", {})
    state_models: dict[str, dict[str, Any]] = {}
    if isinstance(state, dict) and isinstance(state.get("models"), dict):
        state_models = {
            key: dict(value) for key, value in state["models"].items() if isinstance(value, dict)
        }

    deprecations: list[Any] = [
        entry for entry in catalog.get("deprecations", []) if isinstance(entry, dict)
    ]

    for provider_entry in providers:
        provider = provider_entry.get("provider")
        if provider not in ("anthropic", "openai"):
            continue
        models = provider_entry.get("models")
        if not isinstance(models, list):
            models = []
            provider_entry["models"] = models
        model_by_id: dict[str, dict[str, Any]] = {}
        for model in models:
            if isinstance(model, dict) and isinstance(model.get("id"), str):
                model_by_id[model["id"]] = model

        if offline or provider not in remote_models:
            continue

        remotes = remote_models[provider]
        seen_ids = {remote["id"] for remote in remotes if isinstance(remote.get("id"), str)}
        admitted = sorted(
            (
                remote
                for remote in remotes
                if is_admitted(provider, remote.get("id", ""), admit_all=admit_all)
            ),
            key=lambda remote: remote["id"],
        )

        # New models first, so pricing can be applied to them in the same run.
        for remote in admitted:
            model_id = remote["id"]
            if model_id in model_by_id:
                continue
            new_model = make_new_model(provider, remote)
            models.append(new_model)
            model_by_id[model_id] = new_model
            summary.added.append(f"{provider}:{model_id}")

        for model_id, model in model_by_id.items():
            state_key = f"{provider}:{model_id}"
            if model_id in seen_ids:
                state_models.pop(state_key, None)
                if model.get("lifecycle") == "retired":
                    automation_entry = find_automation_deprecation(deprecations, provider, model_id)
                    if automation_entry is not None:
                        model["lifecycle"] = "active"
                        deprecations.remove(automation_entry)
                        summary.reactivated.append(state_key)
            else:
                if model.get("lifecycle") == "retired":
                    continue  # already retired; nothing to track
                counter = state_models.get(state_key, {"missingRuns": 0})
                missing_runs = counter.get("missingRuns", 0)
                missing_runs = missing_runs + 1 if isinstance(missing_runs, int) else 1
                if missing_runs >= grace_runs:
                    model["lifecycle"] = "retired"
                    state_models.pop(state_key, None)
                    summary.retired.append(state_key)
                    if find_automation_deprecation(deprecations, provider, model_id) is None:
                        deprecations.append(
                            {
                                "provider": provider,
                                "modelID": model_id,
                                "replacedBy": None,
                                "message": (
                                    AUTO_DEPRECATION_PREFIX
                                    + f"not returned by the {provider} model list for "
                                    + f"{grace_runs} consecutive refreshes; retired automatically."
                                ),
                                "retiresAt": now,
                            }
                        )
                else:
                    state_models[state_key] = {"missingRuns": missing_runs}
                    summary.pending_drift.append(f"{state_key} (missingRuns={missing_runs})")

    # ---- merge public OpenAI pricing when available -------------------------
    if merge_pricing and not offline:
        pricing_payload = fetch_pricing_source(pricing_url, warnings)
        pricing_models = pricing_payload.get("models") if isinstance(pricing_payload, dict) else None
        if isinstance(pricing_models, dict):
            for provider_entry in providers:
                if provider_entry.get("provider") != "openai":
                    continue
                for model in provider_entry.get("models", []):
                    if not isinstance(model, dict) or not isinstance(model.get("id"), str):
                        continue
                    entry = pricing_models.get(model["id"])
                    merged, long_state = pricing_from_public_entry(entry)
                    if merged is None:
                        continue
                    current = model.get("pricing")
                    if isinstance(current, dict):
                        # The public source is authoritative for every rate it
                        # publishes; buckets it does not publish keep their
                        # existing catalog value (no invented numbers).
                        final = dict(current)
                        final["inputPerMTok"] = merged["inputPerMTok"]
                        final["outputPerMTok"] = merged["outputPerMTok"]
                        if "cacheReadPerMTok" in merged:
                            final["cacheReadPerMTok"] = merged["cacheReadPerMTok"]
                        if "cacheWritePerMTok" in merged:
                            final["cacheWritePerMTok"] = merged["cacheWritePerMTok"]
                        if long_state == "set":
                            final["longContext"] = merged.get("longContext")
                        elif long_state == "none":
                            final.pop("longContext", None)
                    else:
                        final = dict(merged)
                    if current != final:
                        model["pricing"] = final
                        summary.pricing_updated.append(f"openai:{model['id']}")
                    if model.get("details") == DETAILS_PENDING:
                        model["details"] = DETAILS_ADDED
        elif pricing_payload is None:
            warnings.append("pricing: public OpenAI pricing catalog unavailable; prices left unchanged")

    # ---- state / deprecation hygiene ----------------------------------------
    catalog_ids = set()
    for provider_entry in providers:
        provider = provider_entry.get("provider")
        for model in provider_entry.get("models", []):
            if isinstance(model, dict) and isinstance(model.get("id"), str):
                catalog_ids.add(f"{provider}:{model['id']}")
    state_models = {key: value for key, value in state_models.items() if key in catalog_ids}
    if state_models:
        catalog["_automation"] = {"schemaVersion": 1, "models": state_models}
    else:
        catalog.pop("_automation", None)
    catalog["deprecations"] = deprecations

    # generatedAt only moves when content actually changed.
    if content_without_timestamp(catalog) == content_without_timestamp(existing):
        return existing, summary
    catalog["generatedAt"] = now
    return catalog, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the Motion Glass model catalog from live provider data.",
    )
    parser.add_argument(
        "--catalog",
        default=str(default_catalog_path()),
        help="catalog JSON file to read/update (default: catalog.json next to the scripts dir)",
    )
    parser.add_argument("--check", action="store_true", help="report whether the catalog would change (exit 1 if so)")
    parser.add_argument("--offline", action="store_true", help="no network access; canonical-form check only")
    parser.add_argument("--dry-run", action="store_true", help="print the summary without writing the catalog")
    parser.add_argument(
        "--grace-runs",
        type=int,
        default=2,
        help="consecutive successful refreshes a model must be missing before it is retired (default 2)",
    )
    parser.add_argument("--admit-all", action="store_true", help="ignore the admission version floor (still rejects non-chat families)")
    parser.add_argument("--no-pricing", action="store_true", help="do not merge the public OpenAI pricing catalog")
    parser.add_argument("--pricing-source", default=DEFAULT_PRICING_URL, help="OpenAI pricing JSON URL")
    parser.add_argument("--anthropic-models-url", default=DEFAULT_ANTHROPIC_MODELS_URL)
    parser.add_argument("--openai-models-url", default=DEFAULT_OPENAI_MODELS_URL)
    parser.add_argument("--now", default=None, help="pin generatedAt (ISO-8601 UTC); for reproducible runs/tests")
    parser.add_argument("--quiet", action="store_true", help="only print warnings/errors and the final status")
    return parser.parse_args(argv)


def print_summary(summary: RefreshSummary, old_count: int, new_count: int) -> None:
    def short(items: list[str], limit: int = 12) -> str:
        shown = ", ".join(items[:limit])
        if len(items) > limit:
            shown += f", … (+{len(items) - limit})"
        return shown

    print(f"catalog: {old_count} models → {new_count} models")
    if summary.added:
        print(f"  added: {len(summary.added)} ({short(summary.added)})")
    if summary.pricing_updated:
        print(f"  pricing updated: {len(summary.pricing_updated)} ({short(summary.pricing_updated)})")
    if summary.reactivated:
        print(f"  reactivated: {len(summary.reactivated)} ({short(summary.reactivated)})")
    if summary.retired:
        print(f"  retired (grace exceeded): {len(summary.retired)} ({short(summary.retired)})")
    if summary.pending_drift:
        print(f"  drift pending (missing this run): {len(summary.pending_drift)} ({short(summary.pending_drift)})")
    if not summary.changed:
        print("  no content changes")


def count_models(catalog: dict[str, Any]) -> int:
    total = 0
    for provider in catalog.get("providers", []):
        if isinstance(provider, dict) and isinstance(provider.get("models"), list):
            total += len(provider["models"])
    return total


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    catalog_path = Path(args.catalog)
    if not catalog_path.is_file():
        print(f"error: catalog file not found: {catalog_path}", file=sys.stderr)
        return 2
    try:
        existing = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        print(f"error: {catalog_path}: invalid JSON: {error}", file=sys.stderr)
        return 2
    if not isinstance(existing, dict):
        print(f"error: {catalog_path}: catalog root must be a JSON object", file=sys.stderr)
        return 2

    api_keys = {
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        "openai": os.environ.get("OPENAI_API_KEY", "").strip(),
    }

    if not args.offline:
        if not api_keys["anthropic"] and not api_keys["openai"]:
            print(
                "error: neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set; "
                "refusing to refresh (use --offline for a no-network check)",
                file=sys.stderr,
            )
            return 2
        if not api_keys["anthropic"]:
            print("warning: ANTHROPIC_API_KEY is not set; Anthropic models will not be refreshed", file=sys.stderr)
        if not api_keys["openai"]:
            print("warning: OPENAI_API_KEY is not set; OpenAI models will not be refreshed", file=sys.stderr)

    grace_runs = max(1, args.grace_runs)
    now = args.now or utc_now_iso()
    try:
        refreshed, summary = refresh_catalog(
            existing,
            offline=args.offline,
            grace_runs=grace_runs,
            admit_all=args.admit_all,
            anthropic_models_url=args.anthropic_models_url,
            openai_models_url=args.openai_models_url,
            pricing_url=args.pricing_source,
            merge_pricing=not args.no_pricing,
            now=now,
            api_keys=api_keys,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for warning in summary.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    # Never write an invalid catalog.
    errors, validation_warnings = validate_catalog_document(refreshed)
    for warning in validation_warnings:
        print(f"warning: validate: {warning}", file=sys.stderr)
    if errors:
        for message in errors:
            print(f"error: validate: {message}", file=sys.stderr)
        return 2

    changed = content_without_timestamp(refreshed) != content_without_timestamp(existing)
    old_count = count_models(existing)
    new_count = count_models(refreshed)

    if args.check:
        if changed:
            print_summary(summary, old_count, new_count)
            print(f"catalog would change: {catalog_path}")
            return 1
        print("catalog is up to date")
        return 0

    if args.dry_run:
        print_summary(summary, old_count, new_count)
        print(f"dry run: no changes written to {catalog_path}")
        return 0

    if not changed:
        print_summary(summary, old_count, new_count)
        print(f"catalog is up to date: {catalog_path}")
        return 0

    rendered = render_catalog(refreshed)
    catalog_path.write_text(rendered, encoding="utf-8")
    print_summary(summary, old_count, new_count)
    print(f"wrote {catalog_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
