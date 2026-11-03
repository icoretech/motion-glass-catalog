#!/usr/bin/env python3
"""Validate Motion Glass model catalog JSON.

Standard library only (no third-party dependencies), so it can run in a bare
GitHub Actions job or on a developer machine.

Usage:
    python3 validate_catalog.py [catalog.json ...]   # validate specific files
    python3 validate_catalog.py                      # validate the default files:
                                                     #   <repo>/catalog/catalog.json
                                                     #   <repo>/Packages/.../Resources/catalog.json
                                                     #   (public catalog repo: <repo>/catalog.json)

Checks include: required fields and types, provider/lifecycle enums, positive
prices, pricing completeness, ISO-8601 dates, URLs, duplicate model/provider ids,
deprecation records, the optional automation state block, and any `*Hex` fields
present (the current schema has none, but future additions are covered).

Exit codes: 0 = valid, 1 = invalid, 2 = no catalog file found / usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PROVIDERS = ("anthropic", "openai")
LIFECYCLES = ("active", "preview", "deprecated", "retired", "unknown")

ISO8601_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})$"
)
HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
URL_RE = re.compile(r"^https://[^\s]+$")

REQUIRED_MODEL_FIELDS = (
    "provider",
    "id",
    "displayName",
    "contextWindow",
    "maxOutputTokens",
    "lifecycle",
    "supportsStructuredJSON",
    "supportsVision",
    "supportsStreaming",
    "isRecommended",
)


def default_catalog_paths() -> list[Path]:
    """Candidate catalog files, from most specific to most generic."""
    script_dir = Path(__file__).resolve().parent
    public_repo_root = script_dir.parent  # scripts/.. == (public) repo root
    app_repo_root = script_dir.parents[1]  # app repo: catalog/scripts -> repo root
    return [
        public_repo_root / "catalog.json",
        app_repo_root / "Packages/MotionGlassKit/Sources/MotionGlassKit/Resources/catalog.json",
    ]


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_pricing(pricing: Any, where: str, errors: list[str], warnings: list[str]) -> None:
    if pricing is None:
        # A null price is valid: it means "unknown", and the app marks the cost
        # estimate as unavailable instead of inventing a number.
        return
    if not isinstance(pricing, dict):
        errors.append(f"{where}: pricing must be an object or null")
        return
    for field in ("inputPerMTok", "outputPerMTok"):
        if field not in pricing:
            errors.append(f"{where}.pricing: missing required '{field}'")
        elif not _is_positive_number(pricing[field]):
            errors.append(f"{where}.pricing.{field}: must be a positive number, got {pricing[field]!r}")
    for field in ("cacheReadPerMTok", "cacheWritePerMTok"):
        if field in pricing and pricing[field] is not None and not _is_positive_number(pricing[field]):
            errors.append(f"{where}.pricing.{field}: must be a positive number, got {pricing[field]!r}")
    if not any(pricing.get(field) is not None for field in ("cacheReadPerMTok", "cacheWritePerMTok")):
        warnings.append(f"{where}.pricing: no prompt-cache rates published")

    long_context = pricing.get("longContext")
    if long_context is None:
        return
    if not isinstance(long_context, dict):
        errors.append(f"{where}.pricing.longContext: must be an object or null")
        return
    threshold = long_context.get("inputTokensAbove")
    if not _is_positive_int(threshold):
        errors.append(f"{where}.pricing.longContext.inputTokensAbove: must be a positive integer")
    for field in ("inputPerMTok", "outputPerMTok"):
        if not _is_positive_number(long_context.get(field)):
            errors.append(f"{where}.pricing.longContext.{field}: must be a positive number")
    for field in ("cacheReadPerMTok", "cacheWritePerMTok"):
        value = long_context.get(field)
        if value is not None and not _is_positive_number(value):
            errors.append(f"{where}.pricing.longContext.{field}: must be a positive number when present")


def _validate_model(
    model: Any, provider: str, where: str, errors: list[str], warnings: list[str]
) -> None:
    if not isinstance(model, dict):
        errors.append(f"{where}: model entry must be an object")
        return
    for field in REQUIRED_MODEL_FIELDS:
        if field not in model:
            errors.append(f"{where}: missing required field '{field}'")
    if model.get("provider") != provider:
        errors.append(f"{where}: model.provider {model.get('provider')!r} does not match provider '{provider}'")

    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        errors.append(f"{where}: model.id must be a non-empty string")
    display_name = model.get("displayName")
    if not isinstance(display_name, str) or not display_name.strip():
        errors.append(f"{where}: model.displayName must be a non-empty string")

    for field in ("contextWindow", "maxOutputTokens"):
        if not _is_positive_int(model.get(field)):
            errors.append(f"{where}.{field}: must be a positive integer")
    context_window = model.get("contextWindow")
    max_output = model.get("maxOutputTokens")
    if _is_positive_int(context_window) and _is_positive_int(max_output) and max_output > context_window:
        errors.append(f"{where}: maxOutputTokens ({max_output}) is larger than contextWindow ({context_window})")

    if model.get("lifecycle") not in LIFECYCLES:
        errors.append(f"{where}.lifecycle: invalid value {model.get('lifecycle')!r}; expected one of {LIFECYCLES}")
    for field in ("supportsStructuredJSON", "supportsVision", "supportsStreaming", "isRecommended"):
        if not isinstance(model.get(field), bool):
            errors.append(f"{where}.{field}: must be a boolean")

    released_at = model.get("releasedAt")
    if released_at is not None and (not isinstance(released_at, str) or not ISO8601_RE.match(released_at)):
        errors.append(f"{where}.releasedAt: must be an ISO-8601 UTC timestamp")

    replacement = model.get("replacementID")
    if replacement is not None and (not isinstance(replacement, str) or not replacement.strip()):
        errors.append(f"{where}.replacementID: must be a non-empty string when present")

    details = model.get("details")
    if details is not None and not isinstance(details, str):
        errors.append(f"{where}.details: must be a string when present")

    if model.get("lifecycle") == "retired" and model.get("isRecommended") is True:
        warnings.append(f"{where}: retired model is marked isRecommended")

    _validate_pricing(model.get("pricing"), f"{where}", errors, warnings)


def _validate_deprecation(entry: Any, where: str, errors: list[str]) -> None:
    if not isinstance(entry, dict):
        errors.append(f"{where}: deprecation entry must be an object")
        return
    if entry.get("provider") not in PROVIDERS:
        errors.append(f"{where}.provider: invalid provider {entry.get('provider')!r}")
    if not isinstance(entry.get("modelID"), str) or not entry.get("modelID", "").strip():
        errors.append(f"{where}.modelID: must be a non-empty string")
    if not isinstance(entry.get("message"), str) or not entry.get("message", "").strip():
        errors.append(f"{where}.message: must be a non-empty string")
    replaced_by = entry.get("replacedBy")
    if replaced_by is not None and (not isinstance(replaced_by, str) or not replaced_by.strip()):
        errors.append(f"{where}.replacedBy: must be a non-empty string when present")
    retires_at = entry.get("retiresAt")
    if retires_at is not None and (not isinstance(retires_at, str) or not ISO8601_RE.match(retires_at)):
        errors.append(f"{where}.retiresAt: must be an ISO-8601 UTC timestamp when present")


def _validate_automation(block: Any, errors: list[str]) -> None:
    if block is None:
        return
    if not isinstance(block, dict):
        errors.append("_automation: must be an object when present")
        return
    if not isinstance(block.get("schemaVersion"), int) or isinstance(block.get("schemaVersion"), bool):
        errors.append("_automation.schemaVersion: must be an integer")
    models = block.get("models")
    if models is None:
        return
    if not isinstance(models, dict):
        errors.append("_automation.models: must be an object")
        return
    for key, value in models.items():
        if not isinstance(key, str) or ":" not in key:
            errors.append(f"_automation.models: key {key!r} must be 'provider:model-id'")
            continue
        provider, _, model_id = key.partition(":")
        if provider not in PROVIDERS or not model_id:
            errors.append(f"_automation.models: invalid key {key!r}")
        if not isinstance(value, dict):
            errors.append(f"_automation.models[{key!r}]: must be an object")
            continue
        missing_runs = value.get("missingRuns")
        if not isinstance(missing_runs, int) or isinstance(missing_runs, bool) or missing_runs < 0:
            errors.append(f"_automation.models[{key!r}].missingRuns: must be a non-negative integer")


def _validate_hex_fields(value: Any, path: str, errors: list[str]) -> None:
    """Validate any `*Hex` field (the current schema has none, future-proofing)."""
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if isinstance(key, str) and key.endswith("Hex") and isinstance(child, str):
                if not HEX_RE.match(child):
                    errors.append(f"{child_path}: '{child}' is not a #RRGGBB hex color")
            else:
                _validate_hex_fields(child, child_path, errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_hex_fields(child, f"{path}[{index}]", errors)


def validate_catalog_document(catalog: Any) -> tuple[list[str], list[str]]:
    """Validate one decoded catalog document. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(catalog, dict):
        return ["catalog: root must be a JSON object"], []

    schema_version = catalog.get("schemaVersion")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        errors.append("schemaVersion: must be an integer")
    elif schema_version != SCHEMA_VERSION:
        warnings.append(f"schemaVersion: {schema_version} (this validator knows version {SCHEMA_VERSION})")

    generated_at = catalog.get("generatedAt")
    if not isinstance(generated_at, str) or not ISO8601_RE.match(generated_at):
        errors.append("generatedAt: must be an ISO-8601 UTC timestamp")

    source_url = catalog.get("sourceURL")
    if source_url is not None and (not isinstance(source_url, str) or not URL_RE.match(source_url)):
        errors.append("sourceURL: must be an https URL when present")

    providers = catalog.get("providers")
    if not isinstance(providers, list) or not providers:
        errors.append("providers: must be a non-empty list")
        providers = []

    seen_providers: set[str] = set()
    seen_ids: set[str] = set()
    for provider_index, provider_entry in enumerate(providers):
        where = f"providers[{provider_index}]"
        if not isinstance(provider_entry, dict):
            errors.append(f"{where}: must be an object")
            continue
        provider = provider_entry.get("provider")
        if provider not in PROVIDERS:
            errors.append(f"{where}.provider: invalid provider {provider!r}")
            continue
        if provider in seen_providers:
            errors.append(f"{where}.provider: duplicate provider {provider!r}")
        seen_providers.add(provider)
        for field in ("displayName",):
            if not isinstance(provider_entry.get(field), str) or not provider_entry.get(field, "").strip():
                errors.append(f"{where}.{field}: must be a non-empty string")
        for field in ("apiBaseURL", "docsURL"):
            value = provider_entry.get(field)
            if not isinstance(value, str) or not URL_RE.match(value):
                errors.append(f"{where}.{field}: must be an https URL")
        models = provider_entry.get("models")
        if not isinstance(models, list):
            errors.append(f"{where}.models: must be a list")
            continue
        for model_index, model in enumerate(models):
            model_where = f"{where}.models[{model_index}]"
            _validate_model(model, provider, model_where, errors, warnings)
            if isinstance(model, dict):
                model_id = model.get("id")
                if isinstance(model_id, str):
                    key = f"{provider}:{model_id}"
                    if key in seen_ids:
                        errors.append(f"{model_where}: duplicate model id {key!r}")
                    seen_ids.add(key)

    deprecations = catalog.get("deprecations", [])
    if deprecations is None:
        warnings.append("deprecations: missing; treating as empty")
        deprecations = []
    if not isinstance(deprecations, list):
        errors.append("deprecations: must be a list")
        deprecations = []
    seen_deprecations: set[str] = set()
    for index, entry in enumerate(deprecations):
        _validate_deprecation(entry, f"deprecations[{index}]", errors)
        if isinstance(entry, dict):
            key = f"{entry.get('provider')}:{entry.get('modelID')}"
            if key in seen_deprecations:
                errors.append(f"deprecations[{index}]: duplicate deprecation {key!r}")
            seen_deprecations.add(key)

    _validate_automation(catalog.get("_automation"), errors)
    _validate_hex_fields(catalog, "", errors)

    return errors, warnings


def load_catalog(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_file(path: Path) -> tuple[list[str], list[str], int]:
    document = load_catalog(path)
    errors, warnings = validate_catalog_document(document)
    model_count = 0
    if isinstance(document, dict) and isinstance(document.get("providers"), list):
        for provider in document["providers"]:
            if isinstance(provider, dict) and isinstance(provider.get("models"), list):
                model_count += len(provider["models"])
    return errors, warnings, model_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Motion Glass model catalog JSON.")
    parser.add_argument(
        "files",
        nargs="*",
        help="catalog JSON files to validate; defaults to the bundled + hosted snapshots",
    )
    parser.add_argument("--quiet", action="store_true", help="only print errors")
    args = parser.parse_args(argv)

    if args.files:
        paths = [Path(name) for name in args.files]
    else:
        paths = [path for path in default_catalog_paths() if path.is_file()]
        if not paths:
            print(
                "error: no catalog file found; pass paths explicitly or run from the app/catalog repo",
                file=sys.stderr,
            )
            return 2

    exit_code = 0
    for path in paths:
        if not path.is_file():
            print(f"error: {path}: file not found", file=sys.stderr)
            exit_code = 1
            continue
        try:
            errors, warnings, model_count = validate_file(path)
        except json.JSONDecodeError as error:
            print(f"error: {path}: invalid JSON: {error}", file=sys.stderr)
            exit_code = 1
            continue
        if errors:
            for message in errors:
                print(f"error: {path}: {message}", file=sys.stderr)
            exit_code = 1
        elif not args.quiet:
            try:
                shown = path.resolve().relative_to(Path.cwd().resolve())
            except ValueError:
                shown = path.resolve()
            print(f"OK {shown} · {model_count} models")
        if warnings and not args.quiet:
            for message in warnings:
                print(f"warning: {path}: {message}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
