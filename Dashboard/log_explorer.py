"""Read-only report mapping for CrashLens. No synthetic epochs or evaluation scores."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_log_map() -> dict:
    payload = json.loads((ROOT / 'data' / 'log_map.json').read_text(encoding='utf-8'))
    ids = [s['id'] for s in payload['steps']]
    assert len(ids) == len(set(ids)), 'Duplicate stage IDs'
    tracks = {t['id'] for t in payload['tracks']}
    for s in payload['steps']:
        assert s['track'] in tracks
        assert (ROOT / 'sources' / s['source']).exists()
        assert s['status'] in {'Recorded','Failed attempt','Partial'}
    return payload


def markdown_sections(text: str) -> list[tuple[str, str]]:
    """Preserve exactly recorded Markdown, including code and tables."""
    matches = list(re.finditer(r'^##\s+([^\n]+)\s*$', text, re.MULTILINE))
    if not matches:
        return [('Complete report', text)]
    sections: list[tuple[str,str]] = []
    if text[:matches[0].start()].strip():
        sections.append(('Introduction', text[:matches[0].start()]))
    for idx, match in enumerate(matches):
        end = matches[idx+1].start() if idx+1 < len(matches) else len(text)
        sections.append((match.group(1).strip(), text[match.start():end]))
    return sections


def report_section(source: str, section_number: int | None) -> str | None:
    if section_number is None or not source.endswith('.md'):
        return None
    content = (ROOT / 'sources' / source).read_text(encoding='utf-8')
    for title, body in markdown_sections(content):
        if re.match(rf'^{section_number}\s*[.\-]', title):
            return body
    return None


def track_steps(payload: dict, track_id: str) -> list[dict]:
    return [s for s in payload['steps'] if s['track'] == track_id]


def matching_catalog(catalog, source: str):
    return catalog[catalog['source'] == source].copy()
