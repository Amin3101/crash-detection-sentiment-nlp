"""Accuracy and traceability requirements for the v3 research log explorer."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from log_explorer import read_log_map, track_steps, report_section, markdown_sections


def test_routes_and_source_integrity():
    data=read_log_map()
    assert len(data['tracks'])==4
    assert len(data['steps'])==15
    assert {t['id']:len(track_steps(data,t['id'])) for t in data['tracks']}=={
        'clean':8,'event':3,'alert':3,'full':1,
    }
    assert any(s['status']=='Failed attempt' for s in data['steps'])
    assert any(s['status']=='Partial' for s in data['steps'])
    for step in data['steps']:
        assert step['settings'] and step['metrics']
        assert all(m['scope'] and m['label'] and m['value'] for m in step['metrics'])
        assert (ROOT/'sources'/step['source']).exists()
        if step['section'] is not None:
            part=report_section(step['source'],step['section'])
            assert part is not None, (step['id'],step['source'],step['section'])
            assert part.startswith(f"## {step['section']}.")


def test_partial_run_not_mislabeled_as_final_test():
    steps={s['id']:s for s in read_log_map()['steps']}
    b=steps['B']
    assert b['status']=='Partial'
    assert '8/15' in b['short']
    assert all('Val' in m['label'] for m in b['metrics'])
    assert 'No independent test' in b['caveat']
    assert '0.7209' in [m['value'] for m in b['metrics']]


def test_verbatim_sections():
    text=(ROOT/'sources'/'log.md').read_text(encoding='utf-8')
    sections=markdown_sections(text)
    assert len(sections)>=12
    assert '0.7846' in report_section('log.md',8)
    assert '0.700' in report_section('log.md',2)
    assert 'FAILED' in report_section('log.md',4)


def test_removed_extra_page_and_kept_live_models():
    code=(ROOT/'app.py').read_text(encoding='utf-8')
    assert 'st.Page(video_explorer' not in code
    assert 'Scored Video Explorer' not in code
    assert 'st.Page(inference_console' in code
    assert 'st.Page(project_journey' in code
    assert 'def experiment_graph' in code
    assert 'Read the exact original source' in code
