"""Sanity checks: archival counts and task boundaries must remain intact."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def test_history_epochs_and_sources():
    hist=pd.read_csv(ROOT/'data/experiment_history.csv')
    assert len(hist)==68
    assert hist.groupby('run').size().to_dict()=={
        'Clean pre-event MViT · run 1':7,
        'Event-centered detection · Linear head':7,
        'Event-centered detection · MLP head':7,
        'Full-video detection · MViT temporal head':15,
        'Original temporal sample':9,
        'Temporal anticipation · Attention pooling (A)':15,
        'Temporal anticipation · Transformer + CNN (B)':8,
    }
    assert hist['source'].notna().all()
    assert (hist['val_auc'].dropna().between(0,1)).all()


def test_timeline_and_catalog_traceable():
    catalog=pd.read_csv(ROOT/'data/experiment_catalog.csv')
    timeline=pd.read_csv(ROOT/'data/project_timeline.csv')
    assert len(catalog)==7 and len(timeline)==13
    assert all((ROOT/'sources'/x).exists() for x in catalog['source'].dropna())
    assert 'Partial (8/15' in catalog.loc[catalog.run.str.contains('CNN \\(B\\)',regex=True),'status'].iloc[0]
    assert 'Failed attempt' in set(timeline['status'])


def test_distinct_checkpoint_destinations():
    from pathlib import Path
    base=ROOT/'models'
    assert (base/'detection.pth') != (base/'anticipation.pth')
    assert not (base/'detection.pth').exists()
    assert not (base/'anticipation.pth').exists()
