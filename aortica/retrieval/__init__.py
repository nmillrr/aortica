"""Aortica Retrieval — case-based ECG retrieval via latent space indexing.

Provides approximate nearest-neighbor search over de-identified PhysioNet ECGs
using backbone feature vectors.  The index enables clinicians to retrieve
historically similar ECGs with verified diagnoses for any new prediction.

Functions:
    build_index: Encode ECGs through the backbone and build an ANN index.
    retrieve_similar: Query the index for top-K similar ECGs.
"""

from __future__ import annotations

from aortica.retrieval.index_builder import build_index

__all__ = ["build_index"]
