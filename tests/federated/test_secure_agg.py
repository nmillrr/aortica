"""Tests for aortica.federated.secure_agg — Secure Aggregation.

Tests cover:
- SecureAggConfig (defaults, validation)
- SecureAggregator in mock mode (encrypt, decrypt, aggregate)
- Encrypted aggregation matches plaintext (within tolerance)
- Empty/edge cases
- Imports
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# SecureAggConfig
# ---------------------------------------------------------------------------


class TestSecureAggConfig:

    def test_defaults(self) -> None:
        from aortica.federated.secure_agg import SecureAggConfig
        cfg = SecureAggConfig()
        assert cfg.poly_modulus_degree == 8192
        assert cfg.coeff_mod_bit_sizes == [60, 40, 40, 60]
        assert cfg.global_scale == 2**40
        assert cfg.use_mock is False

    def test_invalid_poly_modulus(self) -> None:
        from aortica.federated.secure_agg import SecureAggConfig
        with pytest.raises(ValueError, match="poly_modulus_degree"):
            SecureAggConfig(poly_modulus_degree=512)

    def test_to_dict(self) -> None:
        from aortica.federated.secure_agg import SecureAggConfig
        d = SecureAggConfig().to_dict()
        assert d["poly_modulus_degree"] == 8192

    def test_custom_coeff_mod(self) -> None:
        from aortica.federated.secure_agg import SecureAggConfig
        cfg = SecureAggConfig(coeff_mod_bit_sizes=[40, 20, 40])
        assert cfg.coeff_mod_bit_sizes == [40, 20, 40]


# ---------------------------------------------------------------------------
# SecureAggregator — mock mode
# ---------------------------------------------------------------------------


class TestSecureAggregatorMock:

    def test_construction_mock(self) -> None:
        from aortica.federated.secure_agg import SecureAggregator, SecureAggConfig
        agg = SecureAggregator(SecureAggConfig(use_mock=True))
        assert agg.context is None

    def test_encrypt_decrypt_roundtrip(self) -> None:
        from aortica.federated.secure_agg import SecureAggregator, SecureAggConfig
        agg = SecureAggregator(SecureAggConfig(use_mock=True))
        params = [np.array([1.0, 2.0, 3.0]), np.array([[4.0, 5.0], [6.0, 7.0]])]
        encrypted = agg.encrypt(params)
        decrypted = agg.decrypt(encrypted)
        assert len(decrypted) == 2
        np.testing.assert_array_almost_equal(decrypted[0], params[0])
        np.testing.assert_array_almost_equal(decrypted[1], params[1])

    def test_encrypt_preserves_shapes(self) -> None:
        from aortica.federated.secure_agg import SecureAggregator, SecureAggConfig
        agg = SecureAggregator(SecureAggConfig(use_mock=True))
        params = [np.zeros((3, 4, 5))]
        encrypted = agg.encrypt(params)
        assert encrypted.shapes == [(3, 4, 5)]
        decrypted = agg.decrypt(encrypted)
        assert decrypted[0].shape == (3, 4, 5)

    def test_aggregate_two_clients(self) -> None:
        from aortica.federated.secure_agg import SecureAggregator, SecureAggConfig
        agg = SecureAggregator(SecureAggConfig(use_mock=True))
        p1 = [np.array([1.0, 2.0, 3.0])]
        p2 = [np.array([3.0, 4.0, 5.0])]
        enc1 = agg.encrypt(p1)
        enc2 = agg.encrypt(p2)
        agg_enc = agg.aggregate_encrypted([enc1, enc2])
        result = agg.decrypt(agg_enc)
        # Equal-weight average: [2.0, 3.0, 4.0]
        np.testing.assert_array_almost_equal(result[0], [2.0, 3.0, 4.0])

    def test_aggregate_weighted(self) -> None:
        from aortica.federated.secure_agg import SecureAggregator, SecureAggConfig
        agg = SecureAggregator(SecureAggConfig(use_mock=True))
        p1 = [np.array([0.0, 0.0])]
        p2 = [np.array([10.0, 10.0])]
        enc1 = agg.encrypt(p1)
        enc2 = agg.encrypt(p2)
        # Weight 3:1 toward p2
        agg_enc = agg.aggregate_encrypted([enc1, enc2], weights=[1.0, 3.0])
        result = agg.decrypt(agg_enc)
        np.testing.assert_array_almost_equal(result[0], [7.5, 7.5])

    def test_aggregate_single_client(self) -> None:
        from aortica.federated.secure_agg import SecureAggregator, SecureAggConfig
        agg = SecureAggregator(SecureAggConfig(use_mock=True))
        p = [np.array([1.0, 2.0])]
        enc = agg.encrypt(p)
        agg_enc = agg.aggregate_encrypted([enc])
        result = agg.decrypt(agg_enc)
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0])

    def test_aggregate_empty_raises(self) -> None:
        from aortica.federated.secure_agg import SecureAggregator, SecureAggConfig
        agg = SecureAggregator(SecureAggConfig(use_mock=True))
        with pytest.raises(ValueError, match="empty"):
            agg.aggregate_encrypted([])

    def test_aggregate_matches_plaintext(self) -> None:
        from aortica.federated.secure_agg import SecureAggregator, SecureAggConfig
        agg = SecureAggregator(SecureAggConfig(use_mock=True))

        np.random.seed(42)
        p1 = [np.random.randn(10, 5).astype(np.float32)]
        p2 = [np.random.randn(10, 5).astype(np.float32)]
        p3 = [np.random.randn(10, 5).astype(np.float32)]

        # Plaintext average
        expected = [(p1[0] + p2[0] + p3[0]) / 3.0]

        # Encrypted average
        enc1, enc2, enc3 = agg.encrypt(p1), agg.encrypt(p2), agg.encrypt(p3)
        agg_enc = agg.aggregate_encrypted([enc1, enc2, enc3])
        result = agg.decrypt(agg_enc)

        np.testing.assert_array_almost_equal(result[0], expected[0], decimal=5)

    def test_repr(self) -> None:
        from aortica.federated.secure_agg import SecureAggregator, SecureAggConfig
        agg = SecureAggregator(SecureAggConfig(use_mock=True))
        assert "mock" in repr(agg)

    def test_non_mock_requires_tenseal(self) -> None:
        from aortica.federated.secure_agg import HAS_TENSEAL, SecureAggregator, SecureAggConfig
        if HAS_TENSEAL:
            pytest.skip("TenSEAL is installed")
        with pytest.raises(ImportError, match="TenSEAL"):
            SecureAggregator(SecureAggConfig(use_mock=False))


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:

    def test_import_module(self) -> None:
        import aortica.federated.secure_agg  # noqa: F401

    def test_import_from_package(self) -> None:
        from aortica.federated import SecureAggregator, SecureAggConfig  # noqa: F401

    def test_import_encrypted_weights(self) -> None:
        from aortica.federated import EncryptedWeights  # noqa: F401
