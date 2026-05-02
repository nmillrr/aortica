"""Secure aggregation via CKKS homomorphic encryption.

Provides :class:`SecureAggregator` which encrypts model weight updates
using the CKKS scheme (via TenSEAL) for privacy-preserving aggregation.

When TenSEAL is not available, falls back to a plaintext mock mode
for development and testing.

Example::

    from aortica.federated.secure_agg import SecureAggregator, SecureAggConfig

    agg = SecureAggregator(SecureAggConfig())
    enc_params = agg.encrypt(model_params)
    aggregated = agg.aggregate_encrypted([enc_params_1, enc_params_2])
    result = agg.decrypt(aggregated)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency check
# ---------------------------------------------------------------------------

try:
    import tenseal as ts  # type: ignore[import-untyped]

    HAS_TENSEAL = True
except ImportError:
    HAS_TENSEAL = False


def _check_tenseal() -> None:
    """Raise ``ImportError`` if TenSEAL is not installed."""
    if not HAS_TENSEAL:
        raise ImportError(
            "TenSEAL is required for secure aggregation. "
            "Install with: pip install tenseal"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SecureAggConfig:
    """Configuration for CKKS-based secure aggregation.

    Attributes:
        poly_modulus_degree: Polynomial modulus degree for CKKS.
            Higher values = more security but slower.  Default ``8192``.
        coeff_mod_bit_sizes: Coefficient modulus bit sizes for CKKS.
            Default ``[60, 40, 40, 60]`` (128-bit security).
        global_scale: Global scale for CKKS encoding.  Default ``2**40``.
        use_mock: If ``True``, skip encryption and use plaintext
            (for testing without TenSEAL).  Default ``False``.
    """

    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = None  # type: ignore[assignment]
    global_scale: float = 2**40
    use_mock: bool = False

    def __post_init__(self) -> None:
        if self.coeff_mod_bit_sizes is None:
            self.coeff_mod_bit_sizes = [60, 40, 40, 60]
        if self.poly_modulus_degree < 1024:
            raise ValueError("poly_modulus_degree must be >= 1024")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Encrypted weight container
# ---------------------------------------------------------------------------


@dataclass
class EncryptedWeights:
    """Container for encrypted model weights.

    Attributes:
        encrypted_vectors: List of encrypted vectors (TenSEAL CKKSVectors
            or plain numpy arrays in mock mode).
        shapes: Original shapes of each parameter array.
        num_params: Total number of parameter arrays.
    """

    encrypted_vectors: List[Any]
    shapes: List[Tuple[int, ...]]
    num_params: int


# ---------------------------------------------------------------------------
# SecureAggregator
# ---------------------------------------------------------------------------


class SecureAggregator:
    """CKKS homomorphic encryption for secure weight aggregation.

    Encrypts model parameter updates client-side, enables aggregation
    in the encrypted domain server-side, and decrypts the result back
    on the client.

    Args:
        config: Secure aggregation configuration.

    Example::

        agg = SecureAggregator()
        enc = agg.encrypt(client_weights)
        agg_result = agg.aggregate_encrypted([enc1, enc2, enc3])
        plaintext = agg.decrypt(agg_result)
    """

    def __init__(self, config: Optional[SecureAggConfig] = None) -> None:
        self._config = config or SecureAggConfig()
        self._context: Any = None

        if not self._config.use_mock:
            self._init_context()

    @property
    def config(self) -> SecureAggConfig:
        """Return the configuration."""
        return self._config

    @property
    def context(self) -> Any:
        """Return the TenSEAL context (None in mock mode)."""
        return self._context

    def _init_context(self) -> None:
        """Initialise the TenSEAL CKKS context."""
        if self._config.use_mock:
            return

        _check_tenseal()

        self._context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self._config.poly_modulus_degree,
            coeff_mod_bit_sizes=self._config.coeff_mod_bit_sizes,
        )
        self._context.global_scale = self._config.global_scale
        self._context.generate_galois_keys()

        logger.info(
            "CKKS context initialised: poly_mod=%d, scale=2^%.0f",
            self._config.poly_modulus_degree,
            np.log2(self._config.global_scale),
        )

    def encrypt(self, parameters: List[np.ndarray]) -> EncryptedWeights:
        """Encrypt model parameters.

        Each parameter array is flattened and encrypted as a CKKS vector.

        Args:
            parameters: List of numpy arrays (model weights).

        Returns:
            :class:`EncryptedWeights` containing encrypted vectors.
        """
        shapes = [p.shape for p in parameters]
        encrypted_vectors: List[Any] = []

        for param in parameters:
            flat = param.flatten().astype(np.float64).tolist()

            if self._config.use_mock:
                # Mock mode: store as plain list
                encrypted_vectors.append(flat)
            else:
                _check_tenseal()
                enc_vec = ts.ckks_vector(self._context, flat)
                encrypted_vectors.append(enc_vec)

        return EncryptedWeights(
            encrypted_vectors=encrypted_vectors,
            shapes=shapes,
            num_params=len(parameters),
        )

    def decrypt(self, encrypted: EncryptedWeights) -> List[np.ndarray]:
        """Decrypt encrypted weights back to numpy arrays.

        Args:
            encrypted: :class:`EncryptedWeights` to decrypt.

        Returns:
            List of numpy arrays with original shapes restored.
        """
        result: List[np.ndarray] = []

        for enc_vec, shape in zip(encrypted.encrypted_vectors, encrypted.shapes):
            if self._config.use_mock:
                flat = np.array(enc_vec, dtype=np.float32)
            else:
                _check_tenseal()
                flat = np.array(enc_vec.decrypt(), dtype=np.float32)

            # Truncate to original size (CKKS may pad)
            n_elements = int(np.prod(shape))
            flat = flat[:n_elements]
            result.append(flat.reshape(shape))

        return result

    def aggregate_encrypted(
        self,
        encrypted_list: List[EncryptedWeights],
        weights: Optional[List[float]] = None,
    ) -> EncryptedWeights:
        """Aggregate multiple encrypted weight sets in the encrypted domain.

        Computes a weighted average of encrypted weight vectors without
        decrypting them (homomorphic addition and scalar multiplication).

        Args:
            encrypted_list: List of :class:`EncryptedWeights` from clients.
            weights: Optional per-client weights for weighted averaging.
                If ``None``, uses equal weights.

        Returns:
            Aggregated :class:`EncryptedWeights`.

        Raises:
            ValueError: If the list is empty or shapes don't match.
        """
        if not encrypted_list:
            raise ValueError("encrypted_list must not be empty")

        n_clients = len(encrypted_list)
        if weights is None:
            client_weights = [1.0 / n_clients] * n_clients
        else:
            total_w = sum(weights)
            client_weights = [w / total_w for w in weights]

        num_params = encrypted_list[0].num_params
        shapes = encrypted_list[0].shapes

        aggregated_vectors: List[Any] = []

        for param_idx in range(num_params):
            if self._config.use_mock:
                # Plaintext weighted average
                acc = np.array(
                    encrypted_list[0].encrypted_vectors[param_idx],
                    dtype=np.float64,
                ) * client_weights[0]

                for client_idx in range(1, n_clients):
                    arr = np.array(
                        encrypted_list[client_idx].encrypted_vectors[param_idx],
                        dtype=np.float64,
                    )
                    acc += arr * client_weights[client_idx]

                aggregated_vectors.append(acc.tolist())
            else:
                _check_tenseal()
                # Homomorphic weighted average
                acc = encrypted_list[0].encrypted_vectors[param_idx] * client_weights[0]

                for client_idx in range(1, n_clients):
                    vec = encrypted_list[client_idx].encrypted_vectors[param_idx]
                    acc += vec * client_weights[client_idx]

                aggregated_vectors.append(acc)

        return EncryptedWeights(
            encrypted_vectors=aggregated_vectors,
            shapes=shapes,
            num_params=num_params,
        )

    def __repr__(self) -> str:
        mode = "mock" if self._config.use_mock else "CKKS"
        return (
            f"SecureAggregator(scheme={mode}, "
            f"poly_mod={self._config.poly_modulus_degree})"
        )
