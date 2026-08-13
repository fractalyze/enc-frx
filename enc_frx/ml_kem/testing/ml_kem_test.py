# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The FO transform against CCTV's published run, and around the rejection path.

The published files carry the whole of one encapsulation and its decapsulation —
`ek`, `dk`, `m`, `c`, `K`, and `KBar` — so `encaps` and `decaps` are gated on
values someone else computed, at every parameter set. Key generation is not: the
files predate FIPS 203 Algorithm 13's `G(d ‖ k)` and so cannot reproduce a key
from `d` (see [`docs/schemes/ml-kem.md`](../../../docs/schemes/ml-kem.md)), which
leaves the *layout* of `dk` as what this file gates, the lattice work to
[`_k_pke_test.py`](_k_pke_test.py), and key generation from `d` to
[`acvp_test.py`](acvp_test.py).

**`KBar` is the published implicit-rejection secret** — `J(z ‖ c)` for the file's
own `c`. Reaching it needs a `dk` that fails a check while `z` and `c` stay
intact, which a corrupted *key* gives and a corrupted ciphertext cannot: change
`c` and the expected secret changes with it. So both negatives are here, one
compared against the published value and one against `hashlib`.

Nothing here is self-consistency. A `decaps` that returned `J(z ‖ c)` for every
ciphertext, valid ones included, round-trips against itself perfectly.
"""

from __future__ import annotations

import functools
import hashlib
import re
from unittest import mock

import frx
import numpy as np
from absl.testing import absltest, parameterized

from enc_frx.kem import Kem
from enc_frx.ml_kem import _k_pke, encoding, sampling
from enc_frx.ml_kem.ml_kem import MlKem
from enc_frx.ml_kem.params import (
    ML_KEM_512,
    ML_KEM_768,
    ML_KEM_1024,
    PARAMETER_SETS,
    SEED_SIZE,
    MlKemParams,
)
from enc_frx.ml_kem.testing import cctv_vectors
from enc_frx.testing.kat import to_bytes

_NAMED = tuple((params.name, params) for params in PARAMETER_SETS)

# FIPS 203 Table 3, transcribed: `(encapsulation key, decapsulation key,
# ciphertext)`. The scheme computes these from `k`, `du` and `dv`, so a formula
# and a transcription of the answer are what have to agree.
_TABLE_3 = {
    ML_KEM_512: (800, 1632, 768),
    ML_KEM_768: (1184, 2400, 1088),
    ML_KEM_1024: (1568, 3168, 1568),
}

# ML-KEM-768 for the cases that are about the transform rather than about the
# numbers, so they run once instead of three times.
_768 = MlKem(ML_KEM_768)
_BATCH = 4

# `cond` and `while` as jaxpr primitives — a decapsulation that branched on
# validity would show up as one of these.
_BRANCH = re.compile(r"\b(cond|while)\[")


def _published(params: MlKemParams, name: str, index: int = 0) -> np.ndarray:
    return cctv_vectors.intermediate(params.name).array_at(name, index)


def _j(z: np.ndarray, c: np.ndarray) -> bytes:
    """`J(z ‖ c) = SHAKE256(z ‖ c, 32)`, §4.1 — the rejection secret, from
    `hashlib` rather than from this repo's own Keccak."""
    return hashlib.shake_256(bytes(z) + bytes(c)).digest(SEED_SIZE)


def _flip(value: np.ndarray, index: int) -> np.ndarray:
    out = value.copy()
    out[..., index] ^= 1
    return out


class PublishedRunTest(parameterized.TestCase):
    """Every value FIPS 203 names in one run of the transform, as published."""

    @parameterized.named_parameters(*_NAMED)
    def test_encaps_matches_the_published_ciphertext_and_secret(
        self, params: MlKemParams
    ) -> None:
        c, shared = MlKem(params).encaps(
            _published(params, "ek"), randomness=_published(params, "m")
        )
        vectors = cctv_vectors.intermediate(params.name)
        self.assertEqual(to_bytes(c), vectors.hex_at("c"))
        self.assertEqual(to_bytes(shared), vectors.hex_at("K"))

    @parameterized.named_parameters(*_NAMED)
    def test_encaps_internal_is_the_same_call_under_its_own_name(
        self, params: MlKemParams
    ) -> None:
        # The seam's randomness argument and the standard's derandomized entry
        # point are the same operation here, and a harness driving one must not
        # be driving something else.
        scheme = MlKem(params)
        ek, m = _published(params, "ek"), _published(params, "m")
        self.assertEqual(
            [to_bytes(part) for part in scheme.encaps(ek, randomness=m)],
            [to_bytes(part) for part in scheme.encaps_internal(ek, m)],
        )

    @parameterized.named_parameters(*_NAMED)
    def test_decaps_recovers_the_published_secret(self, params: MlKemParams) -> None:
        got = MlKem(params).decaps(_published(params, "dk"), _published(params, "c"))
        self.assertEqual(
            to_bytes(got), cctv_vectors.intermediate(params.name).hex_at("K")
        )

    @parameterized.named_parameters(*_NAMED)
    def test_a_key_whose_halves_disagree_yields_the_published_rejection_secret(
        self, params: MlKemParams
    ) -> None:
        """`KBar` exactly — the one negative the files publish a value for.

        Corrupting `H(ek)` inside `dk` fails FIPS 203 §7.3's hash check while
        leaving `z` and `c` untouched, so the expected secret is still `J(z ‖ c)`
        — which the file publishes as `KBar`. A ciphertext corruption cannot
        reach it: it changes the value being derived.
        """
        scheme = MlKem(params)
        dk = _published(params, "dk").copy()
        # `dk = dk_PKE ‖ ek ‖ H(ek) ‖ z`, so the hash starts 64 bytes from the end.
        dk[-2 * SEED_SIZE] ^= 1
        got = scheme.decaps(dk, _published(params, "c"))
        self.assertEqual(
            to_bytes(got), cctv_vectors.intermediate(params.name).hex_at("KBar")
        )

    @parameterized.named_parameters(*_NAMED)
    def test_a_corrupted_ciphertext_yields_j_of_z_and_that_ciphertext(
        self, params: MlKemParams
    ) -> None:
        # Not merely "a different secret": the expected value is computed from
        # `hashlib`, so a rejection path deriving with `H` instead of `J`, or
        # over the wrong `z`, fails here rather than passing as "different".
        scheme = MlKem(params)
        c = _flip(_published(params, "c"), 0)
        z = _published(params, "z")
        got = scheme.decaps(_published(params, "dk"), c)
        self.assertEqual(to_bytes(got), _j(z, c))

    @parameterized.named_parameters(*_NAMED)
    def test_keygen_lays_the_decapsulation_key_out_as_published(
        self, params: MlKemParams
    ) -> None:
        """`dk = dk_PKE ‖ ek ‖ H(ek) ‖ z`, §7.1, against the published `dk`.

        The vectors' `d` cannot drive `keygen` — the expansion they predate is
        the module docstring's subject — so what is gated here is the frame the
        transform adds: which field holds what, that the hash is `H` and not `G`,
        and that `z` arrives from the seed rather than from anywhere else.
        """
        scheme = MlKem(params)
        vectors = cctv_vectors.intermediate(params.name)
        framed = encoding.encode_dk(
            _published(params, "dkPKE", 0),
            _published(params, "ek"),
            _published(params, "H(ek)"),
            _published(params, "z"),
        )
        self.assertEqual(to_bytes(framed), vectors.hex_at("dk"))

        z = _published(params, "z")
        ek, dk = scheme.keygen(np.concatenate([_published(params, "d"), z]))
        _, ek_field, hash_field, z_field = encoding.decode_dk(dk, params.k)
        self.assertEqual(to_bytes(ek_field), to_bytes(ek))
        self.assertEqual(to_bytes(hash_field), hashlib.sha3_256(to_bytes(ek)).digest())
        self.assertEqual(to_bytes(z_field), bytes(z))

    @parameterized.named_parameters(*_NAMED)
    def test_a_generated_key_round_trips(self, params: MlKemParams) -> None:
        # The published run enters at `ek`/`dk`; this is the only case that
        # drives the amended key generation into the transform.
        scheme = MlKem(params)
        rng = np.random.default_rng(len(params.name))
        seed = rng.integers(0, 256, scheme.seed_size, dtype=np.uint8)
        m = rng.integers(0, 256, scheme.randomness_size, dtype=np.uint8)
        ek, dk = scheme.keygen(seed)
        c, shared = scheme.encaps(ek, randomness=m)
        self.assertEqual(to_bytes(scheme.decaps(dk, c)), to_bytes(shared))


class RejectionTest(absltest.TestCase):
    """That the re-encryption runs, consumes the whole ciphertext, and decides
    per batch entry — none of which a published positive vector can see."""

    def test_re_encryption_runs_once_over_the_whole_batch(self) -> None:
        """The cost, asserted rather than inferred from the answer.

        Every entry is invalid here, which is where an implementation tempted to
        skip the work would skip it — and skipping is not an optimization,
        because there is no signal that a ciphertext is fine other than this
        computation.
        """
        keys, ciphertexts, _, _ = _batch()
        corrupted = _flip(ciphertexts, 0)
        with mock.patch.object(
            _k_pke, "encrypt", wraps=_k_pke.encrypt
        ) as re_encryption:
            _768.decaps(keys, corrupted)
        re_encryption.assert_called_once()
        self.assertEqual(
            np.shape(re_encryption.call_args.args[0])[0],
            _BATCH,
            "re-encryption ran on a subset of the batch",
        )

    def test_a_ciphertext_that_still_decrypts_to_the_message_is_rejected(self) -> None:
        """The check consumes the whole ciphertext, not just what decrypts.

        Compression throws bits away, so a small change to the tail of `c`
        recovers the *same* message — and an implementation that stopped at
        "decryption succeeded" would answer with the sender's own secret. Only
        re-encryption sees the difference.
        """
        keys, ciphertexts, secrets, seeds = _batch()
        tampered = _flip(ciphertexts, -1)
        dk_pke, _, _, _ = encoding.decode_dk(keys, ML_KEM_768.k)
        args = {"k": ML_KEM_768.k, "du": ML_KEM_768.du, "dv": ML_KEM_768.dv}
        np.testing.assert_array_equal(
            np.asarray(_k_pke.decrypt(dk_pke, tampered, **args)),
            np.asarray(_k_pke.decrypt(dk_pke, ciphertexts, **args)),
            "the flipped bit changed the message, so this case proves nothing",
        )
        got = np.asarray(_768.decaps(keys, tampered))
        for entry, (z, c) in enumerate(zip(seeds, tampered)):
            self.assertEqual(bytes(got[entry]), _j(z, c), f"entry {entry}")
        self.assertNotEqual(bytes(got[0]), bytes(secrets[0]))

    def test_a_mixed_batch_is_decided_per_entry(self) -> None:
        """Entries 1 and 2 corrupted, and only those two rejected.

        A check reduced over the batch rather than over each ciphertext passes
        every all-valid and every all-invalid case ever written.
        """
        keys, ciphertexts, secrets, seeds = _batch()
        corrupted = (1, 2)
        mixed = ciphertexts.copy()
        for entry in corrupted:
            mixed[entry] = _flip(mixed[entry], 0)
        got = np.asarray(_768.decaps(keys, mixed))
        for entry in range(_BATCH):
            want = (
                _j(seeds[entry], mixed[entry])
                if entry in corrupted
                else bytes(secrets[entry])
            )
            self.assertEqual(bytes(got[entry]), want, f"entry {entry}")

    def test_rejection_repeats(self) -> None:
        # Deterministic in `(z, c)`, so a caller cannot separate a rejection from
        # a real secret by asking twice.
        keys, ciphertexts, _, _ = _batch()
        corrupted = _flip(ciphertexts, 0)
        np.testing.assert_array_equal(
            np.asarray(_768.decaps(keys, corrupted)),
            np.asarray(_768.decaps(keys, corrupted)),
        )


class TracedShapeTest(absltest.TestCase):
    def test_the_decapsulation_path_has_no_branch(self) -> None:
        # The select is arithmetic: both secrets are computed, and which one
        # comes back is a mask. A `cond` here would be the timing signal the
        # transform exists to withhold.
        keys, ciphertexts, _, _ = _batch()
        text = str(frx.make_jaxpr(_768.decaps)(keys, ciphertexts))
        # Positively, so the negative below cannot pass on an empty trace.
        self.assertIn("select_n", text, "decaps traced without a select")
        self.assertIsNone(_BRANCH.search(text), "decaps traced to a branch")

    def test_the_traced_secret_matches_the_eager_one(self) -> None:
        keys, ciphertexts, _, _ = _batch()
        np.testing.assert_array_equal(
            np.asarray(frx.jit(_768.decaps)(keys, ciphertexts)),
            np.asarray(_768.decaps(keys, ciphertexts)),
        )


class KeyCheckTest(absltest.TestCase):
    """FIPS 203 §7.2 and §7.3 as standalone operations, and folded into `decaps`."""

    def test_the_published_keys_pass_both_checks(self) -> None:
        self.assertTrue(_768.check_encapsulation_key(_published(ML_KEM_768, "ek")))
        self.assertTrue(_768.check_decapsulation_key(_published(ML_KEM_768, "dk")))

    def test_an_unreduced_coefficient_fails_the_encapsulation_key_check(self) -> None:
        # The first 12-bit value becomes 0xfff = 4095, which is not below `q`.
        ek = _published(ML_KEM_768, "ek").copy()
        ek[0], ek[1] = 0xFF, 0xFF
        self.assertFalse(_768.check_encapsulation_key(ek))

    def test_halves_that_disagree_fail_the_decapsulation_key_check(self) -> None:
        dk = _published(ML_KEM_768, "dk").copy()
        dk[-2 * SEED_SIZE] ^= 1
        self.assertFalse(_768.check_decapsulation_key(dk))

    def test_a_key_check_is_per_entry_over_a_batch(self) -> None:
        good = _published(ML_KEM_768, "ek")
        bad = good.copy()
        bad[0], bad[1] = 0xFF, 0xFF
        got = _768.check_encapsulation_key(np.stack([good, bad, good]))
        self.assertEqual(np.asarray(got).tolist(), [True, False, True])

    def test_decaps_rejects_an_unreduced_key_rather_than_raising(self) -> None:
        """§7.2's check reaches `decaps` through the rejection path, not an
        exception — the same channel a bad ciphertext takes, for the same
        reason."""
        dk = _published(ML_KEM_768, "dk").copy()
        c = _published(ML_KEM_768, "c")
        # The embedded `ek` starts where `dk_PKE` ends.
        start = ML_KEM_768.decryption_key_size
        dk[start], dk[start + 1] = 0xFF, 0xFF
        got = _768.decaps(dk, c)
        self.assertEqual(to_bytes(got), _j(_published(ML_KEM_768, "z"), c))


class PrecomputedDecapsTest(absltest.TestCase):
    """`precompute_decaps` / `decaps_precomputed`: one key, many ciphertexts.

    A *narrower* operation than the seam's `decaps`, not a faster one — so what
    these check is first that it answers identically, and then that it is
    actually narrower rather than quietly accepting the batch of keys it cannot
    serve.
    """

    def test_it_answers_exactly_what_the_seam_answers(self) -> None:
        """Including the rejections, and per entry.

        A mixed batch rather than an all-valid one: the whole point of the
        precomputed path is that `key_ok` is now a single value AND-ed into
        every entry, and an implementation that reduced the ciphertext
        comparison over the batch alongside it would pass an all-valid and an
        all-invalid case alike.
        """
        dk, ciphertexts, secrets, z = _same_key_batch()
        corrupted = (1, 2)
        mixed = ciphertexts.copy()
        for entry in corrupted:
            mixed[entry] = _flip(mixed[entry], 0)

        got = np.asarray(_768.decaps_precomputed(_768.precompute_decaps(dk), mixed))
        np.testing.assert_array_equal(
            got,
            np.asarray(_768.decaps(np.broadcast_to(dk, (_BATCH, dk.shape[-1])), mixed)),
            "the precomputed path diverged from the seam",
        )
        for entry in range(_BATCH):
            want = _j(z, mixed[entry]) if entry in corrupted else bytes(secrets[entry])
            self.assertEqual(bytes(got[entry]), want, f"entry {entry}")

    def test_the_matrix_is_expanded_for_one_key_rather_than_for_every_entry(
        self,
    ) -> None:
        """The win, asserted as work that moved rather than as elapsed time.

        `Â` is a function of `ρ` alone, so the seam expands `[B, k, k, 256]` of
        it for a batch whose keys happen to be equal — XLA cannot know the rows
        are equal, because `ρ` is traced data and not shape. The hoisted path
        expands `[k, k, 256]` once and none per ciphertext, and those two shapes
        differing by exactly the batch axis is the whole claim.
        """
        dk, ciphertexts, _, _ = _same_key_batch()
        original = sampling.expand_matrix
        shapes: list[tuple[int, ...]] = []

        def _record(rho: object, k: int) -> object:
            expanded = original(rho, k)
            shapes.append(tuple(expanded.shape))
            return expanded

        with mock.patch.object(sampling, "expand_matrix", _record):
            _768.decaps(np.broadcast_to(dk, (_BATCH, dk.shape[-1])), ciphertexts)
            seam = shapes.pop()

            precomputed = _768.precompute_decaps(dk)
            hoisted = shapes.pop()

            _768.decaps_precomputed(precomputed, ciphertexts)
            self.assertEmpty(shapes, "decaps_precomputed expanded the matrix again")

        self.assertEqual(seam, (_BATCH, *hoisted))

    def test_a_batch_of_keys_is_refused(self) -> None:
        """The restriction is a shape error, not a docstring.

        Entry `i` of `decaps` belongs to `key[i]`; this pair has one key for the
        whole batch. Broadcasting a batched key here would answer every entry
        under `key[0]` — plausible bytes, wrong for every entry but the first,
        and nothing raised.
        """
        dk, _, _, _ = _same_key_batch()
        with self.assertRaisesRegex(ValueError, "takes one key"):
            _768.precompute_decaps(np.broadcast_to(dk, (_BATCH, dk.shape[-1])))

    def test_a_malformed_key_rejects_rather_than_raising(self) -> None:
        """§7.2 and §7.3 moved to `precompute_decaps`, and neither gained a
        failure channel on the way.

        Both sections separately, because they are checked together here and a
        `key_ok` that dropped one would still reject the other's cases.
        """
        dk, ciphertexts, _, z = _same_key_batch()
        start = ML_KEM_768.decryption_key_size
        unreduced = dk.copy()  # §7.2: the embedded `ek`'s first coefficient.
        unreduced[start], unreduced[start + 1] = 0xFF, 0xFF
        disagreeing = dk.copy()  # §7.3: `H(ek)` no longer matches `ek`.
        disagreeing[-2 * SEED_SIZE] ^= 1

        for name, malformed in (("§7.2", unreduced), ("§7.3", disagreeing)):
            got = _768.decaps_precomputed(
                _768.precompute_decaps(malformed), ciphertexts
            )
            for entry in range(_BATCH):
                self.assertEqual(
                    bytes(np.asarray(got)[entry]),
                    _j(z, ciphertexts[entry]),
                    f"{name}, entry {entry}",
                )

    def test_the_precomputed_path_has_no_branch(self) -> None:
        # Same requirement as `decaps`: `key_ok` is a value AND-ed into a mask,
        # so a `cond` on it would be the timing signal the transform withholds.
        dk, ciphertexts, _, _ = _same_key_batch()
        text = str(
            frx.make_jaxpr(_768.decaps_precomputed)(
                _768.precompute_decaps(dk), ciphertexts
            )
        )
        self.assertIn("select_n", text, "decaps_precomputed traced without a select")
        self.assertIsNone(_BRANCH.search(text), "decaps_precomputed traced to a branch")

    def test_the_traced_secret_matches_the_eager_one(self) -> None:
        dk, ciphertexts, _, _ = _same_key_batch()
        precomputed = _768.precompute_decaps(dk)
        np.testing.assert_array_equal(
            np.asarray(frx.jit(_768.decaps_precomputed)(precomputed, ciphertexts)),
            np.asarray(_768.decaps_precomputed(precomputed, ciphertexts)),
        )

    def test_the_parsed_key_holds_no_secret_vector(self) -> None:
        """`ŝ` stays encoded, which is what keeps this value's handling
        requirements the same as the encapsulation key's.

        `dk_pke` is carried as the bytes `decode_dk` produced — the same bytes
        already inside the key the caller handed over — so nothing here is a
        parsed secret. A field holding `ŝ` as field elements would be a new kind
        of object for `docs/reference/security.md` to have an opinion about.
        """
        dk, _, _, _ = _same_key_batch()
        precomputed = _768.precompute_decaps(dk)
        np.testing.assert_array_equal(
            np.asarray(precomputed.dk_pke),
            dk[: ML_KEM_768.decryption_key_size],
        )


class SeamTest(parameterized.TestCase):
    def test_the_protocol_accepts_the_scheme(self) -> None:
        self.assertIsInstance(_768, Kem)

    def test_the_protocol_does_not_gain_the_precomputed_pair(self) -> None:
        """The pair lives on `MlKem`, below the seam, like `encaps_internal`.

        A protocol method returning a scheme-shaped opaque value would make
        generic code hold ML-KEM-shaped state, which is the coupling the seam
        exists to prevent — and it would not generalize, since a hybrid KEM has
        no `Â` (`enc_frx/kem.py`).
        """
        for name in ("precompute_decaps", "decaps_precomputed"):
            self.assertTrue(hasattr(MlKem, name))
            self.assertFalse(hasattr(Kem, name), f"Kem gained {name}")

    @parameterized.named_parameters(*_NAMED)
    def test_the_sizes_are_the_standards(self, params: MlKemParams) -> None:
        """FIPS 203 Table 3, as published rather than as recomputed."""
        scheme = MlKem(params)
        encapsulation, decapsulation, ciphertext = _TABLE_3[params]
        self.assertEqual(scheme.encapsulation_key_size, encapsulation)
        self.assertEqual(scheme.decapsulation_key_size, decapsulation)
        self.assertEqual(scheme.ciphertext_size, ciphertext)
        # Table 3's fourth column, and §7.1's seed, are the same at every set.
        self.assertEqual(scheme.shared_secret_size, 32)
        self.assertEqual(scheme.seed_size, 64)

    def test_equality_is_by_value(self) -> None:
        # Identity equality would not error; it would silently re-trace the
        # enclosing jit zone for every freshly built instance.
        twin = MlKem(ML_KEM_768)
        self.assertEqual(_768, twin)
        self.assertEqual(hash(_768), hash(twin))
        self.assertNotEqual(_768, MlKem(ML_KEM_1024))

    def test_two_instances_of_one_set_share_a_trace(self) -> None:
        """What value-based equality buys, asserted as the cost it avoids.

        A scheme reaches a traced function as a static argument, where it is
        compared by `__eq__` against the cache. The equality test above says the
        comparison answers `True`; this says the cache believes it, and identity
        equality — which would not error anywhere else — traces twice here.

        The body is the cheapest method that still reads the set (`params.k`
        sizes the slice), because what is being counted is decided before
        tracing starts: a real encapsulation would compile K-PKE to assert a
        dictionary lookup.
        """
        traces = 0

        @functools.partial(frx.jit, static_argnums=0)
        def check(scheme: MlKem, ek: np.ndarray) -> frx.Array:
            nonlocal traces
            traces += 1
            return scheme.check_encapsulation_key(ek)

        ek = _published(ML_KEM_768, "ek")
        check(MlKem(ML_KEM_768), ek)
        check(MlKem(ML_KEM_768), ek)

        self.assertEqual(traces, 1)

    def test_rejects_wrong_lengths(self) -> None:
        zeros = np.zeros(_768.decapsulation_key_size, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "key generation seed"):
            _768.keygen(zeros[: _768.seed_size - 1])
        with self.assertRaisesRegex(ValueError, "encapsulation key"):
            _768.encaps(
                zeros[: _768.encapsulation_key_size - 1],
                randomness=zeros[:SEED_SIZE],
            )
        with self.assertRaisesRegex(ValueError, "encapsulation randomness"):
            _768.encaps(
                zeros[: _768.encapsulation_key_size], randomness=zeros[: SEED_SIZE - 1]
            )
        with self.assertRaisesRegex(ValueError, "decapsulation key"):
            _768.decaps(zeros[:-1], zeros[: _768.ciphertext_size])
        with self.assertRaisesRegex(ValueError, "ciphertext"):
            _768.decaps(zeros, zeros[: _768.ciphertext_size - 1])


def _batch() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(decaps_keys, ciphertexts, shared_secrets, rejection_seeds)` at 768.

    Generated rather than published: one published run is one entry, and every
    case below is about the batch axis or about a value no file carries.
    """
    rng = np.random.default_rng(0)
    seeds = rng.integers(0, 256, (_BATCH, _768.seed_size), dtype=np.uint8)
    m = rng.integers(0, 256, (_BATCH, _768.randomness_size), dtype=np.uint8)
    encaps_keys, decaps_keys = frx.vmap(_768.keygen)(seeds)
    ciphertexts, secrets = _768.encaps(encaps_keys, randomness=m)
    return (
        np.asarray(decaps_keys),
        np.asarray(ciphertexts),
        np.asarray(secrets),
        seeds[:, SEED_SIZE:],
    )


def _same_key_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(dk, ciphertexts, secrets, z)` at 768 — one key, `_BATCH` ciphertexts.

    The shape `precompute_decaps` serves and the published corpus does not
    contain: ACVP's decapsulation group is ten independent keys.
    """
    rng = np.random.default_rng(1)
    seed = rng.integers(0, 256, _768.seed_size, dtype=np.uint8)
    encaps_key, decaps_key = _768.keygen(seed)
    m = rng.integers(0, 256, (_BATCH, _768.randomness_size), dtype=np.uint8)
    ciphertexts, secrets = _768.encaps(
        np.broadcast_to(np.asarray(encaps_key), (_BATCH, _768.encapsulation_key_size)),
        randomness=m,
    )
    return (
        np.asarray(decaps_key),
        np.asarray(ciphertexts),
        np.asarray(secrets),
        seed[SEED_SIZE:],
    )


if __name__ == "__main__":
    absltest.main()
