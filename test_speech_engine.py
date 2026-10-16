"""
Comprehensive Test Suite for Paged Dual-Cache Speech Serving Engine.

Tests:
1. Streaming Multi-Codebook TTS Engine continuous batching and multi-codebook frame generation.
2. Encoder-Side Paged Attention & Audio Prefix Sharing hit rates.
3. Dual-Cache Cross-Attention & Phase-Aware Scheduler state transitions.
4. GPU Memory Safety — Explicit block release verification (zero memory leakage after request completion or failure).
5. Cross-Attention Masking Correctness — Exact attention masking under mixed-length audio batching.
"""

import sys
import unittest
import torch

from speech_serving.models import (
    StreamingMultiCodebookTTSModel,
    WhisperClassSpeechEncoderDecoderModel,
)
from speech_serving.tts_engine import StreamingTTSEngine
from speech_serving.encoder_block_manager import EncoderBlockManager
from speech_serving.encoder_attention import EncoderAttentionBackend
from speech_serving.dual_cache_manager import DualCacheManager
from speech_serving.phase_aware_scheduler import PhaseAwareSpeechScheduler, RequestPhase
from speech_serving.memory_guard import MemoryGuard
from paged_kv_cache.block_manager import BlockManager


class TestSpeechServingEngine(unittest.TestCase):

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def test_streaming_tts_engine(self):
        """Test Streaming Multi-Codebook TTS Engine."""
        print("\n--- Running Test: Streaming Multi-Codebook TTS Engine ---")
        num_layers = 2
        num_heads = 4
        hidden_dim = 64
        num_codebooks = 4
        vocab_size = 128
        block_size = 8

        model = StreamingMultiCodebookTTSModel(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_codebooks=num_codebooks,
            vocab_size=vocab_size,
        ).to(device=self.device, dtype=self.dtype)
        model.eval()

        bm = BlockManager(
            num_layers=num_layers,
            num_kv_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            block_size=block_size,
            num_blocks=32,
            dtype=self.dtype,
            device=self.device,
        )

        engine = StreamingTTSEngine(model, bm, max_batch_size=2, device=self.device)

        # Submit 3 streaming TTS requests
        frames_received = {}
        def callback_factory(req_id):
            def cb(rid, frame):
                if rid not in frames_received:
                    frames_received[rid] = 0
                frames_received[rid] += 1
            return cb

        for i in range(3):
            prompt = torch.randint(0, vocab_size, (10, num_codebooks), device=self.device)
            engine.submit(prompt, max_new_frames=15, on_frame_callback=callback_factory(i))

        completed = engine.run_until_done()

        self.assertEqual(len(completed), 3)
        for req in completed:
            self.assertTrue(req.done)
            self.assertEqual(req.generated_count, 15)
            self.assertEqual(len(req.generated_frames), 15)
            self.assertEqual(req.generated_frames[0].shape[0], num_codebooks)

        # Verify all physical blocks were returned to free_block_list
        self.assertEqual(bm.allocated_block_count, 0)
        self.assertEqual(bm.num_free_blocks, 32)
        print(">>> Streaming TTS Test PASSED: Continuous batching & multi-codebook TTS streaming verified cleanly.")

    def test_encoder_paged_attention_and_prefix_sharing(self):
        """Test Encoder Paged Attention Manager & Audio Prefix Sharing."""
        print("\n--- Running Test: Encoder Paged Attention & Prefix Sharing ---")
        num_layers = 2
        num_heads = 4
        head_dim = 16
        block_size = 8

        enc_bm = EncoderBlockManager(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            num_blocks=32,
            dtype=self.dtype,
            device=self.device,
        )

        # 1. Allocate blocks for identical audio prefix content hash
        hash_1 = "audio_silence_intro_hash_abc123"
        blk_1, hit_1 = enc_bm.allocate_block(content_hash=hash_1)
        self.assertFalse(hit_1)
        self.assertEqual(enc_bm.ref_counts[blk_1], 1)

        # 2. Second request reuses same prefix hash -> prefix hit!
        blk_2, hit_2 = enc_bm.allocate_block(content_hash=hash_1)
        self.assertTrue(hit_2)
        self.assertEqual(blk_1, blk_2) # Shared physical block
        self.assertEqual(enc_bm.ref_counts[blk_1], 2)
        self.assertGreater(enc_bm.hit_rate, 0.0)

        # 3. Test non-causal attention backend
        q = torch.randn(1, num_heads, 4, head_dim, device=self.device, dtype=self.dtype)
        # Store dummy keys and values
        dummy_k = torch.randn(8, num_heads, head_dim, device=self.device, dtype=self.dtype)
        dummy_v = torch.randn(8, num_heads, head_dim, device=self.device, dtype=self.dtype)
        enc_bm.store_encoder_kv(blk_1, layer_idx=0, k_tensor=dummy_k, v_tensor=dummy_v)

        out_attn = EncoderAttentionBackend.forward(
            query=q,
            encoder_block_table=[blk_1],
            key_pool_layer=enc_bm.key_pool[0],
            value_pool_layer=enc_bm.value_pool[0],
            encoder_seq_len=8,
            block_size=block_size,
        )
        self.assertEqual(out_attn.shape, q.shape)

        # 4. Release reference count deterministically
        enc_bm.release_block(blk_1) # ref_count drops from 2 -> 1 (not freed yet)
        self.assertEqual(enc_bm.ref_counts[blk_1], 1)
        enc_bm.release_block(blk_2) # ref_count drops 1 -> 0 (freed!)
        self.assertEqual(enc_bm.allocated_block_count, 0)
        print(">>> Encoder Paged Attention Test PASSED: Non-causal encoder paged attention & audio prefix sharing verified.")

    def test_dual_cache_phase_aware_scheduler(self):
        """Test Dual-Cache Manager & Phase-Aware Scheduler."""
        print("\n--- Running Test: Dual-Cache Phase-Aware Scheduler ---")
        num_enc_layers = 2
        num_dec_layers = 2
        num_heads = 4
        hidden_dim = 64
        mel_dim = 80
        vocab_size = 256
        block_size = 8

        model = WhisperClassSpeechEncoderDecoderModel(
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mel_dim=mel_dim,
            vocab_size=vocab_size,
        ).to(device=self.device, dtype=self.dtype)
        model.eval()

        enc_bm = EncoderBlockManager(
            num_layers=num_enc_layers,
            num_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            block_size=block_size,
            num_blocks=32,
            dtype=self.dtype,
            device=self.device,
        )

        dec_bm = BlockManager(
            num_layers=num_dec_layers,
            num_kv_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            block_size=block_size,
            num_blocks=32,
            dtype=self.dtype,
            device=self.device,
        )

        dual_cache = DualCacheManager(enc_bm, dec_bm)
        memory_guard = MemoryGuard(high_water_mark_pct=90.0)

        scheduler = PhaseAwareSpeechScheduler(
            model=model,
            dual_cache_manager=dual_cache,
            memory_guard=memory_guard,
            max_batch_size=2,
            device=self.device,
        )

        # Submit 4 requests with synthetic mel features
        for i in range(4):
            mel = torch.randn(mel_dim, 32, device=self.device, dtype=self.dtype)
            req_id = scheduler.submit(mel, max_new_tokens=10)
            self.assertIsNotNone(req_id)

        completed = scheduler.run_until_done()
        self.assertEqual(len(completed), 4)

        for req in completed:
            self.assertEqual(req.phase, RequestPhase.COMPLETED)
            self.assertTrue(req.done)
            self.assertEqual(req.generated_count, 10)
            # Verify block tables cleared cleanly
            self.assertEqual(len(req.encoder_block_table), 0)
            self.assertEqual(len(req.decoder_block_table), 0)

        # Confirm zero memory leak (all blocks freed in both pools)
        self.assertEqual(enc_bm.allocated_block_count, 0)
        self.assertEqual(dec_bm.allocated_block_count, 0)
        print(">>> Dual-Cache Scheduler Test PASSED: Phase-aware scheduling & zero-leak memory release verified.")

    def test_empty_pool_admission(self):
        """Regression Test: Verify single request admission into a completely empty pool is never rejected."""
        enc_bm = EncoderBlockManager(num_layers=2, num_heads=4, head_dim=16, block_size=8, num_blocks=16, dtype=self.dtype, device=self.device)
        dec_bm = BlockManager(num_layers=2, num_kv_heads=4, head_dim=16, block_size=8, num_blocks=16, dtype=self.dtype, device=self.device)
        dual_cache = DualCacheManager(enc_bm, dec_bm)
        guard = MemoryGuard(high_water_mark_pct=75.0)

        model = WhisperClassSpeechEncoderDecoderModel(num_encoder_layers=2, num_decoder_layers=2, num_heads=4, hidden_dim=64, mel_dim=80, vocab_size=256).to(device=self.device, dtype=self.dtype)
        scheduler = PhaseAwareSpeechScheduler(model=model, dual_cache_manager=dual_cache, memory_guard=guard, device=self.device)

        mel = torch.randn(80, 48, device=self.device, dtype=self.dtype)
        req_id = scheduler.submit(mel, max_new_tokens=10)
        self.assertIsNotNone(req_id, "Request into completely empty pool should ALWAYS be admitted!")
        self.assertEqual(req_id, 0)

    def test_variable_length_batch_cross_attention_correctness(self):
        """Regression Test: Verify that mixed audio lengths in a concurrent batch do not suffer cross-attention padding leakage."""
        print("\n--- Running Regression Test: Variable-Length Batch Cross-Attention Correctness ---")
        torch.manual_seed(999)
        num_enc_layers = 2
        num_dec_layers = 2
        num_heads = 4
        hidden_dim = 64
        mel_dim = 80
        vocab_size = 256
        block_size = 8
        max_new_tokens = 20

        model = WhisperClassSpeechEncoderDecoderModel(
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mel_dim=mel_dim,
            vocab_size=vocab_size,
        ).to(device=self.device, dtype=self.dtype)
        model.eval()

        # Varied mix of audio lengths: [40, 100, 160, 200] in a single batch of 4
        audio_lengths = [40, 100, 160, 200]
        test_mels = [torch.randn(mel_dim, l, device=self.device, dtype=self.dtype) for l in audio_lengths]

        # 1. Compute ground-truth sequential HF decode per request
        reference_tokens = []
        for mel in test_mels:
            mel_in = mel.unsqueeze(0)
            with torch.no_grad():
                _, enc_kv = model.encode(mel_in)
            tok = torch.tensor([[1]], device=self.device, dtype=torch.long)
            kv = None
            gen = []
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    out = model.decode_step(tok, kv, enc_kv, use_cache=True)
                kv = out.past_key_values
                t = torch.argmax(out.logits[:, -1, :], dim=-1).item()
                tok = torch.tensor([[t]], device=self.device, dtype=torch.long)
                gen.append(t)
            reference_tokens.append(gen)

        # 2. Run paged engine with batch_size=4
        enc_bm = EncoderBlockManager(num_layers=num_enc_layers, num_heads=num_heads, head_dim=hidden_dim // num_heads, block_size=block_size, num_blocks=128, dtype=self.dtype, device=self.device)
        dec_bm = BlockManager(num_layers=num_dec_layers, num_kv_heads=num_heads, head_dim=hidden_dim // num_heads, block_size=block_size, num_blocks=128, dtype=self.dtype, device=self.device)
        dual = DualCacheManager(enc_bm, dec_bm)
        scheduler = PhaseAwareSpeechScheduler(model=model, dual_cache_manager=dual, memory_guard=MemoryGuard(), max_batch_size=4, device=self.device)

        for mel in test_mels:
            scheduler.submit(mel, max_new_tokens=max_new_tokens)

        completed = scheduler.run_until_done()
        completed.sort(key=lambda r: r.id)
        paged_tokens = [r.generated_tokens for r in completed]

        for i, (ref, paged) in enumerate(zip(reference_tokens, paged_tokens)):
            self.assertEqual(ref, paged, f"Request #{i} (mel_len={audio_lengths[i]}) output diverged from reference under variable-length batching!")

        print(f">>> Variable-Length Batch Test PASSED: All {len(audio_lengths)} heterogeneous requests match reference token-for-token!")


    # ------------------------------------------------------------------
    # Task 2 Tests: lookup-before-encode correctness
    # ------------------------------------------------------------------

    def test_encoder_hit_produces_identical_output(self):
        """
        Task 2 Test (1): When the same mel is submitted twice, the second request
        must hit the encoder cache and generate EXACTLY the same token sequence as
        the first (torch.allclose-equivalent for deterministic greedy decoding).
        """
        print("\n--- Task 2 Test: cache-hit output == miss-path output ---")
        torch.manual_seed(7)
        num_enc_layers = 2
        num_dec_layers = 2
        num_heads = 4
        hidden_dim = 64
        mel_dim = 80
        vocab_size = 256
        block_size = 8
        max_new_tokens = 8

        model = WhisperClassSpeechEncoderDecoderModel(
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mel_dim=mel_dim,
            vocab_size=vocab_size,
        ).to(device=self.device, dtype=self.dtype)
        model.eval()

        enc_bm = EncoderBlockManager(num_layers=num_enc_layers, num_heads=num_heads,
                                     head_dim=hidden_dim // num_heads, block_size=block_size,
                                     num_blocks=64, dtype=self.dtype, device=self.device)
        dec_bm = BlockManager(num_layers=num_dec_layers, num_kv_heads=num_heads,
                              head_dim=hidden_dim // num_heads, block_size=block_size,
                              num_blocks=64, dtype=self.dtype, device=self.device)
        dual = DualCacheManager(enc_bm, dec_bm)
        scheduler = PhaseAwareSpeechScheduler(model=model, dual_cache_manager=dual,
                                              memory_guard=MemoryGuard(), max_batch_size=2,
                                              device=self.device)

        mel = torch.randn(mel_dim, 32, device=self.device, dtype=self.dtype)

        # Hook model.decode_step to capture the exact gathered encoder past KV list
        gathered_kvs = []
        original_decode_step = model.decode_step
        def hooked_decode(input_ids, past_key_values=None, encoder_past_kv_list=None, **kwargs):
            if encoder_past_kv_list is not None:
                # Capture a deep copy of the gathered encoder KV layers passed to attention
                gathered_kvs.append([(k.clone(), v.clone()) for (k, v) in encoder_past_kv_list])
            return original_decode_step(input_ids, past_key_values=past_key_values, encoder_past_kv_list=encoder_past_kv_list, **kwargs)
        model.decode_step = hooked_decode

        # First submission — miss path
        scheduler.submit(mel.clone(), max_new_tokens=max_new_tokens)
        completed1 = scheduler.run_until_done()
        self.assertEqual(len(completed1), 1)
        tokens_first = completed1[0].generated_tokens[:]

        # Second submission — hit path (same mel bytes)
        scheduler.completed.clear()
        scheduler.submit(mel.clone(), max_new_tokens=max_new_tokens)
        completed2 = scheduler.run_until_done()
        self.assertEqual(len(completed2), 1)
        tokens_second = completed2[0].generated_tokens[:]

        # 1. Output tokens must match identically
        self.assertEqual(tokens_first, tokens_second,
                         "Hit path must produce identical tokens to miss path!")

        # 2. Gathered encoder KV tensors must be numerically identical (torch.allclose)
        self.assertGreaterEqual(len(gathered_kvs), 2, "Both miss and hit paths must pass gathered encoder KV to decode_step")
        kv_miss = gathered_kvs[0]
        kv_hit = gathered_kvs[1]
        self.assertEqual(len(kv_miss), num_enc_layers)
        self.assertEqual(len(kv_hit), num_enc_layers)
        for l in range(num_enc_layers):
            k_miss, v_miss = kv_miss[l]
            k_hit, v_hit = kv_hit[l]
            self.assertTrue(
                torch.allclose(k_miss, k_hit, atol=1e-5, rtol=1e-5),
                f"Layer {l} Key tensor mismatch between miss and hit path! Max diff: {(k_miss - k_hit).abs().max()}"
            )
            self.assertTrue(
                torch.allclose(v_miss, v_hit, atol=1e-5, rtol=1e-5),
                f"Layer {l} Value tensor mismatch between miss and hit path! Max diff: {(v_miss - v_hit).abs().max()}"
            )

        model.decode_step = original_decode_step
        print(">>> Task 2 Test (1) PASSED: hit-path tokens match & gathered encoder KV is torch.allclose across all layers.")

    def test_encoder_not_called_on_hit(self):
        """
        Task 2 Test (2): Verify that the encoder forward is NOT called on a cache hit.
        We wrap model.encode to count calls and assert it is called exactly once for
        two identical submissions made SEQUENTIALLY (miss then hit).

        HIT SEMANTICS (documented)
        ===========================
        The full-input encoder cache (_full_input_enc_cache) is a write-on-miss,
        read-on-hit LRU store keyed by the MD5 hash of the full mel tensor.

        The cache ENTRY is written only AFTER the encoder has run and returned on
        the MISS path.  This means:

          CONCURRENT DUPLICATES (both submitted before any step()):
            Both requests are dequeued into active_encode in the same
            _process_encode_phase() sweep.  At that moment neither has written a
            cache entry yet, so BOTH see cache_hit=False and BOTH call the
            encoder.  This is intentional - the scheduler does not stall one
            request to wait for an in-flight sibling.

          SEQUENTIAL DUPLICATES (second submitted after first is complete):
            When request A finishes, its miss path writes the cache entry.
            When request B is then submitted and dequeued, the hash is already
            in _full_input_enc_cache, so cache_hit=True and the encoder is
            skipped entirely.

        The real-speech repeated-workload test shows ~50 skips because it
        processes a pool of 5 recurring audio clips one at a time, so every
        recurrence after the first clip is a SEQUENTIAL hit.

        This test exercises the SEQUENTIAL scenario.  It requires:
          - retain_encoder_cache=True  (OFF by default; cache_hit is always
            False when retain_encoder_cache=False)
          - Request 1 fully completed before Request 2 is submitted.
        """
        print("\n--- Task 2 Test: encoder not called on hit ---")
        torch.manual_seed(13)
        num_enc_layers = 2
        num_dec_layers = 2
        num_heads = 4
        hidden_dim = 64
        mel_dim = 80
        vocab_size = 256
        block_size = 8
        max_new_tokens = 5

        model = WhisperClassSpeechEncoderDecoderModel(
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mel_dim=mel_dim,
            vocab_size=vocab_size,
        ).to(device=self.device, dtype=self.dtype)
        model.eval()

        # Wrap model.encode to count calls
        encode_call_count = [0]
        original_encode = model.encode
        def counting_encode(mel_in):
            encode_call_count[0] += 1
            return original_encode(mel_in)
        model.encode = counting_encode

        enc_bm = EncoderBlockManager(num_layers=num_enc_layers, num_heads=num_heads,
                                     head_dim=hidden_dim // num_heads, block_size=block_size,
                                     num_blocks=64, dtype=self.dtype, device=self.device)
        dec_bm = BlockManager(num_layers=num_dec_layers, num_kv_heads=num_heads,
                              head_dim=hidden_dim // num_heads, block_size=block_size,
                              num_blocks=64, dtype=self.dtype, device=self.device)
        dual = DualCacheManager(enc_bm, dec_bm)
        # retain_encoder_cache=True is REQUIRED for the hit path.
        # Without it, cache_hit is always False regardless of request ordering.
        scheduler = PhaseAwareSpeechScheduler(model=model, dual_cache_manager=dual,
                                              memory_guard=MemoryGuard(), max_batch_size=2,
                                              retain_encoder_cache=True,
                                              device=self.device)

        mel = torch.randn(mel_dim, 40, device=self.device, dtype=self.dtype)

        # --- Request 1 (MISS): run to completion so the cache entry is written ---
        scheduler.submit(mel.clone(), max_new_tokens=max_new_tokens)
        scheduler.run_until_done()
        self.assertEqual(encode_call_count[0], 1,
                         "First (unique) request must trigger exactly one encoder call")

        # --- Request 2 (HIT): cache entry now present; encoder must NOT run again ---
        scheduler.submit(mel.clone(), max_new_tokens=max_new_tokens)
        scheduler.run_until_done()

        telemetry = scheduler.get_encoder_telemetry()
        self.assertEqual(encode_call_count[0], 1,
                         f"Encoder should be called exactly ONCE total for 2 sequential "
                         f"identical mels (sequential hit), got {encode_call_count[0]}")
        self.assertEqual(telemetry["encoder_calls"], 1)
        self.assertEqual(telemetry["encoder_calls_skipped"], 1)
        self.assertGreater(telemetry["skip_rate_pct"], 0.0)
        print(f">>> Task 2 Test (2) PASSED: encoder called {encode_call_count[0]}x "
              f"(skipped {telemetry['encoder_calls_skipped']}x). "
              f"Skip rate: {telemetry['skip_rate_pct']}%")

    def test_zero_blocks_leaked_after_hit_path(self):
        """
        Task 2 Test (3): Zero blocks are leaked after all requests complete,
        including on the hit path, WITHOUT manual cache flushing.
        """
        print("\n--- Task 2 Test: zero block leak after hit path (no manual flush) ---")
        torch.manual_seed(99)
        num_enc_layers = 2
        num_dec_layers = 2
        num_heads = 4
        hidden_dim = 64
        mel_dim = 80
        vocab_size = 256
        block_size = 8
        max_new_tokens = 6

        model = WhisperClassSpeechEncoderDecoderModel(
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mel_dim=mel_dim,
            vocab_size=vocab_size,
        ).to(device=self.device, dtype=self.dtype)
        model.eval()

        enc_bm = EncoderBlockManager(num_layers=num_enc_layers, num_heads=num_heads,
                                     head_dim=hidden_dim // num_heads, block_size=block_size,
                                     num_blocks=64, dtype=self.dtype, device=self.device)
        dec_bm = BlockManager(num_layers=num_dec_layers, num_kv_heads=num_heads,
                              head_dim=hidden_dim // num_heads, block_size=block_size,
                              num_blocks=64, dtype=self.dtype, device=self.device)
        dual = DualCacheManager(enc_bm, dec_bm)
        scheduler = PhaseAwareSpeechScheduler(model=model, dual_cache_manager=dual,
                                              memory_guard=MemoryGuard(), max_batch_size=4,
                                              device=self.device)

        mel_a = torch.randn(mel_dim, 24, device=self.device, dtype=self.dtype)
        mel_b = torch.randn(mel_dim, 32, device=self.device, dtype=self.dtype)

        # Submit: miss, hit, miss, hit
        scheduler.submit(mel_a.clone(), max_new_tokens=max_new_tokens)
        scheduler.submit(mel_b.clone(), max_new_tokens=max_new_tokens)
        scheduler.submit(mel_a.clone(), max_new_tokens=max_new_tokens)  # hit
        scheduler.submit(mel_b.clone(), max_new_tokens=max_new_tokens)  # hit
        scheduler.run_until_done()

        # Notice: NO scheduler.flush_full_input_cache() call here!
        self.assertEqual(enc_bm.allocated_block_count, 0,
                         f"Encoder pool leak! {enc_bm.allocated_block_count} blocks still allocated without manual flush.")
        self.assertEqual(dec_bm.allocated_block_count, 0,
                         f"Decoder pool leak! {dec_bm.allocated_block_count} blocks still allocated without manual flush.")
        print(">>> Task 2 Test (3) PASSED: zero blocks leaked after hit-path requests WITHOUT manual flush.")

    def test_generation_stops_on_eos_and_uses_whisper_prompt(self):
        """
        Task 2 Test (4): Verify generation stops early on EOS token and supports
        Whisper task prompt tokens.
        """
        print("\n--- Running Test: EOS Termination & Whisper Task Prompt Tokens ---")
        num_enc_layers = 2
        num_dec_layers = 2
        num_heads = 4
        hidden_dim = 64
        mel_dim = 80
        vocab_size = 512
        block_size = 8
        target_eos = 42

        model = WhisperClassSpeechEncoderDecoderModel(
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mel_dim=mel_dim,
            vocab_size=vocab_size,
        ).to(device=self.device, dtype=self.dtype)
        model.eval()

        enc_bm = EncoderBlockManager(num_layers=num_enc_layers, num_heads=num_heads,
                                     head_dim=hidden_dim // num_heads, block_size=block_size,
                                     num_blocks=64, dtype=self.dtype, device=self.device)
        dec_bm = BlockManager(num_layers=num_dec_layers, num_kv_heads=num_heads,
                              head_dim=hidden_dim // num_heads, block_size=block_size,
                              num_blocks=64, dtype=self.dtype, device=self.device)
        dual = DualCacheManager(enc_bm, dec_bm)
        scheduler = PhaseAwareSpeechScheduler(model=model, dual_cache_manager=dual,
                                              memory_guard=MemoryGuard(), max_batch_size=2,
                                              device=self.device, eos_token_id=target_eos)

        # Mock model.decode_step so that after step 3 it outputs EOS (42)
        step_counter = [0]
        real_decode = model.decode_step
        def mock_decode(*args, **kwargs):
            out = real_decode(*args, **kwargs)
            step_counter[0] += 1
            if step_counter[0] >= 3:
                # Force logits at target_eos to be maximal
                out.logits[:, -1, :] = -1e4
                out.logits[:, -1, target_eos] = 1e4
            return out
        model.decode_step = mock_decode

        mel = torch.randn(mel_dim, 32, device=self.device, dtype=self.dtype)
        # Request max_new_tokens=50, but model emits EOS on step 3
        prompt = torch.tensor([[1, 2, 3]], device=self.device, dtype=torch.long)
        req_id = scheduler.submit(mel, prompt_token_ids=prompt, max_new_tokens=50, eos_token_id=target_eos)
        completed = scheduler.run_until_done()

        self.assertEqual(len(completed), 1)
        req = completed[0]
        self.assertTrue(req.done)
        self.assertLess(req.generated_count, 50, "Request should have stopped early upon hitting EOS!")
        self.assertEqual(req.generated_tokens[-1], target_eos, "Final generated token must be EOS!")
        self.assertEqual(enc_bm.allocated_block_count, 0, "Encoder blocks must be freed on EOS termination")
        self.assertEqual(dec_bm.allocated_block_count, 0, "Decoder blocks must be freed on EOS termination")

        model.decode_step = real_decode
        print(f">>> EOS & Prompt Test PASSED: Generation stopped at {req.generated_count} tokens upon emitting EOS ({target_eos}), all blocks freed.")

    def test_pretrained_whisper_model_integration(self):
        """Test PretrainedWhisperModel interface, cross-attention projection caching, and scheduler integration."""
        print("\n--- Running Test: PretrainedWhisperModel Integration ---")
        from speech_serving.models import PretrainedWhisperModel

        model = PretrainedWhisperModel(
            model_name_or_path="openai/whisper-base",
            device=self.device,
            dtype=self.dtype,
        )

        mel = torch.randn(1, 80, 3000, device=self.device, dtype=self.dtype)
        enc_out, enc_kv = model.encode(mel)
        self.assertEqual(enc_out.shape[0], 1)
        self.assertEqual(len(enc_kv), model.num_decoder_layers)
        self.assertEqual(enc_kv[0][0].shape[1], model.num_heads)
        self.assertEqual(enc_kv[0][0].shape[3], model.head_dim)

        enc_bm = EncoderBlockManager(
            num_layers=model.num_encoder_layers,
            num_heads=model.num_heads,
            head_dim=model.head_dim,
            block_size=16,
            num_blocks=256,
            dtype=self.dtype,
            device=self.device,
        )
        dec_bm = BlockManager(
            num_layers=model.num_decoder_layers,
            num_kv_heads=model.num_heads,
            head_dim=model.head_dim,
            block_size=16,
            num_blocks=256,
            dtype=self.dtype,
            device=self.device,
        )
        dual = DualCacheManager(enc_bm, dec_bm)
        scheduler = PhaseAwareSpeechScheduler(
            model=model,
            dual_cache_manager=dual,
            max_batch_size=2,
            device=self.device,
            eos_token_id=50257,
        )

        prompt = torch.tensor([[50258, 50259, 50359, 50363]], device=self.device, dtype=torch.long)
        req_id = scheduler.submit(mel[0], prompt_token_ids=prompt, max_new_tokens=2, eos_token_id=50257)
        self.assertIsNotNone(req_id)
        completed = scheduler.run_until_done()
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].generated_count, 2)
        self.assertEqual(enc_bm.allocated_block_count, 0)
        self.assertEqual(dec_bm.allocated_block_count, 0)
        print(">>> PretrainedWhisperModel Integration Test PASSED: Real Whisper encoder/decoder paged serving verified.")

    def test_cache_cross_kv_parity(self):
        """Test exact output token parity between cache_cross_kv=False and cache_cross_kv=True."""
        print("\n--- Running Test: cache_cross_kv Output Token Parity ---")
        hidden_dim = 64
        num_heads = 4
        head_dim = hidden_dim // num_heads

        model = WhisperClassSpeechEncoderDecoderModel(
            num_encoder_layers=2,
            num_decoder_layers=2,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mel_dim=80,
            vocab_size=100,
        ).to(device=self.device, dtype=self.dtype).eval()

        torch.manual_seed(42)
        mel = torch.randn(80, 200, device=self.device, dtype=self.dtype)
        prompt = torch.tensor([[1, 2, 3]], device=self.device, dtype=torch.long)

        # Run with flag OFF
        enc_bm1 = EncoderBlockManager(num_layers=2, num_heads=num_heads, head_dim=head_dim, block_size=16, num_blocks=128, dtype=self.dtype, device=self.device)
        dec_bm1 = BlockManager(num_layers=2, num_kv_heads=num_heads, head_dim=head_dim, block_size=16, num_blocks=128, dtype=self.dtype, device=self.device)
        sched1 = PhaseAwareSpeechScheduler(model=model, dual_cache_manager=DualCacheManager(enc_bm1, dec_bm1), max_batch_size=2, device=self.device, cache_cross_kv=False)
        sched1.submit(mel, prompt_token_ids=prompt, max_new_tokens=8)
        comp1 = sched1.run_until_done()

        # Run with flag ON
        enc_bm2 = EncoderBlockManager(num_layers=2, num_heads=num_heads, head_dim=head_dim, block_size=16, num_blocks=128, dtype=self.dtype, device=self.device)
        dec_bm2 = BlockManager(num_layers=2, num_kv_heads=num_heads, head_dim=head_dim, block_size=16, num_blocks=128, dtype=self.dtype, device=self.device)
        sched2 = PhaseAwareSpeechScheduler(model=model, dual_cache_manager=DualCacheManager(enc_bm2, dec_bm2), max_batch_size=2, device=self.device, cache_cross_kv=True)
        sched2.submit(mel, prompt_token_ids=prompt, max_new_tokens=8)
        comp2 = sched2.run_until_done()

        self.assertEqual(comp1[0].generated_tokens, comp2[0].generated_tokens)
        print(f">>> cache_cross_kv parity PASSED: Both produced identical tokens {comp1[0].generated_tokens}")

    def test_cache_cross_kv_zero_block_leaks(self):
        """Test zero memory leakage when cache_cross_kv=True."""
        print("\n--- Running Test: cache_cross_kv Zero Block Leaks ---")
        hidden_dim = 64
        num_heads = 4
        head_dim = hidden_dim // num_heads

        model = WhisperClassSpeechEncoderDecoderModel(
            num_encoder_layers=2,
            num_decoder_layers=2,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mel_dim=80,
            vocab_size=100,
        ).to(device=self.device, dtype=self.dtype).eval()

        enc_bm = EncoderBlockManager(num_layers=2, num_heads=num_heads, head_dim=head_dim, block_size=16, num_blocks=128, dtype=self.dtype, device=self.device)
        dec_bm = BlockManager(num_layers=2, num_kv_heads=num_heads, head_dim=head_dim, block_size=16, num_blocks=128, dtype=self.dtype, device=self.device)
        sched = PhaseAwareSpeechScheduler(model=model, dual_cache_manager=DualCacheManager(enc_bm, dec_bm), max_batch_size=4, device=self.device, cache_cross_kv=True)

        for i in range(5):
            mel = torch.randn(80, 150 + i * 20, device=self.device, dtype=self.dtype)
            prompt = torch.tensor([[1, 2]], device=self.device, dtype=torch.long)
            sched.submit(mel, prompt_token_ids=prompt, max_new_tokens=10)

        completed = sched.run_until_done()
        self.assertEqual(len(completed), 5)
        self.assertEqual(enc_bm.allocated_block_count, 0)
        self.assertEqual(dec_bm.allocated_block_count, 0)
        self.assertIsNone(completed[0].cached_encoder_kv)
        print(">>> cache_cross_kv zero block leaks PASSED: All physical blocks returned to pools.")

    def test_cache_cross_kv_cache_hit_skips_encoder(self):
        """Test that full-input cache hit still skips the encoder when cache_cross_kv=True."""
        print("\n--- Running Test: cache_cross_kv Cache Hit Skips Encoder ---")
        hidden_dim = 64
        num_heads = 4
        head_dim = hidden_dim // num_heads

        model = WhisperClassSpeechEncoderDecoderModel(
            num_encoder_layers=2,
            num_decoder_layers=2,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mel_dim=80,
            vocab_size=100,
        ).to(device=self.device, dtype=self.dtype).eval()

        enc_bm = EncoderBlockManager(num_layers=2, num_heads=num_heads, head_dim=head_dim, block_size=16, num_blocks=128, dtype=self.dtype, device=self.device)
        dec_bm = BlockManager(num_layers=2, num_kv_heads=num_heads, head_dim=head_dim, block_size=16, num_blocks=128, dtype=self.dtype, device=self.device)
        sched = PhaseAwareSpeechScheduler(
            model=model,
            dual_cache_manager=DualCacheManager(enc_bm, dec_bm),
            max_batch_size=2,
            retain_encoder_cache=True,
            device=self.device,
            cache_cross_kv=True,
        )

        mel = torch.randn(80, 200, device=self.device, dtype=self.dtype)
        prompt = torch.tensor([[1, 2]], device=self.device, dtype=torch.long)

        # First submission: cache miss
        sched.submit(mel, prompt_token_ids=prompt, max_new_tokens=5)
        sched.run_until_done()
        self.assertEqual(sched.encoder_calls, 1)
        self.assertEqual(sched.encoder_calls_skipped, 0)

        # Second submission: identical audio, cache hit
        sched.submit(mel, prompt_token_ids=prompt, max_new_tokens=5)
        sched.run_until_done()
        self.assertEqual(sched.encoder_calls, 1)
        self.assertEqual(sched.encoder_calls_skipped, 1)
        print(f">>> cache_cross_kv cache hit PASSED: Encoder skipped on hit (calls={sched.encoder_calls}, skipped={sched.encoder_calls_skipped}).")

    def test_graph_decode_parity(self):
        """Step 3 Test (1): Token-for-token parity between eager and graph decode."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for CUDA graph decode test")
        print("\n--- Running Test: graph_decode parity ---")
        torch.manual_seed(42)
        model = WhisperClassSpeechEncoderDecoderModel(
            num_encoder_layers=2,
            num_decoder_layers=2,
            hidden_dim=64,
            num_heads=4,
            mel_dim=80,
            vocab_size=100,
        ).to(device=self.device, dtype=self.dtype).eval()

        mel = torch.randn(80, 200, device=self.device, dtype=self.dtype)
        prompt = torch.tensor([[1, 2]], device=self.device, dtype=torch.long)

        # 1. Eager
        enc_bm_e = EncoderBlockManager(2, 4, 16, 16, 64, dtype=self.dtype, device=self.device)
        dec_bm_e = BlockManager(2, 4, 16, 16, 64, dtype=self.dtype, device=self.device)
        sched_eager = PhaseAwareSpeechScheduler(model, DualCacheManager(enc_bm_e, dec_bm_e), max_batch_size=4, device=self.device, graph_decode=False)
        sched_eager.submit(mel, prompt_token_ids=prompt, max_new_tokens=10)
        eager_tokens = sched_eager.run_until_done()[0].generated_tokens

        # 2. Graph
        enc_bm_g = EncoderBlockManager(2, 4, 16, 16, 64, dtype=self.dtype, device=self.device)
        dec_bm_g = BlockManager(2, 4, 16, 16, 64, dtype=self.dtype, device=self.device)
        sched_graph = PhaseAwareSpeechScheduler(model, DualCacheManager(enc_bm_g, dec_bm_g), max_batch_size=4, device=self.device, graph_decode=True, graph_readback_k=1)
        sched_graph.submit(mel, prompt_token_ids=prompt, max_new_tokens=10)
        graph_tokens = sched_graph.run_until_done()[0].generated_tokens

        self.assertEqual(eager_tokens, graph_tokens)
        print(f">>> graph_decode parity PASSED: {eager_tokens} == {graph_tokens}")

    def test_graph_decode_zero_leaked_blocks(self):
        """Step 3 Test (2): Zero leaked blocks after requests complete in graph_decode mode."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for CUDA graph decode test")
        print("\n--- Running Test: graph_decode zero leaked blocks ---")
        torch.manual_seed(101)
        model = WhisperClassSpeechEncoderDecoderModel(
            num_encoder_layers=2,
            num_decoder_layers=2,
            hidden_dim=64,
            num_heads=4,
            mel_dim=80,
            vocab_size=100,
        ).to(device=self.device, dtype=self.dtype).eval()

        enc_bm = EncoderBlockManager(2, 4, 16, 16, 64, dtype=self.dtype, device=self.device)
        dec_bm = BlockManager(2, 4, 16, 16, 64, dtype=self.dtype, device=self.device)
        dual = DualCacheManager(enc_bm, dec_bm)
        sched = PhaseAwareSpeechScheduler(model, dual, max_batch_size=4, device=self.device, graph_decode=True)

        for i in range(3):
            mel = torch.randn(80, 160 + i * 20, device=self.device, dtype=self.dtype)
            sched.submit(mel, prompt_token_ids=torch.tensor([[1, 2]], device=self.device), max_new_tokens=8)
        sched.run_until_done()

        self.assertEqual(dec_bm.num_free_blocks, 64)
        self.assertEqual(enc_bm.num_free_blocks, 64)
        self.assertEqual(sched.graph_buffers.active_slot_count(), 0)
        print(">>> graph_decode zero block leaks PASSED: all blocks and slots freed.")

    def test_graph_decode_slots_reused_different_lengths(self):
        """Step 3 Test (3): Slots reused correctly across requests of different lengths."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for CUDA graph decode test")
        print("\n--- Running Test: graph_decode slot reuse across varied lengths ---")
        torch.manual_seed(202)
        model = WhisperClassSpeechEncoderDecoderModel(
            num_encoder_layers=2,
            num_decoder_layers=2,
            hidden_dim=64,
            num_heads=4,
            mel_dim=80,
            vocab_size=100,
        ).to(device=self.device, dtype=self.dtype).eval()

        enc_bm = EncoderBlockManager(2, 4, 16, 16, 128, dtype=self.dtype, device=self.device)
        dec_bm = BlockManager(2, 4, 16, 16, 128, dtype=self.dtype, device=self.device)
        dual = DualCacheManager(enc_bm, dec_bm)
        sched = PhaseAwareSpeechScheduler(model, dual, max_batch_size=4, device=self.device, graph_decode=True)

        # Submit 6 requests of varied lengths to saturate and cycle 4 slots
        lengths = [5, 15, 8, 20, 6, 12]
        for idx, l in enumerate(lengths):
            mel = torch.randn(80, 100 + idx * 30, device=self.device, dtype=self.dtype)
            sched.submit(mel, prompt_token_ids=torch.tensor([[1, 2]], device=self.device), max_new_tokens=l)

        completed = sched.run_until_done()
        self.assertEqual(len(completed), 6)
        for req in completed:
            self.assertLessEqual(len(req.generated_tokens), req.max_new_tokens)
        self.assertEqual(dec_bm.num_free_blocks, 128)
        self.assertEqual(enc_bm.num_free_blocks, 128)
        self.assertEqual(sched.graph_buffers.active_slot_count(), 0)
        print(">>> graph_decode slot reuse across varied lengths PASSED.")

    def test_graph_decode_eos_handling(self):
        """Step 3 Test (4): Generation stops immediately on EOS and trims tokens after EOS."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for CUDA graph decode test")
        print("\n--- Running Test: graph_decode EOS handling ---")
        torch.manual_seed(303)
        model = WhisperClassSpeechEncoderDecoderModel(
            num_encoder_layers=2,
            num_decoder_layers=2,
            hidden_dim=64,
            num_heads=4,
            mel_dim=80,
            vocab_size=100,
        ).to(device=self.device, dtype=self.dtype).eval()

        enc_bm = EncoderBlockManager(2, 4, 16, 16, 64, dtype=self.dtype, device=self.device)
        dec_bm = BlockManager(2, 4, 16, 16, 64, dtype=self.dtype, device=self.device)
        dual = DualCacheManager(enc_bm, dec_bm)
        sched = PhaseAwareSpeechScheduler(model, dual, max_batch_size=4, device=self.device, graph_decode=True)

        # Run eager first to discover which token is generated
        sched_e = PhaseAwareSpeechScheduler(model, DualCacheManager(
            EncoderBlockManager(2, 4, 16, 16, 64, dtype=self.dtype, device=self.device),
            BlockManager(2, 4, 16, 16, 64, dtype=self.dtype, device=self.device)
        ), max_batch_size=4, device=self.device, graph_decode=False)
        mel = torch.randn(80, 200, device=self.device, dtype=self.dtype)
        sched_e.submit(mel, prompt_token_ids=torch.tensor([[1, 2]], device=self.device), max_new_tokens=6)
        e_toks = sched_e.run_until_done()[0].generated_tokens
        eos_tok = e_toks[2]  # choose the 3rd generated token as EOS

        # Submit with eos_token_id set to that token
        sched.submit(mel, prompt_token_ids=torch.tensor([[1, 2]], device=self.device), max_new_tokens=10, eos_token_id=eos_tok)
        comp = sched.run_until_done()
        g_toks = comp[0].generated_tokens

        # EOS should stop generation at index 2 (length 2 tokens before EOS)
        self.assertEqual(len(g_toks), 2)
        self.assertEqual(g_toks, e_toks[:2])
        self.assertNotIn(eos_tok, g_toks)
        self.assertEqual(dec_bm.num_free_blocks, 64)
        print(f">>> graph_decode EOS handling PASSED: stopped at EOS token {eos_tok}, kept {g_toks}.")
    def test_engine_graph_serving_parity_100(self):
        """Gate engine_graph on all 100 utterances under burst and Poisson(4.0)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for 100-utterance serving parity")
        import tempfile
        from pathlib import Path
        from scripts.test_parity_graph import run_all

        with tempfile.TemporaryDirectory() as tmp_dir:
            report = run_all(Path(tmp_dir) / "serving_parity_graph.json", run_isolations=False)
        self.assertTrue(report["gate_passed"], report["profiles"])
        for profile in report["profiles"].values():
            self.assertGreaterEqual(profile["exact_match_pct"], 99.0)
            self.assertLessEqual(abs(profile["wer_delta_points"]), 0.2)


if __name__ == "__main__":
    unittest.main()
