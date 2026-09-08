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


if __name__ == "__main__":
    unittest.main()
