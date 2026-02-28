"""
Unit tests for BlockAccessed KV cache event.

Tests the BlockAccessed event class, its serialization via msgspec msgpack,
and the RadixCache._record_access_event() method that produces these events.

Test Coverage:
- test_block_accessed_event: _record_access_event() produces correct BlockAccessed
- test_block_accessed_serialization: msgpack encode/decode roundtrip
- test_block_accessed_empty_input: empty fill_ids produces no event
- test_block_accessed_all_cached: all blocks are cache hits
- test_block_accessed_none_cached: cold cache, no blocks cached

Usage:
    python test_block_accessed_event.py
    python -m pytest test_block_accessed_event.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-based unit test, runs quickly on any GPU runner
register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=5, suite="stage-b-test-small-1-gpu-amd")

import unittest

import msgspec

from sglang.srt.disaggregation.kv_events import (
    MEDIUM_CPU,
    MEDIUM_GPU,
    BlockAccessed,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


class TestBlockAccessedEvent(unittest.TestCase):
    """Test _record_access_event() on RadixCache produces correct BlockAccessed events."""

    def setUp(self):
        """Create a RadixCache with events enabled and page_size=4."""
        self.page_size = 4
        self.cache = RadixCache.create_simulated(
            page_size=self.page_size,
            enable_kv_cache_events=True,
        )

    def test_block_accessed_event(self):
        """_record_access_event() produces a BlockAccessed with proper fields."""
        # Simulate a request with 8 tokens (2 blocks of page_size=4),
        # where the first 4 tokens (1 block) were cached on device.
        fill_ids = list(range(1, 9))  # [1, 2, 3, 4, 5, 6, 7, 8]

        self.cache._record_access_event(
            req_id="req-test-001",
            fill_ids=fill_ids,
            cached_token_count=4,   # 1 block cached
            cached_tokens_device=4, # all cached on GPU
            cached_tokens_host=0,
        )

        events = self.cache.take_events()
        self.assertEqual(len(events), 1)

        event = events[0]
        self.assertIsInstance(event, BlockAccessed)
        self.assertEqual(event.request_id, "req-test-001")
        self.assertEqual(len(event.block_hashes), 2)
        self.assertEqual(event.num_cached, 1)
        self.assertEqual(event.num_prefilled, 1)
        self.assertEqual(event.cached_mask, [True, False])
        self.assertEqual(event.medium_per_block, [MEDIUM_GPU, None])

        # block_hashes should be non-zero int64 values
        for h in event.block_hashes:
            self.assertIsInstance(h, int)

    def test_block_accessed_serialization(self):
        """BlockAccessed survives msgspec msgpack encode/decode roundtrip."""
        event = BlockAccessed(
            block_hashes=[100, 200, 300],
            request_id="req-serial-001",
            num_cached=2,
            num_prefilled=1,
            cached_mask=[True, True, False],
            medium_per_block=[MEDIUM_GPU, MEDIUM_CPU, None],
        )

        batch = KVEventBatch(ts=1234567890.0, events=[event])

        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

        payload = encoder.encode(batch)
        decoded = decoder.decode(payload)

        self.assertEqual(len(decoded.events), 1)
        decoded_event = decoded.events[0]
        self.assertIsInstance(decoded_event, BlockAccessed)
        self.assertEqual(decoded_event.block_hashes, [100, 200, 300])
        self.assertEqual(decoded_event.request_id, "req-serial-001")
        self.assertEqual(decoded_event.num_cached, 2)
        self.assertEqual(decoded_event.num_prefilled, 1)
        self.assertEqual(decoded_event.cached_mask, [True, True, False])
        self.assertEqual(decoded_event.medium_per_block, [MEDIUM_GPU, MEDIUM_CPU, None])

    def test_block_accessed_empty_input(self):
        """Empty fill_ids produces no event."""
        self.cache._record_access_event(
            req_id="req-empty",
            fill_ids=[],
            cached_token_count=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
        )

        events = self.cache.take_events()
        self.assertEqual(len(events), 0, "Empty fill_ids should produce no event")

    def test_block_accessed_sub_page_input(self):
        """fill_ids shorter than page_size produces no event (page-aligned to 0)."""
        # page_size=4, only 3 tokens => page_aligned_len = 0
        self.cache._record_access_event(
            req_id="req-subpage",
            fill_ids=[1, 2, 3],
            cached_token_count=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
        )

        events = self.cache.take_events()
        self.assertEqual(len(events), 0,
                         "Sub-page fill_ids should produce no event")

    def test_block_accessed_all_cached(self):
        """All blocks are cached (full cache hit)."""
        # 12 tokens = 3 blocks, all cached on GPU
        fill_ids = list(range(1, 13))

        self.cache._record_access_event(
            req_id="req-all-cached",
            fill_ids=fill_ids,
            cached_token_count=12,
            cached_tokens_device=12,
            cached_tokens_host=0,
        )

        events = self.cache.take_events()
        self.assertEqual(len(events), 1)

        event = events[0]
        self.assertIsInstance(event, BlockAccessed)
        self.assertEqual(event.num_cached, 3)
        self.assertEqual(event.num_prefilled, 0)
        self.assertTrue(all(event.cached_mask),
                        "All blocks should be marked as cached")
        self.assertEqual(event.medium_per_block,
                         [MEDIUM_GPU, MEDIUM_GPU, MEDIUM_GPU])

    def test_block_accessed_none_cached(self):
        """Cold cache: no blocks are cached."""
        fill_ids = list(range(1, 13))  # 12 tokens = 3 blocks

        self.cache._record_access_event(
            req_id="req-cold",
            fill_ids=fill_ids,
            cached_token_count=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
        )

        events = self.cache.take_events()
        self.assertEqual(len(events), 1)

        event = events[0]
        self.assertIsInstance(event, BlockAccessed)
        self.assertEqual(event.num_cached, 0)
        self.assertEqual(event.num_prefilled, 3)
        self.assertFalse(any(event.cached_mask),
                         "No blocks should be marked as cached")
        self.assertEqual(event.medium_per_block, [None, None, None])

    def test_block_accessed_mixed_medium(self):
        """Some blocks on GPU, some on CPU (host), some uncached."""
        # 16 tokens = 4 blocks with page_size=4
        # 12 tokens cached total (3 blocks), 8 on device (2 blocks), 4 on host (1 block)
        fill_ids = list(range(1, 17))

        self.cache._record_access_event(
            req_id="req-mixed-medium",
            fill_ids=fill_ids,
            cached_token_count=12,
            cached_tokens_device=8,
            cached_tokens_host=4,
        )

        events = self.cache.take_events()
        self.assertEqual(len(events), 1)

        event = events[0]
        self.assertEqual(len(event.block_hashes), 4)
        self.assertEqual(event.num_cached, 3)
        self.assertEqual(event.num_prefilled, 1)
        self.assertEqual(event.cached_mask, [True, True, True, False])
        self.assertEqual(event.medium_per_block,
                         [MEDIUM_GPU, MEDIUM_GPU, MEDIUM_CPU, None])

    def test_block_accessed_in_mixed_batch_serialization(self):
        """BlockAccessed serializes correctly alongside BlockStored and BlockRemoved."""
        events = [
            BlockStored(
                block_hashes=[111],
                parent_block_hash=None,
                token_ids=[1, 2, 3, 4],
                block_size=4,
                lora_id=None,
                medium=MEDIUM_GPU,
            ),
            BlockAccessed(
                block_hashes=[111, 222],
                request_id="req-mixed-batch",
                num_cached=1,
                num_prefilled=1,
                cached_mask=[True, False],
                medium_per_block=[MEDIUM_GPU, None],
            ),
            BlockRemoved(
                block_hashes=[111],
                medium=MEDIUM_GPU,
            ),
        ]

        batch = KVEventBatch(ts=99.0, events=events)

        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

        payload = encoder.encode(batch)
        decoded = decoder.decode(payload)

        self.assertEqual(len(decoded.events), 3)
        self.assertIsInstance(decoded.events[0], BlockStored)
        self.assertIsInstance(decoded.events[1], BlockAccessed)
        self.assertIsInstance(decoded.events[2], BlockRemoved)

        # Verify the BlockAccessed content survived
        accessed = decoded.events[1]
        self.assertEqual(accessed.block_hashes, [111, 222])
        self.assertEqual(accessed.request_id, "req-mixed-batch")

    def test_block_accessed_events_not_emitted_when_disabled(self):
        """No events produced when enable_kv_cache_events is False."""
        cache_no_events = RadixCache.create_simulated(
            page_size=self.page_size,
            enable_kv_cache_events=False,
        )

        cache_no_events._record_access_event(
            req_id="req-disabled",
            fill_ids=list(range(1, 9)),
            cached_token_count=4,
            cached_tokens_device=4,
            cached_tokens_host=0,
        )

        events = cache_no_events.take_events()
        self.assertEqual(len(events), 0,
                         "Events should not be emitted when disabled")


if __name__ == "__main__":
    unittest.main()
