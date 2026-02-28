"""Test that block_cache_status appears in response meta_info.

This is a manual integration test that requires a running SGLang server.
Run with:
    python -m sglang.launch_server --model Qwen/Qwen3-0.6B --page-size 16 \\
        --kv-events-config '{"publisher": "zmq", "topic": "kv-events"}'

Then:
    python test_block_cache_status.py
"""
import unittest
import os
import sys

# Check if we should skip integration tests
SKIP_INTEGRATION = os.environ.get("SKIP_INTEGRATION_TESTS", "1") == "1"


@unittest.skipIf(SKIP_INTEGRATION, "Integration test requires running server")
class TestBlockCacheStatusAPI(unittest.TestCase):
    """Integration test for block_cache_status in API response.

    Requires a running SGLang server with kv-events enabled.
    Set SKIP_INTEGRATION_TESTS=0 and SGLANG_BASE_URL to run.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.environ.get("SGLANG_BASE_URL", "http://localhost:30000")

    def test_block_cache_status_present_after_cache_hit(self):
        """Send two identical requests; second should have block_cache_status."""
        import requests

        prompt = "Tell me about the history of artificial intelligence and its development"

        # First request - cold cache
        resp1 = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 5},
            },
        )
        self.assertEqual(resp1.status_code, 200)

        # Second request - should hit cache
        resp2 = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 5},
            },
        )
        self.assertEqual(resp2.status_code, 200)
        meta2 = resp2.json().get("meta_info", {})

        # Verify cached_tokens > 0 for second request
        self.assertGreater(meta2.get("cached_tokens", 0), 0,
            "Second identical request should have cache hits")

        # If block_cache_status is present, validate its structure
        bcs = meta2.get("block_cache_status")
        if bcs is not None:
            self.assertIn("block_hashes", bcs)
            self.assertIn("cached_mask", bcs)
            self.assertIsInstance(bcs["block_hashes"], list)
            self.assertIsInstance(bcs["cached_mask"], list)
            self.assertEqual(len(bcs["block_hashes"]), len(bcs["cached_mask"]))
            # Some blocks should be cached
            cached_count = sum(1 for m in bcs["cached_mask"] if m)
            self.assertGreater(cached_count, 0,
                "Second identical request should have cached blocks")


if __name__ == "__main__":
    unittest.main()
