"""
Unit tests for cache lifecycle operation IO structs and base cache interactions.

Tests:
- EvictPrefixReqInput/Output, DemotePrefixReqInput/Output, PromotePrefixReqInput/Output
- Dataclass field defaults, serialization
- RadixCache tree walking patterns (find_node_by_token_ids uses inherited methods)

Usage:
    python -m pytest test_cache_lifecycle_ops.py -v
"""

from sglang.test.ci.ci_register import register_cuda_ci

# CPU-based unit test, runs quickly on any GPU runner
register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")

import unittest

from sglang.srt.managers.io_struct import (
    EvictPrefixReqInput,
    EvictPrefixReqOutput,
    DemotePrefixReqInput,
    DemotePrefixReqOutput,
    PromotePrefixReqInput,
    PromotePrefixReqOutput,
    RegisterPrefixOwnerReqInput,
    RegisterPrefixOwnerReqOutput,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode


class TestEvictPrefixIOStructs(unittest.TestCase):
    """Test EvictPrefixReqInput and EvictPrefixReqOutput dataclasses."""

    def test_input_defaults(self):
        req = EvictPrefixReqInput()
        self.assertEqual(req.token_ids, [])
        self.assertFalse(req.force)

    def test_input_with_values(self):
        req = EvictPrefixReqInput(token_ids=[1, 2, 3], force=True)
        self.assertEqual(req.token_ids, [1, 2, 3])
        self.assertTrue(req.force)

    def test_output_defaults(self):
        out = EvictPrefixReqOutput()
        self.assertFalse(out.success)
        self.assertEqual(out.num_tokens_evicted, 0)
        self.assertEqual(out.message, "")

    def test_output_with_values(self):
        out = EvictPrefixReqOutput(
            success=True, num_tokens_evicted=128, message="evicted 128 tokens"
        )
        self.assertTrue(out.success)
        self.assertEqual(out.num_tokens_evicted, 128)
        self.assertEqual(out.message, "evicted 128 tokens")


class TestDemotePrefixIOStructs(unittest.TestCase):
    """Test DemotePrefixReqInput and DemotePrefixReqOutput dataclasses."""

    def test_input_defaults(self):
        req = DemotePrefixReqInput()
        self.assertEqual(req.token_ids, [])
        self.assertEqual(req.target, "host")

    def test_input_with_storage_target(self):
        req = DemotePrefixReqInput(token_ids=[10, 20], target="storage")
        self.assertEqual(req.target, "storage")

    def test_output_defaults(self):
        out = DemotePrefixReqOutput()
        self.assertFalse(out.success)
        self.assertEqual(out.num_tokens_demoted, 0)

    def test_output_with_values(self):
        out = DemotePrefixReqOutput(
            success=True, num_tokens_demoted=64, message="demoted 64 tokens"
        )
        self.assertTrue(out.success)
        self.assertEqual(out.num_tokens_demoted, 64)


class TestPromotePrefixIOStructs(unittest.TestCase):
    """Test PromotePrefixReqInput and PromotePrefixReqOutput dataclasses."""

    def test_input_defaults(self):
        req = PromotePrefixReqInput()
        self.assertEqual(req.token_ids, [])

    def test_input_with_values(self):
        req = PromotePrefixReqInput(token_ids=[5, 6, 7, 8])
        self.assertEqual(req.token_ids, [5, 6, 7, 8])

    def test_output_defaults(self):
        out = PromotePrefixReqOutput()
        self.assertFalse(out.success)
        self.assertEqual(out.num_tokens_promoted, 0)

    def test_output_with_values(self):
        out = PromotePrefixReqOutput(
            success=True, num_tokens_promoted=256, message="promoted 256 tokens"
        )
        self.assertTrue(out.success)
        self.assertEqual(out.num_tokens_promoted, 256)


class TestIOStructsInheritBaseReq(unittest.TestCase):
    """Verify all new IO structs properly inherit from BaseReq."""

    def test_evict_input_has_rid(self):
        req = EvictPrefixReqInput(token_ids=[1, 2], rid="test-rid-1")
        self.assertEqual(req.rid, "test-rid-1")

    def test_demote_input_has_rid(self):
        req = DemotePrefixReqInput(token_ids=[1], rid="test-rid-2")
        self.assertEqual(req.rid, "test-rid-2")

    def test_promote_input_has_rid(self):
        req = PromotePrefixReqInput(token_ids=[1], rid="test-rid-3")
        self.assertEqual(req.rid, "test-rid-3")

    def test_evict_output_has_rid(self):
        out = EvictPrefixReqOutput(success=True, rid="test-rid-4")
        self.assertEqual(out.rid, "test-rid-4")

    def test_regenerate_rid(self):
        req = EvictPrefixReqInput(token_ids=[1, 2])
        old_rid = req.rid
        new_rid = req.regenerate_rid()
        self.assertNotEqual(old_rid, new_rid)
        self.assertEqual(req.rid, new_rid)


class TestRadixCacheTreeWalkPatterns(unittest.TestCase):
    """Test tree walking patterns used by find_node_by_token_ids.

    Uses RadixCache.create_simulated() which works without GPU.
    Tests the tree structure operations that underpin the HiRadixCache methods.
    """

    def test_insert_and_match_prefix(self):
        """Verify basic insert + match works (pattern used by find_node)."""
        import torch

        cache = RadixCache.create_simulated()
        # Insert a sequence
        key = RadixKey([1, 2, 3, 4, 5])
        value = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64)
        cache.insert(InsertParams(key=key, value=value))

        # Match prefix should find it
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5])))
        self.assertIsNotNone(result)
        self.assertEqual(len(result.device_indices), 5)

    def test_partial_prefix_match(self):
        """Partial prefix returns matched portion."""
        import torch

        cache = RadixCache.create_simulated()
        key = RadixKey([1, 2, 3, 4, 5, 6, 7, 8])
        value = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int64)
        cache.insert(InsertParams(key=key, value=value))

        result = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3])))
        self.assertIsNotNone(result)
        self.assertGreater(len(result.device_indices), 0)

    def test_no_match_returns_empty(self):
        """Unknown prefix returns empty match."""
        cache = RadixCache.create_simulated()
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey([99, 98, 97])))
        self.assertEqual(len(result.device_indices), 0)

    def test_tree_children_structure(self):
        """Verify tree builds correct parent-child relationships."""
        import torch

        cache = RadixCache.create_simulated()

        # Insert two sequences sharing a prefix
        cache.insert(InsertParams(
            key=RadixKey([1, 2, 3, 4, 5]),
            value=torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64),
        ))
        cache.insert(InsertParams(
            key=RadixKey([1, 2, 3, 6, 7]),
            value=torch.tensor([10, 20, 30, 60, 70], dtype=torch.int64),
        ))

        # Root should have a child, and that child should have children
        self.assertGreater(len(cache.root_node.children), 0)

    def test_eviction_removes_leaf(self):
        """Eviction removes leaf nodes correctly."""
        import torch
        from unittest.mock import MagicMock

        mock_allocator = MagicMock()
        mock_allocator.free = MagicMock()
        cache = RadixCache.create_simulated(mock_allocator=mock_allocator)
        key = RadixKey([1, 2, 3, 4])
        value = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        cache.insert(InsertParams(key=key, value=value))

        # There should be evictable leaves
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4])))
        last_node = result.last_device_node

        # Decrement lock to make evictable
        if last_node and last_node.lock_ref > 0:
            cache.dec_lock_ref(last_node)

        # Now evict
        evict_result = cache.evict(EvictParams(num_tokens=4))
        self.assertGreater(evict_result.num_tokens_evicted, 0)
        mock_allocator.free.assert_called()


class TestSubtreeCollectionPattern(unittest.TestCase):
    """Test the DFS subtree collection pattern used by _collect_subtree_nodes."""

    def test_post_order_collection(self):
        """Verify post-order DFS collects children before parents."""
        # Create manual tree nodes
        parent = TreeNode()
        parent.key = RadixKey([1, 2])
        child1 = TreeNode()
        child1.key = RadixKey([3, 4])
        child1.parent = parent
        child2 = TreeNode()
        child2.key = RadixKey([5, 6])
        child2.parent = parent
        parent.children = {3: child1, 5: child2}

        # Manual post-order DFS (same algorithm as _collect_subtree_nodes)
        result = []
        visited = set()

        def _dfs(n):
            if id(n) in visited:
                return
            visited.add(id(n))
            for child in list(n.children.values()):
                _dfs(child)
            result.append(n)

        _dfs(parent)

        # Children should come before parent
        self.assertEqual(len(result), 3)
        self.assertIn(child1, result[:2])
        self.assertIn(child2, result[:2])
        self.assertEqual(result[-1], parent)

    def test_single_node_subtree(self):
        """Single node returns just that node."""
        node = TreeNode()
        node.key = RadixKey([1, 2, 3])

        result = []
        visited = set()

        def _dfs(n):
            if id(n) in visited:
                return
            visited.add(id(n))
            for child in list(n.children.values()):
                _dfs(child)
            result.append(n)

        _dfs(node)
        self.assertEqual(result, [node])

    def test_deep_subtree(self):
        """Multi-level tree collected in correct order."""
        root = TreeNode()
        root.key = RadixKey([1])
        mid = TreeNode()
        mid.key = RadixKey([2])
        mid.parent = root
        leaf = TreeNode()
        leaf.key = RadixKey([3])
        leaf.parent = mid
        mid.children = {3: leaf}
        root.children = {2: mid}

        result = []
        visited = set()

        def _dfs(n):
            if id(n) in visited:
                return
            visited.add(id(n))
            for child in list(n.children.values()):
                _dfs(child)
            result.append(n)

        _dfs(root)

        self.assertEqual(result, [leaf, mid, root])


class TestBaseRadixCacheFindNode(unittest.TestCase):
    """Test find_node_by_token_ids on the base RadixCache (no HiCache)."""

    def test_find_exact_match(self):
        """Find a node matching the exact inserted sequence."""
        import torch

        cache = RadixCache.create_simulated()
        cache.insert(InsertParams(
            key=RadixKey([1, 2, 3, 4]),
            value=torch.tensor([10, 20, 30, 40], dtype=torch.int64),
        ))
        node = cache.find_node_by_token_ids([1, 2, 3, 4])
        self.assertIsNotNone(node)

    def test_find_partial_match(self):
        """Partial token_ids find the deepest fully-matched node."""
        import torch

        cache = RadixCache.create_simulated()
        cache.insert(InsertParams(
            key=RadixKey([1, 2, 3, 4, 5, 6]),
            value=torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.int64),
        ))
        node = cache.find_node_by_token_ids([1, 2, 3])
        # Should find something (may be the full node since it's one segment)
        self.assertIsNotNone(node)

    def test_find_no_match(self):
        """Non-existent tokens return None."""
        cache = RadixCache.create_simulated()
        node = cache.find_node_by_token_ids([99, 98, 97])
        self.assertIsNone(node)

    def test_find_empty_tokens(self):
        """Empty token list returns None."""
        cache = RadixCache.create_simulated()
        node = cache.find_node_by_token_ids([])
        self.assertIsNone(node)


class TestBaseRadixCacheEvictPrefix(unittest.TestCase):
    """Test evict_prefix on the base RadixCache (no HiCache).

    This validates the evict_prefix path used when the scheduler
    receives cache_action=evict without --enable-hierarchical-cache.
    (demote_to_host/demote_to_storage are no-ops without HiCache.)
    """

    def test_evict_prefix_basic(self):
        """Evict a prefix from the base RadixCache."""
        import torch
        from unittest.mock import MagicMock

        mock_allocator = MagicMock()
        cache = RadixCache.create_simulated(mock_allocator=mock_allocator)
        cache.insert(InsertParams(
            key=RadixKey([1, 2, 3, 4]),
            value=torch.tensor([10, 20, 30, 40], dtype=torch.int64),
        ))

        # Unlock the node so it's evictable
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4])))
        if result.last_device_node.lock_ref > 0:
            cache.dec_lock_ref(result.last_device_node)

        num_evicted, error = cache.evict_prefix([1, 2, 3, 4])
        self.assertIsNone(error)
        self.assertGreater(num_evicted, 0)
        mock_allocator.free.assert_called()

    def test_evict_prefix_not_found(self):
        """Evicting a non-existent prefix returns error cleanly."""
        cache = RadixCache.create_simulated()
        num_evicted, error = cache.evict_prefix([99, 98, 97])
        self.assertEqual(num_evicted, 0)
        self.assertIn("not found", error)

    def test_evict_prefix_locked_node(self):
        """Locked nodes are skipped, not crashed on."""
        import torch
        from unittest.mock import MagicMock

        mock_allocator = MagicMock()
        cache = RadixCache.create_simulated(mock_allocator=mock_allocator)
        cache.insert(InsertParams(
            key=RadixKey([1, 2, 3, 4]),
            value=torch.tensor([10, 20, 30, 40], dtype=torch.int64),
        ))
        # Simulate an active request holding a lock on the node
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4])))
        cache.inc_lock_ref(result.last_device_node)

        num_evicted, error = cache.evict_prefix([1, 2, 3, 4])
        self.assertEqual(num_evicted, 0)
        self.assertIsNotNone(error)
        self.assertIn("locked", error)

    def test_evict_prefix_with_subtree(self):
        """Eviction removes parent and child nodes."""
        import torch
        from unittest.mock import MagicMock

        mock_allocator = MagicMock()
        cache = RadixCache.create_simulated(mock_allocator=mock_allocator)

        # Insert two sequences sharing a prefix to create branching
        cache.insert(InsertParams(
            key=RadixKey([1, 2, 3, 4, 5]),
            value=torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64),
        ))
        cache.insert(InsertParams(
            key=RadixKey([1, 2, 3, 6, 7]),
            value=torch.tensor([10, 20, 30, 60, 70], dtype=torch.int64),
        ))

        # Unlock all nodes
        r1 = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5])))
        r2 = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 6, 7])))
        cache.dec_lock_ref(r1.last_device_node)
        cache.dec_lock_ref(r2.last_device_node)

        # Evict from the shared prefix [1, 2, 3] downward
        num_evicted, error = cache.evict_prefix([1, 2, 3])
        self.assertIsNone(error)
        self.assertGreater(num_evicted, 0)


class TestRegisterPrefixOwnerIOStructs(unittest.TestCase):
    """Test RegisterPrefixOwnerReqInput and RegisterPrefixOwnerReqOutput."""

    def test_input_defaults(self):
        req = RegisterPrefixOwnerReqInput()
        self.assertEqual(req.token_ids, [])
        self.assertEqual(req.prefix_id, "")

    def test_input_with_values(self):
        req = RegisterPrefixOwnerReqInput(token_ids=[1, 2, 3], prefix_id="agent-A")
        self.assertEqual(req.token_ids, [1, 2, 3])
        self.assertEqual(req.prefix_id, "agent-A")

    def test_output_defaults(self):
        out = RegisterPrefixOwnerReqOutput()
        self.assertTrue(out.success)

    def test_evict_prefix_id_field(self):
        req = EvictPrefixReqInput(token_ids=[1], prefix_id="agent-A")
        self.assertEqual(req.prefix_id, "agent-A")

    def test_demote_prefix_id_field(self):
        req = DemotePrefixReqInput(token_ids=[1], prefix_id="agent-B")
        self.assertEqual(req.prefix_id, "agent-B")


class TestPrefixOwnership(unittest.TestCase):
    """Two agents share a prefix; single-agent eviction preserves shared nodes."""

    def test_shared_prefix_survives_single_agent_eviction(self):
        import torch
        from unittest.mock import MagicMock

        mock_allocator = MagicMock()
        cache = RadixCache.create_simulated(mock_allocator=mock_allocator)

        # Build tree: shared prefix [1,2,3] + agent A tail [4,5] + agent B tail [6,7]
        cache.insert(InsertParams(
            key=RadixKey([1, 2, 3, 4, 5]),
            value=torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64),
        ))
        cache.insert(InsertParams(
            key=RadixKey([1, 2, 3, 6, 7]),
            value=torch.tensor([10, 20, 30, 60, 70], dtype=torch.int64),
        ))

        # Unlock all nodes
        r1 = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5])))
        r2 = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 6, 7])))
        cache.dec_lock_ref(r1.last_device_node)
        cache.dec_lock_ref(r2.last_device_node)

        # Register ownership: A owns [1,2,3,4,5], B owns [1,2,3,6,7]
        cache.register_prefix_owner([1, 2, 3, 4, 5], "A")
        cache.register_prefix_owner([1, 2, 3, 6, 7], "B")

        # Evict with prefix_id="A" — should only evict A's exclusive nodes
        num_evicted, error = cache.evict_prefix([1, 2, 3, 4, 5], prefix_id="A")
        self.assertIsNone(error)
        self.assertGreater(num_evicted, 0)

        # The shared prefix [1,2,3] should still be findable (B still owns it)
        shared_node = cache.find_node_by_token_ids([1, 2, 3])
        self.assertIsNotNone(shared_node)

        # Agent B's nodes should still be intact
        b_node = cache.find_node_by_token_ids([1, 2, 3, 6, 7])
        self.assertIsNotNone(b_node)


class TestPrefixOwnershipFallback(unittest.TestCase):
    """prefix_id=None -> full subtree eviction (backward compat)."""

    def test_no_prefix_id_evicts_all(self):
        import torch
        from unittest.mock import MagicMock

        mock_allocator = MagicMock()
        cache = RadixCache.create_simulated(mock_allocator=mock_allocator)

        cache.insert(InsertParams(
            key=RadixKey([1, 2, 3, 4, 5]),
            value=torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64),
        ))
        cache.insert(InsertParams(
            key=RadixKey([1, 2, 3, 6, 7]),
            value=torch.tensor([10, 20, 30, 60, 70], dtype=torch.int64),
        ))

        # Unlock all
        r1 = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5])))
        r2 = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 6, 7])))
        cache.dec_lock_ref(r1.last_device_node)
        cache.dec_lock_ref(r2.last_device_node)

        # Register ownership but evict without prefix_id
        cache.register_prefix_owner([1, 2, 3, 4, 5], "A")
        cache.register_prefix_owner([1, 2, 3, 6, 7], "B")

        # Evict with prefix_id=None (default) -> should evict everything
        num_evicted, error = cache.evict_prefix([1, 2, 3])
        self.assertIsNone(error)
        self.assertGreater(num_evicted, 0)

        # Everything should be gone
        node = cache.find_node_by_token_ids([1, 2, 3])
        self.assertIsNone(node)


class TestPrefixOwnershipSplit(unittest.TestCase):
    """Node splitting preserves prefix_owners."""

    def test_split_preserves_owners(self):
        import torch

        cache = RadixCache.create_simulated()

        # Insert [1,2,3,4,5] first
        cache.insert(InsertParams(
            key=RadixKey([1, 2, 3, 4, 5]),
            value=torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64),
        ))

        # Register A as owner of [1,2,3,4,5]
        cache.register_prefix_owner([1, 2, 3, 4, 5], "A")

        # Now insert [1,2,3,6,7] which forces a split at [1,2,3]
        cache.insert(InsertParams(
            key=RadixKey([1, 2, 3, 6, 7]),
            value=torch.tensor([10, 20, 30, 60, 70], dtype=torch.int64),
        ))

        # The split should create an intermediate node [1,2,3] that inherits A's ownership
        shared_node = cache.find_node_by_token_ids([1, 2, 3])
        self.assertIsNotNone(shared_node)
        self.assertIn("A", shared_node.prefix_owners)

    def test_tree_node_has_prefix_owners(self):
        """TreeNode initializes with empty prefix_owners set."""
        node = TreeNode()
        self.assertIsInstance(node.prefix_owners, set)
        self.assertEqual(len(node.prefix_owners), 0)


if __name__ == "__main__":
    unittest.main()
