"""Hierarchical softmax over a Huffman tree (Mikolov et al., 2013).

Each vocabulary word is a leaf. Frequent words get shorter codes. Output
vectors live on the inner nodes; training is binary classification at each
node along the path from the root to the target leaf.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class _HuffmanNode:
    freq: int
    tie: int
    word_id: int | None
    node_id: int | None
    left: _HuffmanNode | None
    right: _HuffmanNode | None

    def __lt__(self, other: _HuffmanNode) -> bool:
        return (self.freq, self.tie) < (other.freq, other.tie)


class HuffmanCoding:
    """Maps each word id to its Huffman path of inner nodes and 0/1 branches."""

    def __init__(
        self,
        path_nodes: torch.Tensor,
        path_codes: torch.Tensor,
        path_mask: torch.Tensor,
        num_inner_nodes: int,
    ) -> None:
        self.path_nodes = path_nodes
        self.path_codes = path_codes
        self.path_mask = path_mask
        self.num_inner_nodes = num_inner_nodes

    @classmethod
    def from_counts(cls, counts: Sequence[int]) -> HuffmanCoding:
        if len(counts) < 2:
            raise ValueError("Hierarchical softmax needs at least two vocabulary items.")

        heap: list[_HuffmanNode] = []
        for word_id, freq in enumerate(counts):
            heapq.heappush(
                heap,
                _HuffmanNode(
                    freq=int(freq),
                    tie=word_id,
                    word_id=word_id,
                    node_id=None,
                    left=None,
                    right=None,
                ),
            )

        next_tie = len(counts)
        next_inner = 0
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = _HuffmanNode(
                freq=left.freq + right.freq,
                tie=next_tie,
                word_id=None,
                node_id=next_inner,
                left=left,
                right=right,
            )
            next_tie += 1
            next_inner += 1
            heapq.heappush(heap, parent)

        root = heap[0]
        paths: list[list[int]] = [[] for _ in counts]
        codes: list[list[int]] = [[] for _ in counts]

        def walk(node: _HuffmanNode, path: list[int], code: list[int]) -> None:
            if node.word_id is not None:
                paths[node.word_id] = path
                codes[node.word_id] = code
                return
            assert node.node_id is not None and node.left is not None and node.right is not None
            walk(node.left, path + [node.node_id], code + [0])
            walk(node.right, path + [node.node_id], code + [1])

        walk(root, [], [])

        max_len = max(len(path) for path in paths)
        vocab_size = len(counts)
        path_nodes = torch.full((vocab_size, max_len), -1, dtype=torch.long)
        path_codes = torch.zeros((vocab_size, max_len), dtype=torch.float32)
        path_mask = torch.zeros((vocab_size, max_len), dtype=torch.bool)
        for word_id, path in enumerate(paths):
            length = len(path)
            path_nodes[word_id, :length] = torch.tensor(path, dtype=torch.long)
            path_codes[word_id, :length] = torch.tensor(codes[word_id], dtype=torch.float32)
            path_mask[word_id, :length] = True

        return cls(
            path_nodes=path_nodes,
            path_codes=path_codes,
            path_mask=path_mask,
            num_inner_nodes=next_inner,
        )


class HierarchicalSoftmax(nn.Module):
    """Output-side hierarchical softmax trained with binary cross entropy.

    `forward(center_index, target_index)` looks up center word vectors, then
    treats each Huffman bit as a Bernoulli label (left = 0, right = 1).
    """

    def __init__(self, coding: HuffmanCoding, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        self.coding = coding
        self.node_embeddings = nn.Embedding(coding.num_inner_nodes, embedding_dim)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.register_buffer("path_nodes", coding.path_nodes)
        self.register_buffer("path_codes", coding.path_codes)
        self.register_buffer("path_mask", coding.path_mask)

    def forward(self, center_index: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
        """Return mean path BCE for skip-gram pairs given as word ids.

        `center_index` and `target_index` are 1-D LongTensors of shape (batch,).
        Each target word's Huffman bits are labels; logits are dots between the
        center embedding and the inner-node output embeddings on that path.
        """
        if center_index.ndim != 1 or target_index.ndim != 1:
            raise ValueError("center_index and target_index must be 1-D word-id tensors")
        if center_index.shape[0] != target_index.shape[0]:
            raise ValueError("center_index and target_index must have the same length")

        center_vectors = self.center_embeddings(center_index)
        nodes = self.path_nodes[target_index]
        codes = self.path_codes[target_index]
        mask = self.path_mask[target_index]
        node_vectors = self.node_embeddings(nodes.clamp(min=0))
        logits = (node_vectors * center_vectors.unsqueeze(1)).sum(dim=-1)
        per_node = F.binary_cross_entropy_with_logits(logits, codes, reduction="none")
        per_example = (per_node * mask.float()).sum(dim=1)
        return per_example.mean()
