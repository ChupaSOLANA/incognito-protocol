from __future__ import annotations
from typing import List, Tuple, Optional
from .commitments import sha256

class MerkleTree:
    def __init__(self, leaves_hex: Optional[List[str]] = None):
        self.leaf_bytes: List[bytes] = [bytes.fromhex(h) for h in (leaves_hex or [])]
        self.layers: List[List[bytes]] = []
        if self.leaf_bytes:
            self.build_tree()

    def build_tree(self) -> None:
        layers = [self.leaf_bytes[:]]
        while len(layers[-1]) > 1:
            cur = layers[-1]; nxt = []
            for i in range(0, len(cur), 2):
                left = cur[i]; right = cur[i+1] if i+1 < len(cur) else left
                nxt.append(sha256(left + right))
            layers.append(nxt)
        self.layers = layers

    def root(self) -> bytes:
        if not self.leaf_bytes:
            return sha256(b"")
        return self.layers[-1][0]

    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        if index < 0 or index >= len(self.leaf_bytes):
            raise IndexError("Leaf index out of range.")
        proof: List[Tuple[str, str]] = []
        idx = index
        for layer in self.layers[:-1]:
            is_right = (idx % 2 == 1)
            sibling_idx = idx - 1 if is_right else idx + 1
            sibling = layer[idx] if sibling_idx >= len(layer) else layer[sibling_idx]
            direction = "left" if is_right else "right"
            proof.append((sibling.hex(), direction))
            idx //= 2
        return proof

def verify_merkle(leaf_hex: str, proof: List[Tuple[str, str]], root_hex: str) -> bool:
    h = bytes.fromhex(leaf_hex)
    for sib_hex, direction in proof:
        sib = bytes.fromhex(sib_hex)
        h = sha256(sib + h) if direction == "left" else sha256(h + sib)
    return h.hex() == root_hex
