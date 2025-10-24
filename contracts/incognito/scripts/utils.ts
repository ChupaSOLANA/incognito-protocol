import { createHash } from "crypto";

export type Bytes16 = Uint8Array;
export type Bytes32 = Uint8Array;

export function h1(x: Bytes32): Bytes32 {
  return new Uint8Array(createHash("sha256").update(Buffer.from(x)).digest());
}

export function h2(a: Bytes32, b: Bytes32): Bytes32 {
  const hash = createHash("sha256");
  hash.update(Buffer.from(a));
  hash.update(Buffer.from(b));
  return new Uint8Array(hash.digest());
}

export function computeNfHash(nullifier: Bytes32): Bytes32 {
  return h1(nullifier);
}

export function leafFrom(commitment: Bytes32, nfHash: Bytes32): Bytes32 {
  return h2(commitment, nfHash);
}

export class MerkleTree {
  private leaves: Bytes32[] = [];

  constructor(private depth: number) {}

  private zeroAt(level: number): Bytes32 {
    let z = new Uint8Array(32);
    for (let i = 0; i < level; i++) {
      z = h2(z, z);
    }
    return z;
  }

  addLeaf(leaf: Bytes32): number {
    const index = this.leaves.length;
    this.leaves.push(leaf);
    return index;
  }

  getLeaves(): Bytes32[] {
    return [...this.leaves];
  }

  getMerklePath(targetIndex: number): Bytes32[] {
    const path: Bytes32[] = [];
    let currentLevel = [...this.leaves];
    let idx = targetIndex;

    for (let level = 0; level < this.depth; level++) {
      const zero = this.zeroAt(level);
      const siblingIndex = (idx & 1) === 0 ? idx + 1 : idx - 1;
      const sibling =
        siblingIndex >= 0 && siblingIndex < currentLevel.length
          ? currentLevel[siblingIndex]
          : zero;
      path.push(sibling);

      const nextLevel: Bytes32[] = [];
      for (let i = 0; i < currentLevel.length; i += 2) {
        const left = currentLevel[i];
        const right = i + 1 < currentLevel.length ? currentLevel[i + 1] : zero;
        nextLevel.push(h2(left, right));
      }
      if (nextLevel.length === 0) nextLevel.push(h2(zero, zero));

      currentLevel = nextLevel;
      idx >>= 1;
    }

    return path;
  }
}

export const POOL_STATE_SEED = Buffer.from("pool_state");
export const SOL_VAULT_SEED = Buffer.from("sol_vault");
export const NULLIFIER_SEED = Buffer.from("nf");
export const COMMITMENT_SEED = Buffer.from("commitment");
