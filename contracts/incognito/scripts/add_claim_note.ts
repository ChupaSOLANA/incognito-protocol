#!/usr/bin/env ts-node
/**
 * Add a claim note to the privacy pool (without depositing SOL)
 * Used for escrow claim flows where seller/buyer claims their note
 *
 * Usage:
 *   ts-node scripts/add_claim_note.ts <commitment_hex> <nf_hash_hex> [existing_leaves_csv]
 *
 * Example:
 *   ts-node scripts/add_claim_note.ts $(openssl rand -hex 32) $(openssl rand -hex 32) "leaf1,leaf2,leaf3"
 *
 * Output (JSON):
 *   {"success": true, "index": 0, "commitment": "abc...", "nf_hash": "123...", "tx": "sig..."}
 */

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, SystemProgram } from "@solana/web3.js";
import { Incognito } from "../target/types/incognito";
import {
  MerkleTree,
  leafFrom,
  POOL_STATE_SEED,
  COMMITMENT_SEED,
} from "./utils";

async function main() {
  const args = process.argv.slice(2);

  if (args.length < 2) {
    console.error("Usage: ts-node add_claim_note.ts <commitment_hex> <nf_hash_hex> [existing_leaves_csv]");
    process.exit(1);
  }

  const commitment = Buffer.from(args[0], "hex");
  const nfHash = Buffer.from(args[1], "hex");
  const existingLeavesHex = args[2] || "";

  if (commitment.length !== 32 || nfHash.length !== 32) {
    console.error("Error: commitment and nf_hash must be 32 bytes");
    process.exit(1);
  }

  anchor.setProvider(anchor.AnchorProvider.env());
  const provider = anchor.getProvider() as anchor.AnchorProvider;
  const program = anchor.workspace.Incognito as Program<Incognito>;

  const poolStatePda = PublicKey.findProgramAddressSync(
    [POOL_STATE_SEED],
    program.programId
  )[0];

  // Fetch current pool state to get depth
  const poolState = await program.account.poolState.fetch(poolStatePda);
  const depth = poolState.depth;
  const leafCount = Number(poolState.leafCount.toString());

  // Build merkle tree locally from existing leaves
  const tree = new MerkleTree(depth);

  // Add existing leaves to reconstruct current tree state
  if (existingLeavesHex) {
    const existingLeaves = existingLeavesHex.split(',').filter(l => l.length > 0);
    for (const leafHex of existingLeaves) {
      tree.addLeaf(Buffer.from(leafHex, "hex"));
    }
  }

  // Verify our local tree has same leaf count as on-chain
  if (tree.getLeaves().length !== leafCount) {
    console.error(JSON.stringify({
      success: false,
      error: `Tree sync error: local has ${tree.getLeaves().length} leaves, on-chain has ${leafCount}`
    }));
    process.exit(1);
  }

  // Compute leaf from commitment and nf_hash
  const leaf = leafFrom(new Uint8Array(commitment), nfHash);

  // Get path for next insertion
  const index = leafCount;
  const path = tree.getMerklePath(index);

  const commitmentMarkerPda = PublicKey.findProgramAddressSync(
    [COMMITMENT_SEED, commitment],
    program.programId
  )[0];

  // Execute transaction with retry on blockhash expiration
  let tx;
  let retries = 3;
  let lastError;

  for (let i = 0; i < retries; i++) {
    try {
      tx = await program.methods
        .addClaimNote(
          Array.from(commitment),
          Array.from(nfHash),
          path.map((p) => Array.from(p))
        )
        .accounts({
          payer: provider.publicKey!,
          poolState: poolStatePda,
          commitmentMarker: commitmentMarkerPda,
          systemProgram: SystemProgram.programId,
        })
        .rpc({
          commitment: "confirmed",
          skipPreflight: false,
          preflightCommitment: "confirmed",
        });
      break; // Success, exit retry loop
    } catch (e: any) {
      lastError = e;
      if (i < retries - 1 && e.message?.includes("Blockhash not found")) {
        // Wait a bit and retry with fresh blockhash
        await new Promise((resolve) => setTimeout(resolve, 1000));
        continue;
      }
      throw e; // Re-throw if not blockhash error or last retry
    }
  }

  if (!tx) {
    throw lastError || new Error("Transaction failed after retries");
  }

  // Output as JSON for easy parsing
  const result = {
    success: true,
    index,
    commitment: commitment.toString("hex"),
    nf_hash: Buffer.from(nfHash).toString("hex"),
    tx,
  };

  console.log(JSON.stringify(result, null, 2));
}

main().catch((e) => {
  console.error(JSON.stringify({ success: false, error: e.message }));
  process.exit(1);
});
