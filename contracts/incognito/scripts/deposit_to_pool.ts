#!/usr/bin/env ts-node
/**
 * Deposit SOL to the privacy pool
 *
 * Usage:
 *   ts-node scripts/deposit_to_pool.ts <amount_lamports> <wrapper_stealth_address> [commitment_hex] [nf_hash_hex] [existing_leaves_csv]
 *
 * Example:
 *   ts-node scripts/deposit_to_pool.ts 100000000 WrapperAddr123... $(openssl rand -hex 32) $(openssl rand -hex 32) "leaf1,leaf2,leaf3"
 *
 * Output (JSON):
 *   {"index": 0, "commitment": "abc...", "nf_hash": "123...", "tx": "sig...", "wrapper_stealth": "..."}
 */

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, SystemProgram } from "@solana/web3.js";
import { Incognito } from "../target/types/incognito";
import { randomBytes } from "crypto";
import BN from "bn.js";
import {
  MerkleTree,
  computeNfHash,
  leafFrom,
  POOL_STATE_SEED,
  SOL_VAULT_SEED,
  COMMITMENT_SEED,
} from "./utils";

async function main() {
  const args = process.argv.slice(2);

  if (args.length < 2) {
    console.error("Usage: ts-node deposit_to_pool.ts <amount_lamports> <wrapper_stealth_address> [commitment_hex] [nf_hash_hex] [existing_leaves_csv]");
    process.exit(1);
  }

  const lamports = BigInt(args[0]);
  const wrapperStealthAddress = new PublicKey(args[1]);
  const commitment = args[2]
    ? Buffer.from(args[2], "hex")
    : randomBytes(32);

  // arg[3] is now nf_hash directly (NOT nullifier!)
  // Python pre-computes nf_hash = h1(nullifier) and passes it here
  const nfHash = args[3]
    ? Buffer.from(args[3], "hex")
    : computeNfHash(randomBytes(32));  // If not provided, generate random

  // arg[4] is comma-separated existing leaves (hex strings)
  const existingLeavesHex = args[4] || "";

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
  const solVaultPda = PublicKey.findProgramAddressSync(
    [SOL_VAULT_SEED],
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
        .depositToPool(
          new BN(lamports.toString()),
          Array.from(commitment),
          Array.from(nfHash),
          path.map((p) => Array.from(p))
        )
        .accounts({
          depositor: provider.publicKey!,
          solVault: solVaultPda,
          poolState: poolStatePda,
          commitmentMarker: commitmentMarkerPda,
          wrapperStealthAddress: wrapperStealthAddress,
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
    amount_lamports: lamports.toString(),
    wrapper_stealth_address: wrapperStealthAddress.toBase58(),
    tx,
  };

  console.log(JSON.stringify(result, null, 2));
}

main().catch((e) => {
  console.error(JSON.stringify({ success: false, error: e.message }));
  process.exit(1);
});
