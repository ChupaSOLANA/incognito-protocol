#!/usr/bin/env ts-node
/**
 * Partial withdrawal from privacy pool (creates change note)
 *
 * Usage:
 *   ts-node scripts/partial_withdraw.ts <withdraw_amount> <commitment_hex> <nullifier_hex> <index> <recipient_pubkey> <change_commitment_hex> <change_nullifier_hex>
 *
 * Example:
 *   ts-node scripts/partial_withdraw.ts 3000000000 abc... def... 0 YourPubkey... $(openssl rand -hex 32) $(openssl rand -hex 32)
 *
 * Output (JSON):
 *   {"success": true, "withdraw_amount": "3000000000", "change_index": 1, "change_commitment": "...", "tx": "sig..."}
 */

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, SystemProgram } from "@solana/web3.js";
import { Incognito } from "../target/types/incognito";
import { randomBytes } from "crypto";
import {
  MerkleTree,
  computeNfHash,
  leafFrom,
  POOL_STATE_SEED,
  SOL_VAULT_SEED,
  NULLIFIER_SEED,
  COMMITMENT_SEED,
} from "./utils";

async function main() {
  const args = process.argv.slice(2);

  if (args.length < 5) {
    console.error(
      "Usage: ts-node partial_withdraw.ts <withdraw_amount> <commitment_hex> <nullifier_hex> <index> <recipient_pubkey> [change_commitment_hex] [change_nullifier_hex]"
    );
    process.exit(1);
  }

  const withdrawAmount = BigInt(args[0]);
  const commitment = Buffer.from(args[1], "hex");
  const nullifier = Buffer.from(args[2], "hex");
  const index = parseInt(args[3]);
  const recipientPubkey = new PublicKey(args[4]);

  // Generate change note credentials if not provided
  const changeCommitment = args[5]
    ? Buffer.from(args[5], "hex")
    : randomBytes(32);
  const changeNullifier = args[6]
    ? Buffer.from(args[6], "hex")
    : randomBytes(32);

  if (
    commitment.length !== 32 ||
    nullifier.length !== 32 ||
    changeCommitment.length !== 32 ||
    changeNullifier.length !== 32
  ) {
    console.error("Error: all commitments and nullifiers must be 32 bytes");
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

  // Fetch pool state
  const poolState = await program.account.poolState.fetch(poolStatePda);
  const depth = poolState.depth;
  const leafCount = Number(poolState.leafCount.toString());

  // Build merkle tree
  const tree = new MerkleTree(depth);

  // Get path for original note
  const withdrawPath = tree.getMerklePath(index);

  // Compute change note details
  const changeNfHash = computeNfHash(new Uint8Array(changeNullifier));
  const changeIndex = leafCount; // Next available index
  const changePath = tree.getMerklePath(changeIndex);

  const nullifierPda = PublicKey.findProgramAddressSync(
    [NULLIFIER_SEED, nullifier],
    program.programId
  )[0];

  const changeCommitmentMarkerPda = PublicKey.findProgramAddressSync(
    [COMMITMENT_SEED, changeCommitment],
    program.programId
  )[0];

  const tx = await program.methods
    .withdrawFromPool(
      new anchor.BN(withdrawAmount.toString()),
      Array.from(commitment),
      withdrawPath.map((p) => Array.from(p)),
      Array.from(nullifier),
      new anchor.BN(index),
      Array.from(recipientPubkey.toBytes()),
      Array.from(changeCommitment),
      Array.from(changeNfHash),
      changePath.map((p) => Array.from(p))
    )
    .accounts({
      recipient: recipientPubkey,
      solVault: solVaultPda,
      poolState: poolStatePda,
      nullifierMarker: nullifierPda,
      changeCommitmentMarker: changeCommitmentMarkerPda,
      systemProgram: SystemProgram.programId,
    })
    .rpc({ commitment: "confirmed" });

  const result = {
    success: true,
    withdraw_amount: withdrawAmount.toString(),
    recipient: recipientPubkey.toBase58(),
    nullifier: nullifier.toString("hex"),
    change_index: changeIndex,
    change_commitment: changeCommitment.toString("hex"),
    change_nullifier: changeNullifier.toString("hex"),
    change_nf_hash: Buffer.from(changeNfHash).toString("hex"),
    tx,
  };

  console.log(JSON.stringify(result, null, 2));
}

main().catch((e) => {
  console.error(JSON.stringify({ success: false, error: e.message }));
  process.exit(1);
});
