#!/usr/bin/env ts-node
/**
 * Verify a Merkle proof (creates nullifier marker to prevent double-spend)
 *
 * Usage:
 *   ts-node scripts/verify_proof.ts <commitment_hex> <nullifier_hex> <index>
 *
 * Example:
 *   ts-node scripts/verify_proof.ts abc123... def456... 0
 *
 * Output (JSON):
 *   {"success": true, "commitment": "...", "nullifier": "...", "index": 0, "tx": "sig..."}
 */

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, SystemProgram } from "@solana/web3.js";
import { Incognito } from "../target/types/incognito";
import {
  MerkleTree,
  POOL_STATE_SEED,
  NULLIFIER_SEED,
} from "./utils";

async function main() {
  const args = process.argv.slice(2);

  if (args.length < 3) {
    console.error(
      "Usage: ts-node verify_proof.ts <commitment_hex> <nullifier_hex> <index>"
    );
    process.exit(1);
  }

  const commitment = Buffer.from(args[0], "hex");
  const nullifier = Buffer.from(args[1], "hex");
  const index = parseInt(args[2]);

  if (commitment.length !== 32 || nullifier.length !== 32) {
    console.error("Error: commitment and nullifier must be 32 bytes");
    process.exit(1);
  }

  anchor.setProvider(anchor.AnchorProvider.env());
  const provider = anchor.getProvider() as anchor.AnchorProvider;
  const program = anchor.workspace.Incognito as Program<Incognito>;

  const poolStatePda = PublicKey.findProgramAddressSync(
    [POOL_STATE_SEED],
    program.programId
  )[0];

  // Fetch pool state
  const poolState = await program.account.poolState.fetch(poolStatePda);
  const depth = poolState.depth;

  // Build merkle tree and get path
  // NOTE: In production, you'd reconstruct from on-chain events
  const tree = new MerkleTree(depth);
  const path = tree.getMerklePath(index);

  const nullifierPda = PublicKey.findProgramAddressSync(
    [NULLIFIER_SEED, nullifier],
    program.programId
  )[0];

  const tx = await program.methods
    .verifyProof(
      Array.from(commitment),
      path.map((p) => Array.from(p)),
      Array.from(nullifier),
      new anchor.BN(index)
    )
    .accounts({
      payer: provider.publicKey!,
      poolState: poolStatePda,
      nullifierMarker: nullifierPda,
      systemProgram: SystemProgram.programId,
    })
    .rpc({ commitment: "confirmed" });

  const result = {
    success: true,
    commitment: commitment.toString("hex"),
    nullifier: nullifier.toString("hex"),
    index,
    tx,
  };

  console.log(JSON.stringify(result, null, 2));
}

main().catch((e) => {
  console.error(JSON.stringify({ success: false, error: e.message }));
  process.exit(1);
});
