#!/usr/bin/env ts-node
/**
 * Withdraw SOL from the privacy pool
 *
 * Usage:
 *   ts-node scripts/withdraw_from_pool.ts <amount_lamports> <commitment_hex> <nullifier_hex> <index> <merkle_path_json> [recipient_keypair_path] [change_commitment_hex] [change_nf_hash_hex] [change_merkle_path_json]
 *
 * Example:
 *   ts-node scripts/withdraw_from_pool.ts 1000000 abc123... def456... 0 '["path1","path2",...]' keys/user1.json
 *   ts-node scripts/withdraw_from_pool.ts 3000000 abc123... def456... 0 '["path1","path2",...]' keys/user1.json change_commitment change_nf_hash '["change_path1",...]'
 *
 * Output (JSON):
 *   {"success": true, "amount": "1000000", "tx": "sig..."}
 */

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { Keypair, PublicKey, SystemProgram } from "@solana/web3.js";
import { Incognito } from "../target/types/incognito";
import BN from "bn.js";
import {
  MerkleTree,
  computeNfHash,
  leafFrom,
  POOL_STATE_SEED,
  SOL_VAULT_SEED,
  NULLIFIER_SEED,
  COMMITMENT_SEED,
} from "./utils";
import * as fs from "fs";

async function main() {
  const args = process.argv.slice(2);

  // Check if we're using stdin mode
  const useStdin = args[0] === "--stdin";

  let amount: bigint;
  let commitment: Buffer;
  let nullifier: Buffer;
  let index: number;
  let merklePath: Buffer[];
  let recipientKeypair: Keypair | null = null;
  let changeCommitment: Buffer | null = null;
  let changeNfHash: Buffer | null = null;
  let changeMerklePath: Buffer[] | null = null;

  if (useStdin) {
    // Read JSON from stdin using a more robust method for large inputs
    let stdinData = '';
    const buffer = Buffer.alloc(1024); // Read in 1KB chunks
    let bytesRead;

    // Read all available data from stdin
    while ((bytesRead = fs.readSync(0, buffer, 0, buffer.length, null)) > 0) {
      stdinData += buffer.toString('utf-8', 0, bytesRead);
    }

    const input = JSON.parse(stdinData);

    amount = BigInt(input.amount_lamports);
    commitment = Buffer.from(input.commitment_hex, "hex");
    nullifier = Buffer.from(input.nullifier_hex, "hex");
    index = input.leaf_index;
    merklePath = input.merkle_path.map((p: string) => Buffer.from(p, "hex"));

    // Load recipient keypair
    if (input.recipient_keyfile) {
      const keypairData = JSON.parse(fs.readFileSync(input.recipient_keyfile, "utf-8"));
      recipientKeypair = Keypair.fromSecretKey(new Uint8Array(keypairData));
    }

    // Parse change note parameters
    if (input.change_commitment_hex && input.change_nf_hash_hex && input.change_merkle_path && input.change_merkle_path.length > 0) {
      changeCommitment = Buffer.from(input.change_commitment_hex, "hex");
      changeNfHash = Buffer.from(input.change_nf_hash_hex, "hex");
      changeMerklePath = input.change_merkle_path.map((p: string) => Buffer.from(p, "hex"));
    }
  } else {
    // Legacy: Parse from command-line arguments
    if (args.length < 5) {
      console.error(
        "Usage: ts-node withdraw_from_pool.ts <amount_lamports> <commitment_hex> <nullifier_hex> <index> <merkle_path_json> [recipient_keypair_path]"
      );
      process.exit(1);
    }

    amount = BigInt(args[0]);
    commitment = Buffer.from(args[1], "hex");
    nullifier = Buffer.from(args[2], "hex");
    index = parseInt(args[3]);
    merklePath = JSON.parse(args[4]).map((p: string) =>
      Buffer.from(p, "hex")
    );

    // Load recipient keypair if provided
    if (args[5]) {
      const keypairData = JSON.parse(fs.readFileSync(args[5], "utf-8"));
      recipientKeypair = Keypair.fromSecretKey(new Uint8Array(keypairData));
    }

    // Parse optional change note parameters
    const hasChangeCommitment = args[6] && args[6].trim().length > 0;
    const hasChangeNfHash = args[7] && args[7].trim().length > 0;
    const hasChangePath = args[8] && args[8].trim().length > 0;

    if (hasChangeCommitment && hasChangeNfHash && hasChangePath) {
      changeCommitment = Buffer.from(args[6], "hex");
      changeNfHash = Buffer.from(args[7], "hex");
      changeMerklePath = JSON.parse(args[8]).map((p: string) =>
        Buffer.from(p, "hex")
      );
    }
  }

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
  const solVaultPda = PublicKey.findProgramAddressSync(
    [SOL_VAULT_SEED],
    program.programId
  )[0];

  // Determine recipient public key
  const recipientPubkey = recipientKeypair
    ? recipientKeypair.publicKey
    : provider.publicKey!;

  const nfHash = computeNfHash(new Uint8Array(nullifier));
  const nullifierPda = PublicKey.findProgramAddressSync(
    [NULLIFIER_SEED, nullifier],
    program.programId
  )[0];

  // Determine change commitment marker PDA
  let changeCommitmentMarkerPda: PublicKey;
  if (changeCommitment) {
    changeCommitmentMarkerPda = PublicKey.findProgramAddressSync(
      [COMMITMENT_SEED, changeCommitment],
      program.programId
    )[0];
  } else {
    // Use pool state as dummy account when no change
    changeCommitmentMarkerPda = poolStatePda;
  }

  // Build transaction
  let txBuilder = program.methods
    .withdrawFromPool(
      new BN(amount.toString()),
      Array.from(commitment),
      merklePath.map((p) => Array.from(p)),
      Array.from(nullifier),
      new BN(index),
      Array.from(recipientPubkey.toBytes()),
      changeCommitment ? Array.from(changeCommitment) : null,
      changeNfHash ? Array.from(changeNfHash) : null,
      changeMerklePath ? changeMerklePath.map((p) => Array.from(p)) : null
    )
    .accounts({
      recipient: recipientPubkey,
      solVault: solVaultPda,
      poolState: poolStatePda,
      nullifierMarker: nullifierPda,
      changeCommitmentMarker: changeCommitmentMarkerPda,
      systemProgram: SystemProgram.programId,
    });

  // If using a specific keypair, sign with it
  if (recipientKeypair) {
    txBuilder = txBuilder.signers([recipientKeypair]);
  }

  // Execute transaction with retry on blockhash expiration
  let tx;
  let retries = 3;
  let lastError;

  for (let i = 0; i < retries; i++) {
    try {
      tx = await txBuilder.rpc({
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

  const result = {
    success: true,
    amount: amount.toString(),
    recipient: recipientPubkey.toBase58(),
    nullifier: nullifier.toString("hex"),
    tx,
  };

  console.log(JSON.stringify(result, null, 2));
}

main().catch((e) => {
  console.error(JSON.stringify({ success: false, error: e.message }));
  process.exit(1);
});
