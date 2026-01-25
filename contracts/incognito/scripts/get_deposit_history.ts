#!/usr/bin/env ts-node
/**
 * Query all transactions to rebuild Merkle tree leaves
 *
 * This script queries the blockchain for all depositToPool and addClaimNote transactions
 * and extracts the commitment and nf_hash from each to rebuild the complete tree.
 * This includes both deposits and claim notes (change notes, seller payment notes, etc.)
 */

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, Connection } from "@solana/web3.js";
import { Incognito } from "../target/types/incognito";
import { POOL_STATE_SEED } from "./utils";

async function main() {
  anchor.setProvider(anchor.AnchorProvider.env());
  const provider = anchor.getProvider() as anchor.AnchorProvider;
  const program = anchor.workspace.Incognito as Program<Incognito>;
  const connection = provider.connection;

  const poolStatePda = PublicKey.findProgramAddressSync(
    [POOL_STATE_SEED],
    program.programId
  )[0];

  try {
    // Get all transaction signatures for the pool state account
    console.error("Querying transaction signatures for pool state account...");

    const signatures = await connection.getSignaturesForAddress(
      poolStatePda,
      { limit: 1000 },
      "confirmed"
    );

    console.error(`Found ${signatures.length} transactions`);

    const notes: Array<{
      signature: string;
      commitment: string;
      nf_hash: string;
      timestamp: number;
      type: 'deposit' | 'change';
    }> = [];

    // Process each transaction in INSERTION ORDER (oldest to newest)
    // getSignaturesForAddress returns newest first, so reverse to get oldest first
    // This order MUST match the on-chain tree insertion order for merkle paths to work!
    for (const sigInfo of signatures.reverse()) {
      const tx = await connection.getTransaction(sigInfo.signature, {
        maxSupportedTransactionVersion: 0,
        commitment: "confirmed",
      });

      if (!tx || !tx.meta || tx.meta.err) {
        continue;
      }

      // Check if this is a deposit transaction by looking for the deposit instruction
      const message = tx.transaction.message;
      const accountKeys = message.getAccountKeys();

      // Find instructions that call our program
      for (let i = 0; i < message.compiledInstructions.length; i++) {
        const ix = message.compiledInstructions[i];
        const programId = accountKeys.get(ix.programIdIndex);

        if (programId?.equals(program.programId)) {
          // Parse the instruction data
          const data = Buffer.from(ix.data);

          try {
            // Both instructions have merkle paths, so length doesn't help
            // deposit_to_pool: [8 disc][8 amount][32 commitment][32 nf_hash][path...]
            // add_claim_note:  [8 disc][32 commitment][32 nf_hash][path...]

            // Check discriminator to definitively identify instruction type
            // Discriminators are first 8 bytes of SHA-256("global:<function_name>")
            const DEPOSIT_DISCRIMINATOR = Buffer.from("63880f4255921859", "hex");
            const ADD_CLAIM_NOTE_DISCRIMINATOR = Buffer.from("df550ffb16f54875", "hex");
            const WITHDRAW_DISCRIMINATOR = Buffer.from("3e21805128ea1d4d", "hex");

            const discriminator = data.slice(0, 8);

            if (discriminator.equals(DEPOSIT_DISCRIMINATOR) && data.length >= 80) {
              // deposit_to_pool instruction
              const amount = data.readBigUInt64LE(8);
              const commitment = data.slice(16, 48).toString("hex");
              const nf_hash = data.slice(48, 80).toString("hex");

              notes.push({
                signature: sigInfo.signature,
                commitment,
                nf_hash,
                timestamp: sigInfo.blockTime || 0,
                type: 'deposit',
              });

              console.error(`Found deposit: ${commitment.slice(0, 16)}... (${amount} lamports)`);
            } else if (discriminator.equals(ADD_CLAIM_NOTE_DISCRIMINATOR) && data.length >= 72) {
              // add_claim_note instruction
              const commitment = data.slice(8, 40).toString("hex");
              const nf_hash = data.slice(40, 72).toString("hex");

              notes.push({
                signature: sigInfo.signature,
                commitment,
                nf_hash,
                timestamp: sigInfo.blockTime || 0,
                type: 'change',
              });

              console.error(`Found claim note: ${commitment.slice(0, 16)}...`);
            } else if (discriminator.equals(WITHDRAW_DISCRIMINATOR)) {
              // withdraw_from_pool instruction parsing
              // Layout:
              // - 8 disc
              // - 8 amount
              // - 32 commitment
              // - 4 + len*32 merkle_path
              // - 32 nullifier
              // - 8 index
              // - 32 recipient
              // - 1 + 32 (optional) change_commitment
              // - 1 + 32 (optional) change_nf_hash

              let offset = 8; // Skip discriminator
              offset += 8; // Skip amount
              offset += 32; // Skip commitment

              // Parse merkle_path len
              if (offset + 4 > data.length) continue;
              const pathLen = data.readUInt32LE(offset);
              offset += 4;
              offset += pathLen * 32;

              offset += 32; // Skip nullifier
              offset += 8;  // Skip index
              offset += 32; // Skip recipient

              // change_commitment (Option)
              if (offset + 1 > data.length) continue;
              const hasChangeCommitment = data.readUInt8(offset);
              offset += 1;

              let changeCommitment: string | null = null;
              if (hasChangeCommitment) {
                if (offset + 32 > data.length) continue;
                changeCommitment = data.slice(offset, offset + 32).toString('hex');
                offset += 32;
              }

              // change_nf_hash (Option)
              if (offset + 1 > data.length) continue;
              const hasChangeNfHash = data.readUInt8(offset);
              offset += 1;

              let changeNfHash: string | null = null;
              if (hasChangeNfHash) {
                if (offset + 32 > data.length) continue;
                changeNfHash = data.slice(offset, offset + 32).toString('hex');
                offset += 32;
              }

              if (changeCommitment && changeNfHash) {
                notes.push({
                  signature: sigInfo.signature,
                  commitment: changeCommitment,
                  nf_hash: changeNfHash,
                  timestamp: sigInfo.blockTime || 0,
                  type: 'change'
                });
                console.error(`Found change note from withdraw: ${changeCommitment.slice(0, 16)}...`);
              }
            }
          } catch (e) {
            // Not a recognized instruction, skip
          }
        }
      }

      // Also check for ChangeNoteCreated events in logs
      if (tx.meta?.logMessages) {
        for (const log of tx.meta.logMessages) {
          // Look for event logs (format: "Program log: ChangeNoteCreated")
          if (log.includes("ChangeNoteCreated")) {
            // The event data comes in the next log line usually
            // For simplicity, we'll parse from the program data in the transaction
            // Actually, let's look for the event in the inner instructions
          }
        }
      }
    }

    // Output all notes in INSERTION ORDER (blockchain order)
    // CRITICAL: Do NOT sort by timestamp - it doesn't preserve on-chain insertion order
    // The notes are already in the correct order from processing signatures.reverse()
    // Sorting by timestamp causes merkle path mismatches during withdrawals!
    console.error(`Total notes found: ${notes.length} (deposits + changes)`);

    console.log(JSON.stringify(notes, null, 2));

  } catch (e: any) {
    console.error(JSON.stringify({ success: false, error: e.message }));
    process.exit(1);
  }
}

main().catch((e) => {
  console.error(JSON.stringify({ success: false, error: e.message }));
  process.exit(1);
});
