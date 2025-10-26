#!/usr/bin/env ts-node
/**
 * Initialize the privacy pool and vault
 *
 * Usage:
 *   ts-node scripts/init_pool.ts [depth]
 *
 * Example:
 *   ts-node scripts/init_pool.ts 20
 *
 * Output (JSON):
 *   {"success": true, "depth": 20, "root": "abc...", "vault_tx": "...", "pool_tx": "..."}
 */

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, SystemProgram } from "@solana/web3.js";
// @ts-ignore - ESM import resolution
import { Incognito } from "../target/types/incognito.js";
// @ts-ignore - ESM import resolution
import { POOL_STATE_SEED, SOL_VAULT_SEED } from "./utils.js";

async function main() {
  const args = process.argv.slice(2);
  const depth = args[0] ? parseInt(args[0]) : 20;

  if (depth < 1 || depth > 32) {
    console.error("Error: depth must be between 1 and 32");
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

  let vaultTx = null;
  let poolTx = null;

  // Initialize vault if needed
  try {
    await program.account.solVault.fetch(solVaultPda);
    console.error("Vault already initialized");
  } catch {
    vaultTx = await program.methods
      .initVault()
      .accounts({
        payer: provider.publicKey!,
        solVault: solVaultPda,
        systemProgram: SystemProgram.programId,
      })
      .rpc({ commitment: "confirmed" });
  }

  // Initialize pool
  try {
    await program.account.poolState.fetch(poolStatePda);
    console.error(
      JSON.stringify({ success: false, error: "Pool already initialized" })
    );
    process.exit(1);
  } catch {
    poolTx = await program.methods
      .initPool(depth)
      .accounts({
        payer: provider.publicKey!,
        poolState: poolStatePda,
        systemProgram: SystemProgram.programId,
      })
      .rpc({ commitment: "confirmed" });
  }

  // Fetch to get root
  const poolState = await program.account.poolState.fetch(poolStatePda);

  const result = {
    success: true,
    depth,
    root: Buffer.from(poolState.root).toString("hex"),
    pool_address: poolStatePda.toBase58(),
    vault_address: solVaultPda.toBase58(),
    vault_tx: vaultTx,
    pool_tx: poolTx,
  };

  console.log(JSON.stringify(result, null, 2));
}

main().catch((e) => {
  console.error(JSON.stringify({ success: false, error: e.message }));
  process.exit(1);
});
