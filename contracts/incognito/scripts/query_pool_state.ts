#!/usr/bin/env ts-node
/**
 * Query the current pool state (merkle root, leaf count, depth)
 *
 * Usage:
 *   ts-node scripts/query_pool_state.ts
 *
 * Output (JSON):
 *   {"root": "abc...", "depth": 20, "leaf_count": 5}
 */

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey } from "@solana/web3.js";
import { Incognito } from "../target/types/incognito";
import { POOL_STATE_SEED, SOL_VAULT_SEED } from "./utils";

async function main() {
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

  try {
    const poolState = await program.account.poolState.fetch(poolStatePda);
    const solVault = await program.account.solVault.fetch(solVaultPda);
    const vaultBalance = await provider.connection.getBalance(solVaultPda);

    const result = {
      success: true,
      pool: {
        address: poolStatePda.toBase58(),
        root: Buffer.from(poolState.root).toString("hex"),
        depth: poolState.depth,
        leaf_count: poolState.leafCount.toString(),
      },
      vault: {
        address: solVaultPda.toBase58(),
        total_deposited: solVault.totalDeposited.toString(),
        current_balance: vaultBalance,
      },
    };

    console.log(JSON.stringify(result, null, 2));
  } catch (e: any) {
    if (e.message.includes("Account does not exist")) {
      console.log(
        JSON.stringify({
          success: false,
          error: "Pool not initialized. Run init_pool.ts first.",
        })
      );
    } else {
      throw e;
    }
  }
}

main().catch((e) => {
  console.error(JSON.stringify({ success: false, error: e.message }));
  process.exit(1);
});
