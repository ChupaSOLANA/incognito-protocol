#!/usr/bin/env ts-node

// Setup: Initialize platform and arbiter pool

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { SystemProgram } from "@solana/web3.js";
import { Escrow } from "../target/types/escrow";
import { initializeTestAccounts, outputResult } from "./common";

async function main() {
  try {
    // Configure the client
    anchor.setProvider(anchor.AnchorProvider.env());
    const program = anchor.workspace.Escrow as Program<Escrow>;
    const provider = anchor.getProvider() as anchor.AnchorProvider;

    // Initialize test accounts
    const accounts = await initializeTestAccounts(program, provider);

    // Initialize platform config
    const tx1 = await program.methods
      .initializePlatform(accounts.treasury.publicKey)
      .accounts({
        authority: accounts.owner.publicKey,
        config: accounts.configPDA,
        systemProgram: SystemProgram.programId,
      })
      .signers([accounts.owner])
      .rpc();

    console.log("Platform initialized:", tx1);

    // Initialize arbiter pool
    const tx2 = await program.methods
      .initializeArbiterPool()
      .accounts({
        authority: accounts.owner.publicKey,
        arbiterPool: accounts.arbiterPoolPDA,
        systemProgram: SystemProgram.programId,
      })
      .signers([accounts.owner])
      .rpc();

    console.log("Arbiter pool initialized:", tx2);

    // Add arbiter to pool
    const tx3 = await program.methods
      .addArbiter(accounts.arbiter.publicKey, new anchor.BN(1_000_000_000))
      .accounts({
        authority: accounts.owner.publicKey,
        arbiterPool: accounts.arbiterPoolPDA,
      })
      .signers([accounts.owner])
      .rpc();

    console.log("Arbiter added:", tx3);

    // Verify setup
    const config = await program.account.platformConfig.fetch(accounts.configPDA);
    const pool = await program.account.arbiterPool.fetch(accounts.arbiterPoolPDA);

    outputResult(true, {
      transactions: {
        platform_init: tx1,
        arbiter_pool_init: tx2,
        arbiter_add: tx3,
      },
      config: {
        treasury: config.treasury.toBase58(),
        platformFeeBps: config.platformFeeBps,
        sellerStakeBps: config.sellerStakeBps,
      },
      arbiter_pool: {
        arbiters: pool.arbiters.map((a) => a.toBase58()),
      },
      accounts: {
        buyer: accounts.buyer.publicKey.toBase58(),
        seller: accounts.seller.publicKey.toBase58(),
        arbiter: accounts.arbiter.publicKey.toBase58(),
        treasury: accounts.treasury.publicKey.toBase58(),
      },
    });
  } catch (error) {
    console.error("Error:", error);
    outputResult(false, { error: error.message });
  }
}

main();
