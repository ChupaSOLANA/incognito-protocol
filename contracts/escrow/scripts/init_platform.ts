#!/usr/bin/env ts-node

// Simple initialization script that can be called from Python
// Returns JSON output

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, SystemProgram, Keypair } from "@solana/web3.js";
import { Escrow } from "../target/types/escrow";
import * as fs from "fs";

function loadKeypair(path: string): Keypair {
  const secretKey = JSON.parse(fs.readFileSync(path, "utf8"));
  return Keypair.fromSecretKey(new Uint8Array(secretKey));
}

async function main() {
  try {
    const provider = anchor.AnchorProvider.env();
    anchor.setProvider(provider);

    const program = anchor.workspace.Escrow as Program<Escrow>;

    // Load wrapper and treasury keypairs
    const wrapperPath = process.env.WRAPPER_KEYPAIR || "./keys/wrapper.json";
    const treasuryPath = process.env.TREASURY_KEYPAIR || "./keys/pool.json";

    const wrapper = loadKeypair(wrapperPath);
    const treasury = loadKeypair(treasuryPath);

    // Derive PDAs
    const [configPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("config")],
      program.programId
    );

    const [arbiterPoolPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("arbiter_pool")],
      program.programId
    );

    // Check if already initialized
    try {
      const existingConfig = await program.account.platformConfig.fetch(configPDA);
      console.log(JSON.stringify({
        success: true,
        already_initialized: true,
        config_pda: configPDA.toBase58(),
        treasury: existingConfig.treasury.toBase58(),
      }));
      return;
    } catch (e) {
      // Not initialized, proceed
    }

    // Initialize platform config
    const tx1 = await program.methods
      .initializePlatform(treasury.publicKey)
      .accounts({
        authority: wrapper.publicKey,
        config: configPDA,
        systemProgram: SystemProgram.programId,
      })
      .signers([wrapper])
      .rpc();

    // Initialize arbiter pool
    const tx2 = await program.methods
      .initializeArbiterPool()
      .accounts({
        authority: wrapper.publicKey,
        arbiterPool: arbiterPoolPDA,
        systemProgram: SystemProgram.programId,
      })
      .signers([wrapper])
      .rpc();

    // Add wrapper as arbiter
    const tx3 = await program.methods
      .addArbiter(wrapper.publicKey, new anchor.BN(1_000_000_000))
      .accounts({
        authority: wrapper.publicKey,
        arbiterPool: arbiterPoolPDA,
      })
      .signers([wrapper])
      .rpc();

    // Verify setup
    const config = await program.account.platformConfig.fetch(configPDA);
    const pool = await program.account.arbiterPool.fetch(arbiterPoolPDA);

    console.log(JSON.stringify({
      success: true,
      already_initialized: false,
      config_pda: configPDA.toBase58(),
      arbiter_pool_pda: arbiterPoolPDA.toBase58(),
      treasury: config.treasury.toBase58(),
      platform_fee_bps: config.platformFeeBps,
      seller_stake_bps: config.sellerStakeBps,
      arbiters: pool.arbiters.map(a => a.toBase58()),
      transactions: {
        platform_init: tx1,
        arbiter_pool_init: tx2,
        arbiter_add: tx3,
      },
    }));
  } catch (error) {
    console.error(JSON.stringify({
      success: false,
      error: error instanceof Error ? error.message : String(error),
    }));
    process.exit(1);
  }
}

main();
