#!/usr/bin/env ts-node

// Initialize escrow platform for marketplace use

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, SystemProgram } from "@solana/web3.js";
import { Escrow } from "../target/types/escrow";
import { readKpJson } from "./common";
import * as path from "path";

async function main() {
  try {
    // Configure the client
    anchor.setProvider(anchor.AnchorProvider.env());
    const program = anchor.workspace.Escrow as Program<Escrow>;
    const provider = anchor.getProvider() as anchor.AnchorProvider;

    // Load marketplace accounts
    const repoRoot = path.join(__dirname, "../../..");
    const treasuryPath = process.env.TREASURY_KEYPAIR || path.join(repoRoot, "keys/pool.json");
    const wrapperPath = process.env.WRAPPER_KEYPAIR || path.join(repoRoot, "keys/wrapper.json");

    const treasury = readKpJson(treasuryPath);
    const wrapper = readKpJson(wrapperPath);

    console.log("Treasury:", treasury.publicKey.toBase58());
    console.log("Wrapper (authority):", wrapper.publicKey.toBase58());

    // Derive PDAs
    const [configPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("config")],
      program.programId
    );

    const [arbiterPoolPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("arbiter_pool")],
      program.programId
    );

    console.log("Config PDA:", configPDA.toBase58());
    console.log("Arbiter Pool PDA:", arbiterPoolPDA.toBase58());

    // Check if already initialized
    try {
      const existingConfig = await program.account.platformConfig.fetch(configPDA);
      console.log("✅ Platform already initialized!");
      console.log("Existing treasury:", existingConfig.treasury.toBase58());
      console.log("Platform fee:", existingConfig.platformFeeBps, "bps");
      console.log("Seller stake:", existingConfig.sellerStakeBps, "bps");

      try {
        const existingPool = await program.account.arbiterPool.fetch(arbiterPoolPDA);
        console.log("✅ Arbiter pool already initialized!");
        console.log("Arbiters:", existingPool.arbiters.map(a => a.toBase58()));
      } catch (e) {
        console.log("❌ Arbiter pool not initialized");
      }

      process.exit(0);
    } catch (e) {
      console.log("Platform not yet initialized, proceeding with setup...");
    }

    // Initialize platform config
    console.log("\nInitializing platform config...");
    const tx1 = await program.methods
      .initializePlatform(treasury.publicKey)
      .accounts({
        authority: wrapper.publicKey,
        config: configPDA,
        systemProgram: SystemProgram.programId,
      })
      .signers([wrapper])
      .rpc();

    console.log("✅ Platform initialized:", tx1);

    // Initialize arbiter pool
    console.log("\nInitializing arbiter pool...");
    const tx2 = await program.methods
      .initializeArbiterPool()
      .accounts({
        authority: wrapper.publicKey,
        arbiterPool: arbiterPoolPDA,
        systemProgram: SystemProgram.programId,
      })
      .signers([wrapper])
      .rpc();

    console.log("✅ Arbiter pool initialized:", tx2);

    // Add wrapper as arbiter (for now)
    console.log("\nAdding wrapper as arbiter...");
    const tx3 = await program.methods
      .addArbiter(wrapper.publicKey, new anchor.BN(1_000_000_000))
      .accounts({
        authority: wrapper.publicKey,
        arbiterPool: arbiterPoolPDA,
      })
      .signers([wrapper])
      .rpc();

    console.log("✅ Arbiter added:", tx3);

    // Verify setup
    const config = await program.account.platformConfig.fetch(configPDA);
    const pool = await program.account.arbiterPool.fetch(arbiterPoolPDA);

    console.log("\n=== SETUP COMPLETE ===");
    console.log("Config:");
    console.log("  Treasury:", config.treasury.toBase58());
    console.log("  Platform fee:", config.platformFeeBps, "bps");
    console.log("  Seller stake:", config.sellerStakeBps, "bps");
    console.log("\nArbiter Pool:");
    console.log("  Arbiters:", pool.arbiters.map(a => a.toBase58()));
    console.log("\nTransactions:");
    console.log("  Platform init:", tx1);
    console.log("  Pool init:", tx2);
    console.log("  Arbiter add:", tx3);

    process.exit(0);
  } catch (error) {
    console.error("\n❌ Error:", error);
    process.exit(1);
  }
}

main();
