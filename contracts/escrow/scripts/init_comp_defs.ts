#!/usr/bin/env ts-node
/**
 * Initialize all Arcium computation definitions for the escrow program
 * This must be run once before using any MPC functions
 */

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, Keypair } from "@solana/web3.js";
import { Escrow } from "../target/types/escrow";
import { getMXEAccAddress, getCompDefAccAddress, getCompDefAccOffset } from "@arcium-hq/client";
import * as fs from "fs";

const ESCROW_PROGRAM_ID = new PublicKey("5QvQbnrL7fKpM5pCMS3zNqgTK8ALNkgHgRvgd49YF7v4");

function loadKeypair(path: string): Keypair {
  const secretKey = JSON.parse(fs.readFileSync(path, "utf8"));
  return Keypair.fromSecretKey(new Uint8Array(secretKey));
}

async function main() {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.Escrow as Program<Escrow>;

  const wrapperPath = process.env.WRAPPER_KEYPAIR || "./keys/wrapper.json";
  const payerKeypair = loadKeypair(wrapperPath);

  console.log("Initializing Arcium computation definitions...");

  const compDefs = [
    { name: "encrypt_shipping_address", method: "initEncryptShippingCompDef" },
    { name: "calculate_reputation_score", method: "initReputationCalcCompDef" },
    { name: "verify_escrow_amount", method: "initVerifyAmountCompDef" },
    { name: "calculate_seller_stake", method: "initCalcStakeCompDef" },
    { name: "calculate_platform_fee", method: "initCalcFeeCompDef" },
    { name: "calculate_refund_amount", method: "initCalcRefundCompDef" },
    { name: "calculate_completion_distribution", method: "initCalcCompletionCompDef" },
    { name: "calculate_buyer_dispute_win", method: "initCalcBuyerWinCompDef" },
    { name: "calculate_seller_dispute_win", method: "initCalcSellerWinCompDef" },
    { name: "verify_sufficient_balance", method: "initVerifyBalanceCompDef" },
    { name: "check_timeout", method: "initCheckTimeoutCompDef" },
  ];

  const results: any[] = [];

  // Get MXE account address
  const mxeAccount = getMXEAccAddress(program.programId);
  console.log("MXE Account:", mxeAccount.toBase58());

  for (const compDef of compDefs) {
    try {
      console.log(`\nInitializing ${compDef.name}...`);

      // Get the computation definition account address
      const compDefOffset = Buffer.from(getCompDefAccOffset(compDef.name)).readUInt32LE();
      const compDefAccount = getCompDefAccAddress(program.programId, compDefOffset);

      // @ts-ignore - method names are dynamic
      const tx = await program.methods[compDef.method]()
        .accounts({
          payer: payerKeypair.publicKey,
          mxeAccount: mxeAccount,
          compDefAccount: compDefAccount,
        })
        .signers([payerKeypair])
        .rpc();

      console.log(`✓ ${compDef.name} initialized: ${tx}`);
      results.push({
        name: compDef.name,
        success: true,
        tx,
      });
    } catch (error: any) {
      // Check if already initialized
      if (error.message?.includes("already in use") || error.message?.includes("custom program error: 0x0")) {
        console.log(`✓ ${compDef.name} already initialized`);
        results.push({
          name: compDef.name,
          success: true,
          alreadyInitialized: true,
        });
      } else {
        console.error(`✗ Failed to initialize ${compDef.name}:`, error.message);
        results.push({
          name: compDef.name,
          success: false,
          error: error.message,
        });
      }
    }
  }

  console.log("\n" + "=".repeat(60));
  console.log("SUMMARY");
  console.log("=".repeat(60));

  const successful = results.filter((r) => r.success);
  const failed = results.filter((r) => !r.success);

  console.log(`\nSuccessfully initialized: ${successful.length}/${results.length}`);

  if (failed.length > 0) {
    console.log(`\nFailed initializations:`);
    failed.forEach((r) => {
      console.log(`  - ${r.name}: ${r.error}`);
    });
  }

  console.log(JSON.stringify({
    success: failed.length === 0,
    initialized: successful.length,
    failed: failed.length,
    results,
  }, null, 2));
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
