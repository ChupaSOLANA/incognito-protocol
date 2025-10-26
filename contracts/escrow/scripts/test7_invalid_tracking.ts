#!/usr/bin/env ts-node

// Test 7: Invalid tracking number - Verify tracking number validation

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, SystemProgram } from "@solana/web3.js";
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

    const orderId = new anchor.BN(7);
    const amount = new anchor.BN(300_000_000);
    const encryptedShipping = Buffer.from("encrypted_data");
    const shippingNonce = Buffer.from([
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    ]);

    const [escrowPDA] = PublicKey.findProgramAddressSync(
      [
        Buffer.from("escrow"),
        accounts.buyer.publicKey.toBuffer(),
        orderId.toArrayLike(Buffer, "le", 8),
      ],
      program.programId
    );

    const [buyerRepPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("reputation"), accounts.buyer.publicKey.toBuffer()],
      program.programId
    );

    const [sellerRepPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("reputation"), accounts.seller.publicKey.toBuffer()],
      program.programId
    );

    const tx1 = await program.methods
      .createOrder(
        amount,
        orderId,
        encryptedShipping,
        Array.from(shippingNonce),
        null,
        false
      )
      .accounts({
        buyer: accounts.buyer.publicKey,
        escrow: escrowPDA,
        buyerReputation: buyerRepPDA,
        config: accounts.configPDA,
        arbiterPool: accounts.arbiterPoolPDA,
        systemProgram: SystemProgram.programId,
      })
      .signers([accounts.buyer])
      .rpc();

    const tx2 = await program.methods
      .acceptOrder()
      .accounts({
        seller: accounts.seller.publicKey,
        escrow: escrowPDA,
        sellerReputation: sellerRepPDA,
        config: accounts.configPDA,
        systemProgram: SystemProgram.programId,
      })
      .signers([accounts.seller])
      .rpc();

    const testResults = [];

    // Test 1: Try to ship with empty tracking
    try {
      await program.methods
        .markShipped("")
        .accounts({
          seller: accounts.seller.publicKey,
          escrow: escrowPDA,
          config: accounts.configPDA,
        })
        .signers([accounts.seller])
        .rpc();

      testResults.push({
        test: "empty_tracking",
        passed: false,
        note: "Should have failed - empty tracking number",
      });
    } catch (err) {
      const errorStr = err.toString();
      if (errorStr.includes("InvalidTracking")) {
        console.log("✅ Correctly rejected empty tracking number");
        testResults.push({
          test: "empty_tracking",
          passed: true,
          error: "InvalidTracking",
        });
      } else {
        testResults.push({
          test: "empty_tracking",
          passed: false,
          error: errorStr,
        });
      }
    }

    // Test 2: Try with too long tracking (>64 chars)
    try {
      const longTracking = "A".repeat(65);
      await program.methods
        .markShipped(longTracking)
        .accounts({
          seller: accounts.seller.publicKey,
          escrow: escrowPDA,
          config: accounts.configPDA,
        })
        .signers([accounts.seller])
        .rpc();

      testResults.push({
        test: "long_tracking",
        passed: false,
        note: "Should have failed - tracking too long",
      });
    } catch (err) {
      const errorStr = err.toString();
      if (errorStr.includes("InvalidTracking")) {
        console.log("✅ Correctly rejected long tracking number");
        testResults.push({
          test: "long_tracking",
          passed: true,
          error: "InvalidTracking",
        });
      } else {
        testResults.push({
          test: "long_tracking",
          passed: false,
          error: errorStr,
        });
      }
    }

    const allPassed = testResults.every((r) => r.passed);

    outputResult(allPassed, {
      orderId: orderId.toString(),
      amount: amount.toString(),
      transactions: {
        create: tx1,
        accept: tx2,
      },
      validationTests: testResults,
    });
  } catch (error) {
    console.error("Error:", error);
    outputResult(false, { error: error.message });
  }
}

main();
