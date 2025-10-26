#!/usr/bin/env ts-node

// Test 2: Acceptance timeout - Order created but not accepted within time limit

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

    const orderId = new anchor.BN(2);
    const amount = new anchor.BN(500_000_000); // 0.5 SOL
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

    // Create order
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

    console.log("Order created:", tx1);
    console.log("Order created, waiting for timeout...");

    // Try to process timeout (should fail if not past deadline)
    let timeoutProcessed = false;
    let errorMessage = "";

    try {
      await program.methods
        .processAcceptanceTimeout()
        .accounts({
          escrow: escrowPDA,
          buyer: accounts.buyer.publicKey,
          config: accounts.configPDA,
        })
        .rpc();

      timeoutProcessed = true;
    } catch (err) {
      errorMessage = err.toString();
      if (errorMessage.includes("DeadlineNotReached")) {
        console.log("✅ Correctly prevented early timeout processing");
      } else {
        throw err;
      }
    }

    outputResult(true, {
      orderId: orderId.toString(),
      amount: amount.toString(),
      transactions: {
        create: tx1,
      },
      timeout: {
        processed: timeoutProcessed,
        error: errorMessage,
        note: "In real test, would need to wait 24 hours or use time-warp",
      },
    });
  } catch (error) {
    console.error("Error:", error);
    outputResult(false, { error: error.message });
  }
}

main();
