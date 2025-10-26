#!/usr/bin/env ts-node

// Test 8: Unauthorized operations - Verify access control

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

    const orderId = new anchor.BN(8);
    const amount = new anchor.BN(250_000_000);
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

    // Try buyer marking shipped (should fail - only seller can)
    try {
      await program.methods
        .markShipped("FAKE123")
        .accounts({
          seller: accounts.buyer.publicKey, // Wrong signer
          escrow: escrowPDA,
          config: accounts.configPDA,
        })
        .signers([accounts.buyer])
        .rpc();

      outputResult(false, {
        orderId: orderId.toString(),
        transactions: { create: tx1, accept: tx2 },
        error: "Should have failed - wrong signer for mark_shipped",
      });
    } catch (err) {
      const errorStr = err.toString();
      if (errorStr.includes("Unauthorized")) {
        console.log("✅ Correctly rejected unauthorized mark_shipped");
        outputResult(true, {
          orderId: orderId.toString(),
          amount: amount.toString(),
          transactions: {
            create: tx1,
            accept: tx2,
          },
          authorizationTest: {
            test: "buyer_cannot_mark_shipped",
            passed: true,
            error: "Unauthorized",
          },
        });
      } else {
        outputResult(false, {
          orderId: orderId.toString(),
          transactions: { create: tx1, accept: tx2 },
          error: `Unexpected error: ${errorStr}`,
        });
      }
    }
  } catch (error) {
    console.error("Error:", error);
    outputResult(false, { error: error.message });
  }
}

main();
