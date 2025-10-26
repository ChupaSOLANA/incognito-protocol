#!/usr/bin/env ts-node

// Test 4: Seller wins dispute - Full flow with dispute resolved in seller's favor

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

    const orderId = new anchor.BN(4);
    const amount = new anchor.BN(600_000_000); // 0.6 SOL
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

    // Full flow to dispute
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

    const tx3 = await program.methods
      .markShipped("TRACK77777")
      .accounts({
        seller: accounts.seller.publicKey,
        escrow: escrowPDA,
        config: accounts.configPDA,
      })
      .signers([accounts.seller])
      .rpc();

    const tx4 = await program.methods
      .confirmDelivery()
      .accounts({
        buyer: accounts.buyer.publicKey,
        escrow: escrowPDA,
      })
      .signers([accounts.buyer])
      .rpc();

    const tx5 = await program.methods
      .openDispute("False claim - item is perfect")
      .accounts({
        buyer: accounts.buyer.publicKey,
        escrow: escrowPDA,
        buyerReputation: buyerRepPDA,
        config: accounts.configPDA,
      })
      .signers([accounts.buyer])
      .rpc();

    console.log("Dispute opened:", tx5);

    // Get seller balance before
    const sellerBalanceBefore = await provider.connection.getBalance(
      accounts.seller.publicKey
    );

    // Arbiter resolves for seller
    const tx6 = await program.methods
      .resolveDispute({ seller: {} })
      .accounts({
        arbiter: accounts.arbiter.publicKey,
        escrow: escrowPDA,
        buyer: accounts.buyer.publicKey,
        seller: accounts.seller.publicKey,
        treasury: accounts.treasury.publicKey,
        buyerReputation: buyerRepPDA,
        sellerReputation: sellerRepPDA,
        config: accounts.configPDA,
      })
      .signers([accounts.arbiter])
      .rpc();

    console.log("Dispute resolved for seller:", tx6);

    // Verify seller got payment
    const sellerBalanceAfter = await provider.connection.getBalance(
      accounts.seller.publicKey
    );
    const platformFee = amount.toNumber() * 0.02;
    const sellerStake = amount.toNumber() * 0.1;
    const expectedPayment = amount.toNumber() - platformFee + sellerStake;
    const actualPayment = sellerBalanceAfter - sellerBalanceBefore;

    // Verify reputations
    const sellerRep = await program.account.userReputation.fetch(sellerRepPDA);
    const buyerRep = await program.account.userReputation.fetch(buyerRepPDA);

    outputResult(true, {
      orderId: orderId.toString(),
      amount: amount.toString(),
      transactions: {
        create: tx1,
        accept: tx2,
        ship: tx3,
        deliver: tx4,
        dispute: tx5,
        resolve: tx6,
      },
      dispute: {
        reason: "False claim - item is perfect",
        winner: "seller",
      },
      payment: {
        expected: expectedPayment,
        actual: actualPayment,
        difference: Math.abs(actualPayment - expectedPayment),
      },
      reputations: {
        seller: {
          disputesWon: sellerRep.disputesWon.toString(),
          disputesLost: sellerRep.disputesLost.toString(),
        },
        buyer: {
          disputesWon: buyerRep.disputesWon.toString(),
          disputesLost: buyerRep.disputesLost.toString(),
        },
      },
    });
  } catch (error) {
    console.error("Error:", error);
    outputResult(false, { error: error.message });
  }
}

main();
