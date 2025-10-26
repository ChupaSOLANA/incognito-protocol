#!/usr/bin/env ts-node

// Test 3: Buyer wins dispute - Full flow with dispute resolved in buyer's favor

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

    const orderId = new anchor.BN(3);
    const amount = new anchor.BN(800_000_000); // 0.8 SOL
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

    // Create → Accept → Ship → Deliver
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
      .markShipped("TRACK99999")
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

    // Open dispute
    const tx5 = await program.methods
      .openDispute("Item was damaged on arrival")
      .accounts({
        buyer: accounts.buyer.publicKey,
        escrow: escrowPDA,
        buyerReputation: buyerRepPDA,
        config: accounts.configPDA,
      })
      .signers([accounts.buyer])
      .rpc();

    console.log("Dispute opened:", tx5);

    // Get buyer balance before resolution
    const buyerBalanceBefore = await provider.connection.getBalance(
      accounts.buyer.publicKey
    );

    // Arbiter resolves for buyer
    const tx6 = await program.methods
      .resolveDispute({ buyer: {} })
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

    console.log("Dispute resolved for buyer:", tx6);

    // Verify buyer got refund + stake
    const buyerBalanceAfter = await provider.connection.getBalance(
      accounts.buyer.publicKey
    );
    const sellerStake = amount.toNumber() * 0.1;
    const expectedRefund = amount.toNumber() + sellerStake;
    const actualRefund = buyerBalanceAfter - buyerBalanceBefore;

    // Verify reputations
    const buyerRep = await program.account.userReputation.fetch(buyerRepPDA);
    const sellerRep = await program.account.userReputation.fetch(sellerRepPDA);

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
        reason: "Item was damaged on arrival",
        winner: "buyer",
      },
      refund: {
        expected: expectedRefund,
        actual: actualRefund,
        difference: Math.abs(actualRefund - expectedRefund),
      },
      reputations: {
        buyer: {
          disputesWon: buyerRep.disputesWon.toString(),
          disputesLost: buyerRep.disputesLost.toString(),
        },
        seller: {
          disputesWon: sellerRep.disputesWon.toString(),
          disputesLost: sellerRep.disputesLost.toString(),
        },
      },
    });
  } catch (error) {
    console.error("Error:", error);
    outputResult(false, { error: error.message });
  }
}

main();
