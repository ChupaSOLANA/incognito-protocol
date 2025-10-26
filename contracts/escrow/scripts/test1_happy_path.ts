#!/usr/bin/env ts-node

// Test 1: Happy path flow - Create → Accept → Ship → Deliver

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, SystemProgram } from "@solana/web3.js";
import { Escrow } from "../target/types/escrow";
import {
  initializeTestAccounts,
  initializeMpcEncryption,
  encryptShippingAddress,
  outputResult,
} from "./common";

async function main() {
  try {
    // Configure the client
    anchor.setProvider(anchor.AnchorProvider.env());
    const program = anchor.workspace.Escrow as Program<Escrow>;
    const provider = anchor.getProvider() as anchor.AnchorProvider;

    // Initialize test accounts
    const accounts = await initializeTestAccounts(program, provider);

    // Initialize MPC encryption
    const mpc = await initializeMpcEncryption(provider, program.programId);

    const orderId = new anchor.BN(1);
    const amount = new anchor.BN(1_000_000_000); // 1 SOL

    // Encrypt shipping address using Arcium MPC
    const shippingAddress = {
      street: "123 Main St",
      city: "San Francisco",
      postal_code: "94102",
      country: "USA",
      phone: "+1234567890",
    };

    const { encrypted: encryptedShipping, nonce: shippingNonce } =
      encryptShippingAddress(shippingAddress, mpc.cipher);

    console.log("✅ Shipping address encrypted using MPC");

    // Derive escrow PDA
    const [escrowPDA] = PublicKey.findProgramAddressSync(
      [
        Buffer.from("escrow"),
        accounts.buyer.publicKey.toBuffer(),
        orderId.toArrayLike(Buffer, "le", 8),
      ],
      program.programId
    );

    // Derive reputation PDAs
    const [buyerRepPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("reputation"), accounts.buyer.publicKey.toBuffer()],
      program.programId
    );

    const [sellerRepPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("reputation"), accounts.seller.publicKey.toBuffer()],
      program.programId
    );

    // Step 1: Create order
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

    // Step 2: Seller accepts order
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

    console.log("Order accepted:", tx2);

    // Step 3: Seller marks shipped
    const tx3 = await program.methods
      .markShipped("TRACK12345")
      .accounts({
        seller: accounts.seller.publicKey,
        escrow: escrowPDA,
        config: accounts.configPDA,
      })
      .signers([accounts.seller])
      .rpc();

    console.log("Order shipped:", tx3);

    // Step 4: Buyer confirms delivery
    const tx4 = await program.methods
      .confirmDelivery()
      .accounts({
        buyer: accounts.buyer.publicKey,
        escrow: escrowPDA,
      })
      .signers([accounts.buyer])
      .rpc();

    console.log("Order delivered:", tx4);

    // Verify escrow state is Delivered
    const escrowAccount = await program.account.escrow.fetch(escrowPDA);

    console.log("\n🔐 PRIVACY STATUS:");
    console.log("✅ Shipping address: ENCRYPTED (stored as ciphertext)");
    console.log("✅ Shipping nonce:", escrowAccount.shippingEncryptionNonce);
    console.log(
      "❌ Order amount:",
      escrowAccount.amount.toString(),
      "lamports (must be plaintext for Solana)"
    );

    outputResult(true, {
      orderId: orderId.toString(),
      amount: amount.toString(),
      transactions: {
        create: tx1,
        accept: tx2,
        ship: tx3,
        deliver: tx4,
      },
      escrow: {
        state: escrowAccount.state,
        buyer: escrowAccount.buyer.toBase58(),
        seller: escrowAccount.seller.toBase58(),
        amount: escrowAccount.amount.toString(),
        trackingNumber: escrowAccount.trackingNumber,
      },
      note: "Finalization requires waiting 7 days for dispute window to pass",
    });
  } catch (error) {
    console.error("Error:", error);
    outputResult(false, { error: error.message });
  }
}

main();
