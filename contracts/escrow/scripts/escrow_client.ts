#!/usr/bin/env ts-node
//
// ================================================================================================
// WRAPPER-AS-ESCROW ARCHITECTURE
// ================================================================================================
//
// In this design, the wrapper holds all escrow funds in its confidential balance.
// The escrow contract ONLY tracks state - no token transfers happen in Rust.
//
// FLOW:
// 1. create_order: Buyer sends cSOL to wrapper (confidential transfer)
// 2. accept_order: Seller sends stake to wrapper (confidential transfer)
// 3. finalize_order: Wrapper sends funds to seller (confidential transfer)
// 4. refund: Wrapper sends funds back to buyer (confidential transfer)
//
// Why this works:
// - PDAs cannot hold confidential tokens (they lack private keys for encryption)
// - Wrapper has private keys and can maintain confidential balance
// - Escrow contract validates state transitions and authorization
// - Python API orchestrates wrapper transfers based on escrow state
//
// ================================================================================================

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, Keypair } from "@solana/web3.js";
import { Escrow } from "../../contracts/escrow/target/types/escrow";
import * as fs from "fs";

const ESCROW_PROGRAM_ID = new PublicKey("5QvQbnrL7fKpM5pCMS3zNqgTK8ALNkgHgRvgd49YF7v4");

function loadKeypair(path: string): Keypair {
  const secretKey = JSON.parse(fs.readFileSync(path, "utf8"));
  return Keypair.fromSecretKey(new Uint8Array(secretKey));
}

async function main() {
  const command = process.argv[2];

  if (!command) {
    console.error("Usage: escrow_client.ts <command> [args...]");
    console.error("Commands: create_order, accept_order, mark_shipped, confirm_delivery, finalize_order");
    process.exit(1);
  }

  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.Escrow as Program<Escrow>;

  const wrapperPath = process.env.WRAPPER_KEYPAIR || "./keys/wrapper.json";
  const wrapperKeypair = loadKeypair(wrapperPath);

  try {
    if (command === "create_order") {
      const buyerKeypairPath = process.argv[3];
      const amount = new anchor.BN(process.argv[4]); // amount in lamports
      const orderId = new anchor.BN(process.argv[5]);
      const sellerPub = new PublicKey(process.argv[6]);
      const encryptedShipping = Buffer.from(process.argv[7] || "encrypted_data");
      const shippingNonce = Buffer.from(JSON.parse(process.argv[8] || "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]"));

      const buyerKeypair = loadKeypair(buyerKeypairPath);

      const [escrowPDA] = PublicKey.findProgramAddressSync(
        [Buffer.from("escrow"), buyerKeypair.publicKey.toBuffer(), orderId.toArrayLike(Buffer, "le", 8)],
        program.programId
      );

      const [buyerRepPDA] = PublicKey.findProgramAddressSync(
        [Buffer.from("reputation"), buyerKeypair.publicKey.toBuffer()],
        program.programId
      );

      const [configPDA] = PublicKey.findProgramAddressSync(
        [Buffer.from("config")],
        program.programId
      );

      const [arbiterPoolPDA] = PublicKey.findProgramAddressSync(
        [Buffer.from("arbiter_pool")],
        program.programId
      );

      // ===== CREATE ESCROW STATE ONLY - NO TOKEN TRANSFER IN CONTRACT =====
      // The Python API will handle the actual confidential transfer: buyer → wrapper
      const tx = await program.methods
        .createOrder(amount, orderId, encryptedShipping, Array.from(shippingNonce), null, false)
        .accounts({
          buyer: buyerKeypair.publicKey,
          escrow: escrowPDA,
          buyerReputation: buyerRepPDA,
          config: configPDA,
          arbiterPool: arbiterPoolPDA,
          feePayer: wrapperKeypair.publicKey,
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .signers([buyerKeypair, wrapperKeypair])
        .rpc();

      console.log(JSON.stringify({
        success: true,
        tx,
        escrowPDA: escrowPDA.toBase58(),
        orderId: orderId.toString(),
        amount: amount.toString(),
      }));

    } else if (command === "accept_order") {
      const sellerKeypairPath = process.argv[3];
      const escrowPDAStr = process.argv[4];

      const sellerKeypair = loadKeypair(sellerKeypairPath);
      const escrowPDA = new PublicKey(escrowPDAStr);

      const escrow = await program.account.escrow.fetch(escrowPDA);

      const [sellerRepPDA] = PublicKey.findProgramAddressSync(
        [Buffer.from("reputation"), sellerKeypair.publicKey.toBuffer()],
        program.programId
      );

      const [configPDA] = PublicKey.findProgramAddressSync(
        [Buffer.from("config")],
        program.programId
      );

      const config = await program.account.platformConfig.fetch(configPDA);

      // Calculate seller stake for the API to handle
      const sellerStake = (escrow.amount.toNumber() * config.sellerStakeBps) / 10000;

      // ===== UPDATE ESCROW STATE ONLY - NO TOKEN TRANSFER IN CONTRACT =====
      // The Python API will handle the actual confidential transfer: seller → wrapper (stake)
      const tx = await program.methods
        .acceptOrder()
        .accounts({
          seller: sellerKeypair.publicKey,
          escrow: escrowPDA,
          sellerReputation: sellerRepPDA,
          config: configPDA,
          feePayer: wrapperKeypair.publicKey,
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .signers([sellerKeypair, wrapperKeypair])
        .rpc();

      console.log(JSON.stringify({
        success: true,
        tx,
        sellerStake: sellerStake.toString(),
      }));

    } else if (command === "mark_shipped") {
      const sellerKeypairPath = process.argv[3];
      const escrowPDAStr = process.argv[4];
      const trackingNumber = process.argv[5] || "TRACKING_NUMBER";

      const sellerKeypair = loadKeypair(sellerKeypairPath);
      const escrowPDA = new PublicKey(escrowPDAStr);

      const [configPDA] = PublicKey.findProgramAddressSync(
        [Buffer.from("config")],
        program.programId
      );

      const tx = await program.methods
        .markShipped(trackingNumber)
        .accounts({
          seller: sellerKeypair.publicKey,
          escrow: escrowPDA,
          config: configPDA,
        })
        .signers([sellerKeypair])
        .rpc();

      console.log(JSON.stringify({ success: true, tx }));

    } else if (command === "confirm_delivery") {
      const buyerKeypairPath = process.argv[3];
      const escrowPDAStr = process.argv[4];

      const buyerKeypair = loadKeypair(buyerKeypairPath);
      const escrowPDA = new PublicKey(escrowPDAStr);

      const tx = await program.methods
        .confirmDelivery()
        .accounts({
          buyer: buyerKeypair.publicKey,
          escrow: escrowPDA,
        })
        .signers([buyerKeypair])
        .rpc();

      console.log(JSON.stringify({ success: true, tx }));

    } else if (command === "finalize_order") {
      const escrowPDAStr = process.argv[3];
      const escrowPDA = new PublicKey(escrowPDAStr);

      const escrow = await program.account.escrow.fetch(escrowPDA);

      const [buyerRepPDA] = PublicKey.findProgramAddressSync(
        [Buffer.from("reputation"), escrow.buyer.toBuffer()],
        program.programId
      );

      const [sellerRepPDA] = PublicKey.findProgramAddressSync(
        [Buffer.from("reputation"), escrow.seller.toBuffer()],
        program.programId
      );

      const [configPDA] = PublicKey.findProgramAddressSync(
        [Buffer.from("config")],
        program.programId
      );

      // Calculate total payment for the API to handle
      const totalPayment = escrow.amount.toNumber() + escrow.sellerStake.toNumber();

      // ===== UPDATE ESCROW STATE ONLY - NO TOKEN TRANSFER IN CONTRACT =====
      // The Python API will handle the actual confidential transfer: wrapper → seller (amount + stake)
      const tx = await program.methods
        .finalizeOrder()
        .accounts({
          escrow: escrowPDA,
          buyer: escrow.buyer,
          seller: escrow.seller,
          buyerReputation: buyerRepPDA,
          sellerReputation: sellerRepPDA,
          config: configPDA,
        })
        .rpc();

      console.log(JSON.stringify({
        success: true,
        tx,
        totalPayment: totalPayment.toString(),
        seller: escrow.seller.toBase58(),
      }));

    } else {
      console.error(`Unknown command: ${command}`);
      process.exit(1);
    }
  } catch (error) {
    console.error(JSON.stringify({
      success: false,
      error: error instanceof Error ? error.message : String(error),
    }));
    process.exit(1);
  }
}

main();
