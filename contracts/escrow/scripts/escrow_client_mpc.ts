#!/usr/bin/env ts-node
/**
 * Enhanced escrow client with MPC (Multi-Party Computation) integration
 *
 * This version uses Arcium encrypted computations for:
 * - Seller stake calculation
 * - Platform fee calculation
 * - Refund amount calculation
 * - Fund distribution calculations
 */

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, Keypair } from "@solana/web3.js";
import { Escrow } from "../target/types/escrow";
import * as fs from "fs";
import {
  getMXEPublicKey,
  RescueCipher,
  x25519,
} from "@arcium-hq/client";
import { randomBytes } from "crypto";

const ESCROW_PROGRAM_ID = new PublicKey("5QvQbnrL7fKpM5pCMS3zNqgTK8ALNkgHgRvgd49YF7v4");

function loadKeypair(path: string): Keypair {
  const secretKey = JSON.parse(fs.readFileSync(path, "utf8"));
  return Keypair.fromSecretKey(new Uint8Array(secretKey));
}

// Arcium MPC encryption context
interface MpcEncryption {
  mxePublicKey: Uint8Array;
  cipher: RescueCipher;
  ephemeralPrivateKey: Uint8Array;
  ephemeralPublicKey: Uint8Array;
}

let mpcContext: MpcEncryption | null = null;

// Initialize MPC encryption with Arcium
async function initializeMpcEncryption(
  provider: anchor.AnchorProvider,
  programId: PublicKey
): Promise<MpcEncryption> {
  if (mpcContext) {
    return mpcContext;
  }

  try {
    // Get MXE public key from Arcium network
    const mxePublicKey = await getMXEPublicKey(provider, programId);

    if (!mxePublicKey) {
      throw new Error("Failed to get MXE public key");
    }

    // Generate ephemeral keypair for encryption
    const ephemeralPrivateKey = x25519.utils.randomSecretKey();
    const ephemeralPublicKey = x25519.getPublicKey(ephemeralPrivateKey);

    // Create shared secret and cipher
    const sharedSecret = x25519.getSharedSecret(ephemeralPrivateKey, mxePublicKey);
    const cipher = new RescueCipher(sharedSecret);

    console.log("✅ MPC encryption initialized with Arcium");

    mpcContext = {
      mxePublicKey,
      cipher,
      ephemeralPrivateKey,
      ephemeralPublicKey,
    };

    return mpcContext;
  } catch (error) {
    console.error("❌ Failed to initialize MPC encryption:", error);
    throw error;
  }
}

// Encrypt a u64 value using Arcium MPC (REAL ENCRYPTION)
function encryptU64(value: number, cipher: RescueCipher, nonce: Uint8Array): Uint8Array {
  // Convert u64 to bytes (little-endian)
  const buffer = Buffer.alloc(8);
  buffer.writeBigUInt64LE(BigInt(value), 0);

  // Encrypt using RescueCipher from Arcium
  const encrypted = cipher.encrypt(buffer, nonce);

  // Pad to 32 bytes as required by MPC
  const padded = Buffer.alloc(32);
  encrypted.copy(padded);

  return new Uint8Array(padded);
}

async function main() {
  const command = process.argv[2];

  if (!command) {
    console.error("Usage: escrow_client_mpc.ts <command> [args...]");
    console.error("Commands:");
    console.error("  create_order <buyer_keypair> <amount> <order_id> <seller_pub> [encrypted_shipping] [shipping_nonce]");
    console.error("  accept_order_mpc <seller_keypair> <escrow_pda> <computation_offset>");
    console.error("  calculate_stake <escrow_pda> <computation_offset>");
    console.error("  calculate_fee <escrow_pda> <computation_offset>");
    console.error("  finalize_order_mpc <escrow_pda> <computation_offset>");
    process.exit(1);
  }

  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.Escrow as Program<Escrow>;

  const wrapperPath = process.env.WRAPPER_KEYPAIR || "./keys/wrapper.json";
  const wrapperKeypair = loadKeypair(wrapperPath);

  // Check if this command needs MPC encryption
  const needsMpc = [
    "accept_order_mpc",
    "calculate_stake",
    "calculate_fee",
    "finalize_order_mpc"
  ].includes(command);

  let mpc: MpcEncryption | null = null;

  // Only initialize MPC encryption when needed (optimization)
  if (needsMpc) {
    console.log("🔐 Initializing MPC encryption...");
    mpc = await initializeMpcEncryption(provider, program.programId);
  }

  try {
    if (command === "create_order") {
      // Same as original - no MPC needed for order creation
      const buyerKeypairPath = process.argv[3];
      const amount = new anchor.BN(process.argv[4]);
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

    } else if (command === "accept_order_mpc") {
      const sellerKeypairPath = process.argv[3];
      const escrowPDAStr = process.argv[4];
      const computationOffset = new anchor.BN(process.argv[5] || Date.now());

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

      // Step 1: Accept the order (traditional way)
      const acceptTx = await program.methods
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

      // Step 2: Calculate seller stake using MPC with REAL encryption
      if (!mpc) {
        throw new Error("MPC not initialized");
      }
      const amountNonce = randomBytes(16);
      const stakeNonce = randomBytes(16);
      const encryptedAmount = Array.from(encryptU64(escrow.amount.toNumber(), mpc.cipher, amountNonce));
      const encryptedStakePercentage = Array.from(encryptU64(config.sellerStakeBps, mpc.cipher, stakeNonce));
      const pubKey = Array.from(mpc.ephemeralPublicKey.slice(0, 32));
      const nonce = BigInt("0x" + amountNonce.toString('hex'));

      const stakeTx = await program.methods
        .calculateSellerStakePrivate(
          computationOffset,
          encryptedAmount,
          encryptedStakePercentage,
          pubKey,
          new anchor.BN(nonce)
        )
        .accounts({
          payer: wrapperKeypair.publicKey,
        })
        .signers([wrapperKeypair])
        .rpc();

      console.log(JSON.stringify({
        success: true,
        acceptTx,
        stakeTx,
        message: "Order accepted, stake calculation queued in MPC",
      }));

    } else if (command === "calculate_stake") {
      const escrowPDAStr = process.argv[3];
      const computationOffset = new anchor.BN(process.argv[4] || Date.now());

      const escrowPDA = new PublicKey(escrowPDAStr);
      const escrow = await program.account.escrow.fetch(escrowPDA);

      const [configPDA] = PublicKey.findProgramAddressSync(
        [Buffer.from("config")],
        program.programId
      );

      const config = await program.account.platformConfig.fetch(configPDA);

      // Use REAL Arcium encryption
      const amountNonce = randomBytes(16);
      const stakeNonce = randomBytes(16);
      const encryptedAmount = Array.from(encryptU64(escrow.amount.toNumber(), mpc.cipher, amountNonce));
      const encryptedStakePercentage = Array.from(encryptU64(config.sellerStakeBps, mpc.cipher, stakeNonce));
      const pubKey = Array.from(mpc.ephemeralPublicKey.slice(0, 32));
      const nonce = BigInt("0x" + amountNonce.toString('hex'));

      const tx = await program.methods
        .calculateSellerStakePrivate(
          computationOffset,
          encryptedAmount,
          encryptedStakePercentage,
          pubKey,
          new anchor.BN(nonce)
        )
        .accounts({
          payer: wrapperKeypair.publicKey,
        })
        .signers([wrapperKeypair])
        .rpc();

      console.log(JSON.stringify({
        success: true,
        tx,
        message: "Seller stake calculation queued in MPC",
        computationOffset: computationOffset.toString(),
      }));

    } else if (command === "calculate_fee") {
      const escrowPDAStr = process.argv[3];
      const computationOffset = new anchor.BN(process.argv[4] || Date.now());

      const escrowPDA = new PublicKey(escrowPDAStr);
      const escrow = await program.account.escrow.fetch(escrowPDA);

      const [configPDA] = PublicKey.findProgramAddressSync(
        [Buffer.from("config")],
        program.programId
      );

      const config = await program.account.platformConfig.fetch(configPDA);

      // Use REAL Arcium encryption
      const amountNonce = randomBytes(16);
      const feeNonce = randomBytes(16);
      const encryptedAmount = Array.from(encryptU64(escrow.amount.toNumber(), mpc.cipher, amountNonce));
      const encryptedFeePercentage = Array.from(encryptU64(config.platformFeeBps, mpc.cipher, feeNonce));
      const pubKey = Array.from(mpc.ephemeralPublicKey.slice(0, 32));
      const nonce = BigInt("0x" + amountNonce.toString('hex'));

      const tx = await program.methods
        .calculatePlatformFeePrivate(
          computationOffset,
          encryptedAmount,
          encryptedFeePercentage,
          pubKey,
          new anchor.BN(nonce)
        )
        .accounts({
          payer: wrapperKeypair.publicKey,
        })
        .signers([wrapperKeypair])
        .rpc();

      console.log(JSON.stringify({
        success: true,
        tx,
        message: "Platform fee calculation queued in MPC",
        computationOffset: computationOffset.toString(),
      }));

    } else if (command === "finalize_order_mpc") {
      const escrowPDAStr = process.argv[3];
      const computationOffset = new anchor.BN(process.argv[4] || Date.now());

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

      // Step 1: Finalize order (traditional way)
      const finalizeTx = await program.methods
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

      // Step 2: Calculate fund distribution using MPC with REAL encryption
      const amountNonce = randomBytes(16);
      const stakeNonce = randomBytes(16);
      const feeNonce = randomBytes(16);
      const encryptedAmount = Array.from(encryptU64(escrow.amount.toNumber(), mpc.cipher, amountNonce));
      const encryptedStake = Array.from(encryptU64(escrow.sellerStake.toNumber(), mpc.cipher, stakeNonce));
      const encryptedFee = Array.from(encryptU64(escrow.platformFee.toNumber(), mpc.cipher, feeNonce));
      const pubKey = Array.from(mpc.ephemeralPublicKey.slice(0, 32));
      const nonce = BigInt("0x" + amountNonce.toString('hex'));

      const distributionTx = await program.methods
        .calculateCompletionDistributionPrivate(
          computationOffset,
          encryptedAmount,
          encryptedStake,
          encryptedFee,
          pubKey,
          new anchor.BN(nonce)
        )
        .accounts({
          payer: wrapperKeypair.publicKey,
        })
        .signers([wrapperKeypair])
        .rpc();

      console.log(JSON.stringify({
        success: true,
        finalizeTx,
        distributionTx,
        message: "Order finalized, distribution calculation queued in MPC",
        totalPayment: (escrow.amount.toNumber() + escrow.sellerStake.toNumber()).toString(),
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
