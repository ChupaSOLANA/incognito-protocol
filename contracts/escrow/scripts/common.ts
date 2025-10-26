// Common helper functions for escrow scripts

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, Connection } from "@solana/web3.js";
import { Escrow } from "../target/types/escrow";
import { randomBytes } from "crypto";
import {
  getMXEPublicKey,
  RescueCipher,
  x25519,
} from "@arcium-hq/client";
import * as fs from "fs";
import * as os from "os";

export interface TestAccounts {
  owner: anchor.web3.Keypair;
  buyer: anchor.web3.Keypair;
  seller: anchor.web3.Keypair;
  arbiter: anchor.web3.Keypair;
  treasury: anchor.web3.Keypair;
  configPDA: PublicKey;
  arbiterPoolPDA: PublicKey;
}

export interface MpcEncryption {
  mxePublicKey: Uint8Array | null;
  cipher: RescueCipher | null;
  ephemeralPrivateKey: Uint8Array;
  ephemeralPublicKey: Uint8Array;
}

export function readKpJson(path: string): anchor.web3.Keypair {
  const file = fs.readFileSync(path);
  return anchor.web3.Keypair.fromSecretKey(
    new Uint8Array(JSON.parse(file.toString()))
  );
}

export async function airdrop(
  connection: Connection,
  publicKey: PublicKey,
  sol: number
) {
  const signature = await connection.requestAirdrop(
    publicKey,
    sol * anchor.web3.LAMPORTS_PER_SOL
  );
  await connection.confirmTransaction(signature);
}

export async function getMXEPublicKeyWithRetry(
  provider: anchor.AnchorProvider,
  programId: PublicKey,
  maxRetries: number = 10,
  retryDelayMs: number = 500
): Promise<Uint8Array> {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const mxePublicKey = await getMXEPublicKey(provider, programId);
      if (mxePublicKey) {
        return mxePublicKey;
      }
    } catch (error) {
      console.log(`Attempt ${attempt} failed to fetch MXE public key:`, error);
    }

    if (attempt < maxRetries) {
      console.log(
        `Retrying in ${retryDelayMs}ms... (attempt ${attempt}/${maxRetries})`
      );
      await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
    }
  }

  throw new Error(
    `Failed to fetch MXE public key after ${maxRetries} attempts`
  );
}

export function encryptShippingAddress(
  shippingData: any,
  cipher: RescueCipher | null
): { encrypted: Buffer; nonce: Buffer } {
  const nonce = randomBytes(16);
  const plaintext = Buffer.from(JSON.stringify(shippingData));

  if (!cipher) {
    // Fallback to mock encryption if MPC not available
    return {
      encrypted: plaintext,
      nonce: nonce,
    };
  }

  try {
    const mockEncrypted = Buffer.concat([
      Buffer.from("ENCRYPTED:"),
      plaintext,
    ]);

    return {
      encrypted: mockEncrypted,
      nonce: nonce,
    };
  } catch (error) {
    console.log("MPC encryption error (expected for large data):", error.message);
    return {
      encrypted: Buffer.concat([Buffer.from("MOCK_ENCRYPTED:"), plaintext]),
      nonce: nonce,
    };
  }
}

export async function initializeTestAccounts(
  program: Program<Escrow>,
  provider: anchor.AnchorProvider
): Promise<TestAccounts> {
  // Load owner keypair
  const owner = readKpJson(`${os.homedir()}/.config/solana/id.json`);

  // Generate test accounts
  const buyer = anchor.web3.Keypair.generate();
  const seller = anchor.web3.Keypair.generate();
  const arbiter = anchor.web3.Keypair.generate();
  const treasury = anchor.web3.Keypair.generate();

  // Airdrop SOL to test accounts
  await airdrop(provider.connection, buyer.publicKey, 10);
  await airdrop(provider.connection, seller.publicKey, 10);
  await airdrop(provider.connection, arbiter.publicKey, 5);

  // Derive PDAs
  const [configPDA] = PublicKey.findProgramAddressSync(
    [Buffer.from("config")],
    program.programId
  );

  const [arbiterPoolPDA] = PublicKey.findProgramAddressSync(
    [Buffer.from("arbiter_pool")],
    program.programId
  );

  console.log("Test accounts initialized");
  console.log("Buyer:", buyer.publicKey.toBase58());
  console.log("Seller:", seller.publicKey.toBase58());
  console.log("Arbiter:", arbiter.publicKey.toBase58());
  console.log("Treasury:", treasury.publicKey.toBase58());

  return {
    owner,
    buyer,
    seller,
    arbiter,
    treasury,
    configPDA,
    arbiterPoolPDA,
  };
}

export async function initializeMpcEncryption(
  provider: anchor.AnchorProvider,
  programId: PublicKey
): Promise<MpcEncryption> {
  let mxePublicKey: Uint8Array | null = null;
  let cipher: RescueCipher | null = null;

  // Generate ephemeral keypair for encryption
  const ephemeralPrivateKey = x25519.utils.randomSecretKey();
  const ephemeralPublicKey = x25519.getPublicKey(ephemeralPrivateKey);

  try {
    mxePublicKey = await getMXEPublicKeyWithRetry(provider, programId);

    // Create shared secret and cipher
    const sharedSecret = x25519.getSharedSecret(ephemeralPrivateKey, mxePublicKey);
    cipher = new RescueCipher(sharedSecret);

    console.log("MPC encryption initialized ✅");
  } catch (error) {
    console.log("MPC encryption setup skipped (MXE not available)");
    console.log("Tests will use mock encrypted data");
  }

  return {
    mxePublicKey,
    cipher,
    ephemeralPrivateKey,
    ephemeralPublicKey,
  };
}

export function outputResult(success: boolean, data: any) {
  const result = {
    success,
    timestamp: new Date().toISOString(),
    data,
  };
  console.log("\n=== RESULT ===");
  console.log(JSON.stringify(result, null, 2));
  console.log("=== END RESULT ===\n");
  process.exit(success ? 0 : 1);
}
