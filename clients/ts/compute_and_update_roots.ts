// scripts/compute_and_update_roots.ts
import * as fs from "fs";
import * as path from "path";
import { createHash } from "crypto";
import MerkleTree from "merkletreejs";
import * as anchor from "@coral-xyz/anchor";
import { Program, AnchorProvider, BN } from "@coral-xyz/anchor";
import { PublicKey, Connection, Keypair } from "@solana/web3.js";
import { MerkleRegistry } from "../target/types/merkle_registry";

const sha256 = (d: Buffer) => createHash("sha256").update(d).digest();

function hex32(u: unknown): Buffer {
  if (typeof u !== "string") throw new Error("expect hex string");
  const s = u.startsWith("0x") ? u.slice(2) : u;
  if (s.length !== 64) throw new Error(`bad len ${s.length}: ${s}`);
  return Buffer.from(s.toLowerCase(), "hex");
}

function merkleRoot(hexLeaves: string[]): Buffer {
  if (!hexLeaves.length) return Buffer.alloc(32, 0);
  const leaves = hexLeaves.map(hex32);
  const tree = new MerkleTree(leaves, sha256, {
    hashLeaves: false,
    sortPairs: false,
  });
  return tree.getRoot();
}

function loadJSON<T>(p: string): T {
  return JSON.parse(fs.readFileSync(p, "utf8")) as T;
}

async function main() {
  // ---- Load files ----
  const msPath = path.resolve(__dirname, "../merkle_state.json");
  const pmPath = path.resolve(__dirname, "../pool_merkle_state.json");

  const merkleState = loadJSON<any>(msPath);
  const poolState = loadJSON<any>(pmPath);

  // ---- Build sets of leaves ----
  // commitments from top-level leaves
  const commitmentsA: string[] = Array.from(
    new Set([...(merkleState.leaves ?? [])])
  );

  // commitments from notes[].leaf
  const commitmentsB: string[] = Array.from(
    new Set(
      [...(merkleState.notes ?? []).map((n: any) => n.leaf)].filter(Boolean)
    )
  );

  // commitments from pool records
  const poolCommitments: string[] = Array.from(
    new Set(
      [...(poolState.records ?? []).map((r: any) => r.commitment)].filter(
        Boolean
      )
    )
  );

  // nullifiers
  const nullifiers: string[] = Array.from(
    new Set([...(merkleState.nullifiers ?? [])])
  );

  // ---- Compute roots ----
  const rootCommitmentsA = merkleRoot(commitmentsA);
  const rootCommitmentsB = merkleRoot(commitmentsB);
  const rootPool = merkleRoot(poolCommitments);
  const rootNullifiers = merkleRoot(nullifiers);

  // Print hex
  const hex = (b: Buffer) => "0x" + b.toString("hex");
  console.log("rootCommitmentsA:", hex(rootCommitmentsA));
  console.log("rootCommitmentsB:", hex(rootCommitmentsB));
  console.log("rootPool        :", hex(rootPool));
  console.log("rootNullifiers  :", hex(rootNullifiers));

  // ---- Push one root to on-chain tree (example) ----
  // Configure provider
  const rpc = process.env.RPC_URL ?? "http://127.0.0.1:8899";
  const connection = new Connection(rpc, { commitment: "confirmed" });
  const secret = JSON.parse(
    fs.readFileSync(
      path.join(process.env.HOME!, ".config/solana/id.json"),
      "utf8"
    )
  );
  const kp = Keypair.fromSecretKey(Uint8Array.from(secret));
  const wallet = new anchor.Wallet(kp);
  const provider = new AnchorProvider(connection, wallet, {
    commitment: "confirmed",
    preflightCommitment: "confirmed",
  });
  anchor.setProvider(provider);

  const program = anchor.workspace.MerkleRegistry as Program<MerkleRegistry>;

  // Derive or load your MerkleTree PDA (seed you used at init)
  // If you need a fresh tree: keep the seed fixed and stored client-side.
  const seed = Buffer.alloc(32, 1); // <-- replace by your persisted 32B seed
  const [treePda] = PublicKey.findProgramAddressSync(
    [Buffer.from("merkle"), seed],
    program.programId
  );

  // One-time init (idempotent if already exists)
  try {
    await program.methods
      .initTree([...seed] as any, Array.from(rootCommitmentsA) as any)
      .accounts({ authority: wallet.publicKey } as any)
      .rpc();
  } catch (_) {
    // ignore if already initialized
  }

  // Update to a new root (choose which set is canonical for your tree)
  await program.methods
    .updateRoot(Array.from(rootCommitmentsA) as any)
    .accounts({ tree: treePda, authority: wallet.publicKey } as any)
    .rpc();

  console.log("Updated on-chain root to:", hex(rootCommitmentsA));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
