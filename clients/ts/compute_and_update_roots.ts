import * as fs from "fs";
import * as path from "path";
import { createHash } from "crypto";
import MerkleTree from "merkletreejs";
import * as anchor from "@coral-xyz/anchor";
import { Program, AnchorProvider, BN } from "@coral-xyz/anchor";
import { PublicKey, Connection, Keypair } from "@solana/web3.js";

const sha256 = (d: Buffer) => createHash("sha256").update(d).digest();

function h2(a: Buffer, b: Buffer): Buffer {
  return sha256(Buffer.concat([a, b]));
}

function h1(x: Buffer): Buffer {
  return sha256(x);
}

function leafFrom(commitment: Buffer, nfHash: Buffer): Buffer {
  return h2(commitment, nfHash);
}

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
  const msPath = path.resolve(__dirname, "../../merkle_state.json");
  const pmPath = path.resolve(__dirname, "../../pool_merkle_state.json");

  const merkleState = loadJSON<any>(msPath);
  const poolState = loadJSON<any>(pmPath);

  const poolRecords: any[] = poolState.records ?? [];

  const poolLeaves: string[] = poolRecords
    .filter((r: any) => r.commitment && r.nullifier)
    .map((r: any) => {
      const commitment = hex32(r.commitment);
      const nullifier = hex32(r.nullifier);
      const nfHash = h1(nullifier);
      const leaf = leafFrom(commitment, nfHash);
      return "0x" + leaf.toString("hex");
    });

  const commitmentsA: string[] = Array.from(
    new Set([...(merkleState.leaves ?? [])])
  );

  const rootPool = merkleRoot(poolLeaves.length > 0 ? poolLeaves : commitmentsA);

  const hex = (b: Buffer) => "0x" + b.toString("hex");
  console.log("Pool root (from records):", hex(rootPool));
  console.log("Number of pool leaves:", poolLeaves.length);
  const rpc = process.env.RPC_URL ?? process.env.SOLANA_RPC_URL ?? "http://127.0.0.1:8899";
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

  // Load program
  const programId = new PublicKey(
    process.env.INCOGNITO_PROG_ID ?? "4N49EyRoX9p9zoiv1weeeqpaJTGbEHizbzZVgrsrVQeC"
  );
  const idlPath = process.env.INCOGNITO_IDL_PATH ??
    path.join(__dirname, "../../contracts/incognito/target/idl/incognito.json");
  const idl = JSON.parse(fs.readFileSync(idlPath, "utf8"));
  const program = new Program(idl, programId, provider);

  const [poolStatePda] = PublicKey.findProgramAddressSync(
    [Buffer.from("pool_state")],
    program.programId
  );

  console.log("\n=== On-Chain Verification ===");
  try {
    const onChainPoolState = await program.account.poolState.fetch(poolStatePda);
    const onChainRoot = Buffer.from(onChainPoolState.root as number[]);

    console.log("On-chain root:", hex(onChainRoot));
    console.log("Computed root: ", hex(rootPool));
    console.log("Roots match:   ", onChainRoot.equals(rootPool) ? "✓ YES" : "✗ NO");
    console.log("\nOn-chain leaf count:", onChainPoolState.leafCount.toString());
    console.log("Local leaf count:   ", poolLeaves.length);
  } catch (e) {
    console.error("Failed to fetch on-chain pool state.");
    console.error("Make sure the pool is initialized with: init_pool instruction");
    console.error(e);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
