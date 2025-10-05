// scripts/compare_roots.ts
import * as fs from "fs";
import * as path from "path";
import { createHash } from "crypto";
import MerkleTree from "merkletreejs";
import * as anchor from "@coral-xyz/anchor";
import { AnchorProvider, Program } from "@coral-xyz/anchor";
import { Connection, Keypair, PublicKey } from "@solana/web3.js";
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
    sortPairs: false, // IMPORTANT: mirrors on-chain left/right hashing
  });
  return tree.getRoot();
}

function hex(b: Buffer) {
  return "0x" + Buffer.from(b).toString("hex");
}

function loadJSON<T>(p: string): T {
  return JSON.parse(fs.readFileSync(p, "utf8")) as T;
}

async function main() {
  // ---------- 1) Compute LOCAL root ----------
  const msPath = path.resolve(__dirname, "../../../merkle_state.json");
  const pmPath = path.resolve(__dirname, "../../../pool_merkle_state.json");

  const merkleState = fs.existsSync(msPath) ? loadJSON<any>(msPath) : {};
  const poolState = fs.existsSync(pmPath) ? loadJSON<any>(pmPath) : {};

  // commitments from top-level leaves
  const leavesA: string[] = Array.from(
    new Set([...(merkleState.leaves ?? [])])
  );

  // commitments from notes[].leaf (some UIs track them this way)
  const leavesB: string[] = Array.from(
    new Set(
      [...(merkleState.notes ?? [])].map((n: any) => n.leaf).filter(Boolean)
    )
  );

  // commitments published to the pool file
  const poolLeaves: string[] = Array.from(
    new Set(
      [...(poolState.records ?? [])]
        .map((r: any) => r.commitment)
        .filter(Boolean)
    )
  );

  // choose which source(s) define the canonical tree:
  // Here we merge A + pool. Adjust to your policy if needed.
  const merged = Array.from(new Set([...leavesA, ...poolLeaves]));
  const localRoot = merkleRoot(merged);

  // ---------- 2) Read ON-CHAIN root ----------
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

  const seedHex =
    process.env.MERKLE_SEED_HEX ?? Buffer.alloc(32, 1).toString("hex");
  const seed = Buffer.from(seedHex, "hex");
  const [treePda] = PublicKey.findProgramAddressSync(
    [Buffer.from("merkle"), seed],
    program.programId
  );

  const acct = await program.account.merkleTree.fetchNullable(treePda);
  const onchainRoot: Buffer = acct
    ? Buffer.from((acct.root as number[]) ?? [])
    : Buffer.alloc(32, 0);

  // ---------- 3) Report / exit code ----------
  const localHex = hex(localRoot);
  const chainHex = hex(onchainRoot);
  const ok = localHex.toLowerCase() === chainHex.toLowerCase();

  console.log("Local   root:", localHex);
  console.log("On-chain root:", chainHex);
  console.log(ok ? "✔ Roots MATCH" : "✖ Roots MISMATCH");
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
