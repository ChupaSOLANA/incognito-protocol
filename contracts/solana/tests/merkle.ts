// tests/merkle.ts
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as anchor from "@coral-xyz/anchor";
import { Program, AnchorProvider, BN } from "@coral-xyz/anchor";
import {
  PublicKey,
  Connection,
  Keypair,
  LAMPORTS_PER_SOL,
} from "@solana/web3.js";
import MerkleTree from "merkletreejs";
import { createHash, randomBytes } from "crypto";
import { MerkleRegistry } from "../target/types/merkle_registry";

function sha256Hash(data: Buffer): Buffer {
  return createHash("sha256").update(data).digest();
}

function loadLocalKeypair(): Keypair {
  const p = path.join(os.homedir(), ".config", "solana", "id.json");
  const secret = JSON.parse(fs.readFileSync(p, "utf8")) as number[];
  return Keypair.fromSecretKey(Uint8Array.from(secret));
}

async function airdropIfNeeded(conn: Connection, pk: PublicKey) {
  const bal = await conn.getBalance(pk, "confirmed");
  if (bal < 0.5 * LAMPORTS_PER_SOL) {
    const sig = await conn.requestAirdrop(pk, 2 * LAMPORTS_PER_SOL);
    await conn.confirmTransaction(sig, "confirmed");
  }
}

function printIdlAccounts(program: any, ixName: string) {
  const idl = program?.idl;
  const ix = idl?.instructions?.find((i: any) => i.name === ixName);
  if (!ix) {
    console.log(`IDL: instruction ${ixName} not found`);
    return;
  }
  console.log(
    `IDL accounts for ${ixName}:`,
    ix.accounts.map((a: any) => a.name)
  );
}

(async () => {
  const rpc = process.env.RPC_URL ?? "http://127.0.0.1:8899";
  const connection = new Connection(rpc, { commitment: "confirmed" });
  const kp = loadLocalKeypair();
  const wallet = new anchor.Wallet(kp);
  await airdropIfNeeded(connection, wallet.publicKey);

  const provider = new AnchorProvider(connection, wallet, {
    commitment: "confirmed",
    preflightCommitment: "confirmed",
  });
  anchor.setProvider(provider);

  const program = anchor.workspace.MerkleRegistry as Program<MerkleRegistry>;
  console.log("Client program id:", program.programId.toBase58());

  printIdlAccounts(program, "initTree");
  printIdlAccounts(program, "updateRoot");
  printIdlAccounts(program, "verifyProof");

  // Build Merkle root
  const leaves = ["alice", "bob", "carol", "dave"].map((s) => Buffer.from(s));
  const tree = new MerkleTree(leaves, sha256Hash, {
    hashLeaves: true,
    sortPairs: false,
  });
  const root = tree.getRoot(); // 32 bytes

  // Local seed to derive PDA client-side
  const seed = randomBytes(32);
  const [treePda] = PublicKey.findProgramAddressSync(
    [Buffer.from("merkle"), seed],
    program.programId
  );
  console.log("Derived tree PDA:", treePda.toBase58());

  // ===== initTree =====
  await program.methods
    .initTree([...seed] as any, [...root] as any)
    .accounts({
      authority: wallet.publicKey,
    } as any)
    .simulate()
    .then((sim) => console.log("initTree simulate logs:", sim.raw));

  const initSig = await program.methods
    .initTree([...seed] as any, [...root] as any)
    .accounts({
      authority: wallet.publicKey,
    } as any)
    .rpc();
  console.log("initTree sig:", initSig);

  const acct = await program.account.merkleTree.fetch(treePda);
  console.log("tree account after init:", acct);

  // ===== updateRoot =====
  await program.methods
    .updateRoot([...root] as any)
    .accounts({
      tree: treePda,
      authority: wallet.publicKey,
    } as any)
    .simulate()
    .then((sim) => console.log("updateRoot simulate logs:", sim.raw));

  const updSig = await program.methods
    .updateRoot([...root] as any)
    .accounts({
      tree: treePda,
      authority: wallet.publicKey,
    } as any)
    .rpc();
  console.log("updateRoot sig:", updSig);

  // ===== verifyProof (index 2 = "carol") =====
  const idx = 2;
  const leafRaw = leaves[idx];
  const leafHash = sha256Hash(leafRaw);

  // Build proof against the *hashed* leaf
  const proof = tree.getProof(leafHash).map((p) => p.data as Buffer);

  await program.methods
    .verifyProof(
      [...leafHash] as any,
      proof.map((b) => [...b]) as any,
      new BN(idx)
    )
    .accounts({
      tree: treePda,
      authority: wallet.publicKey,
    } as any)
    .simulate()
    .then((sim) => console.log("verifyProof simulate logs:", sim.raw));

  const verSig = await program.methods
    .verifyProof(
      [...leafHash] as any,
      proof.map((b) => [...b]) as any,
      new BN(idx)
    )
    .accounts({
      tree: treePda,
      authority: wallet.publicKey,
    } as any)
    .rpc();
  console.log("verifyProof sig:", verSig);

  console.log("ok");
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
