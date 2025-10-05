// scripts/get_current_root.ts
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as anchor from "@coral-xyz/anchor";
import { Program, AnchorProvider } from "@coral-xyz/anchor";
import { PublicKey, Connection, Keypair } from "@solana/web3.js";
import { MerkleRegistry } from "../target/types/merkle_registry";

function loadLocalKeypair(): Keypair {
  const p = path.join(os.homedir(), ".config", "solana", "id.json");
  const secret = JSON.parse(fs.readFileSync(p, "utf8")) as number[];
  return Keypair.fromSecretKey(Uint8Array.from(secret));
}

async function main() {
  const rpc = process.env.RPC_URL ?? "http://127.0.0.1:8899";
  const connection = new Connection(rpc, { commitment: "confirmed" });
  const kp = loadLocalKeypair();
  const wallet = new anchor.Wallet(kp);

  const provider = new AnchorProvider(connection, wallet, {
    commitment: "confirmed",
    preflightCommitment: "confirmed",
  });
  anchor.setProvider(provider);

  const program = anchor.workspace.MerkleRegistry as Program<MerkleRegistry>;

  // ⚠️ utilise le même seed 32 octets que celui passé à initTree
  const seed = Buffer.alloc(32, 1); // <-- remplace par ton seed réel
  const [treePda] = PublicKey.findProgramAddressSync(
    [Buffer.from("merkle"), seed],
    program.programId
  );

  const acct = await program.account.merkleTree.fetch(treePda);
  const rootBuf = Buffer.from(acct.root as number[]);
  console.log("Tree PDA:", treePda.toBase58());
  console.log("Current root:", "0x" + rootBuf.toString("hex"));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
