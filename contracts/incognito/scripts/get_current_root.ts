// scripts/get_current_root.ts
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as anchor from "@coral-xyz/anchor";
import { Program, AnchorProvider } from "@coral-xyz/anchor";
import { PublicKey, Connection, Keypair } from "@solana/web3.js";

// Import the Incognito IDL type (you may need to generate this)
type Incognito = any; // TODO: Generate proper type from IDL

function loadLocalKeypair(): Keypair {
  const p = path.join(os.homedir(), ".config", "solana", "id.json");
  const secret = JSON.parse(fs.readFileSync(p, "utf8")) as number[];
  return Keypair.fromSecretKey(Uint8Array.from(secret));
}

async function main() {
  const rpc = process.env.RPC_URL ?? process.env.SOLANA_RPC_URL ?? "http://127.0.0.1:8899";
  const connection = new Connection(rpc, { commitment: "confirmed" });
  const kp = loadLocalKeypair();
  const wallet = new anchor.Wallet(kp);

  const provider = new AnchorProvider(connection, wallet, {
    commitment: "confirmed",
    preflightCommitment: "confirmed",
  });
  anchor.setProvider(provider);

  // Load program from environment or use hardcoded ID
  const programId = new PublicKey(
    process.env.INCOGNITO_PROG_ID ?? "4N49EyRoX9p9zoiv1weeeqpaJTGbEHizbzZVgrsrVQeC"
  );

  // Load IDL
  const idlPath = process.env.INCOGNITO_IDL_PATH ??
    path.join(process.cwd(), "target/idl/incognito.json");
  const idl = JSON.parse(fs.readFileSync(idlPath, "utf8"));
  const program = new Program(idl, programId, provider);

  // Derive pool_state PDA (uses seed "pool_state")
  const [poolStatePda] = PublicKey.findProgramAddressSync(
    [Buffer.from("pool_state")],
    program.programId
  );

  try {
    const poolState = await program.account.poolState.fetch(poolStatePda);
    const rootBuf = Buffer.from(poolState.root as number[]);

    console.log("Pool State PDA:", poolStatePda.toBase58());
    console.log("Current root:", "0x" + rootBuf.toString("hex"));
    console.log("Depth:", poolState.depth);
    console.log("Leaf count:", poolState.leafCount.toString());
  } catch (e) {
    console.error("Failed to fetch pool state. Make sure the pool is initialized.");
    console.error(e);
    process.exit(1);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
