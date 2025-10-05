// scripts/read_onchain_root.ts
import * as fs from "fs";
import * as path from "path";
import * as anchor from "@coral-xyz/anchor";
import { AnchorProvider, Program, Idl } from "@coral-xyz/anchor";
import { Connection, Keypair, PublicKey } from "@solana/web3.js";
import type { MerkleRegistry } from "../target/types/merkle_registry";

type DeployConfig = {
  program_id?: string;
  tree_seed_hex?: string;
  idl_path?: string;
  rpc_url?: string;
  authority_keypair?: string;
  generated_at_epoch?: number;
};

function resolveDeployConfigPath(): string {
  if (process.env.MERKLE_DEPLOY_CONFIG) {
    return path.resolve(process.env.MERKLE_DEPLOY_CONFIG);
  }
  const guess1 = path.resolve(__dirname, "../deploy_config.json");
  const guess2 = path.resolve(
    __dirname,
    "../../../contracts/solana/deploy_config.json"
  );
  if (fs.existsSync(guess1)) return guess1;
  if (fs.existsSync(guess2)) return guess2;
  return "/Users/alex/Desktop/incognito-protocol-1/contracts/solana/deploy_config.json";
}

function loadJSON<T>(p: string): T {
  return JSON.parse(fs.readFileSync(p, "utf8")) as T;
}

function toHex(b: Buffer) {
  return "0x" + Buffer.from(b).toString("hex");
}

(async () => {
  try {
    // ---- Load same deploy config used elsewhere
    const configPath = resolveDeployConfigPath();
    if (!fs.existsSync(configPath)) {
      throw new Error(`Deploy config not found at ${configPath}`);
    }
    const cfg = loadJSON<DeployConfig>(configPath);
    console.log("Using deploy_config.json:", configPath);

    // ---- RPC / wallet
    const rpc =
      cfg.rpc_url ??
      process.env.SOLANA_RPC_URL ??
      process.env.RPC_URL ??
      "http://127.0.0.1:8899";
    const connection = new Connection(rpc, { commitment: "confirmed" });

    const authorityKeyPath =
      cfg.authority_keypair ??
      process.env.MERKLE_AUTHORITY ??
      path.join(process.env.HOME || "", ".config/solana/id.json");
    const secret = JSON.parse(fs.readFileSync(authorityKeyPath, "utf8"));
    const kp = Keypair.fromSecretKey(Uint8Array.from(secret));
    const wallet = new anchor.Wallet(kp);

    const provider = new AnchorProvider(connection, wallet, {
      commitment: "confirmed",
      preflightCommitment: "confirmed",
    });
    anchor.setProvider(provider);

    // ---- Load IDL and pick program id from config
    const idlPath = cfg.idl_path ?? process.env.MERKLE_IDL_PATH;
    if (!idlPath || !fs.existsSync(idlPath)) {
      throw new Error(`Missing IDL file at ${idlPath || "(undefined)"}`);
    }
    const idl = JSON.parse(fs.readFileSync(idlPath, "utf8")) as Idl;

    const programIdStr = cfg.program_id ?? process.env.MERKLE_PROG_ID;
    if (!programIdStr) {
      throw new Error("Missing program id (program_id or MERKLE_PROG_ID).");
    }
    const programId = new PublicKey(programIdStr);

    // Soft sanity check: accept idl.address OR idl.metadata.address
    const idlAddr =
      (idl as any)?.address || (idl as any)?.metadata?.address || null;
    if (idlAddr && idlAddr !== programId.toBase58()) {
      console.warn(
        `Warning: IDL address (${idlAddr}) != deploy_config program_id (${programId.toBase58()}). Proceeding with deploy_config program_id.`
      );
    }

    // Bind Program to the programId from config
    const program = new Program(
      idl as Idl,
      provider
    ) as Program<MerkleRegistry>;
    console.log("Program ID:", programId.toBase58());

    // ---- Derive PDA with seed from config
    const seedHex =
      cfg.tree_seed_hex ??
      process.env.MERKLE_SEED_HEX ??
      process.env.TREE_SEED_HEX;
    if (!seedHex || !/^[0-9a-fA-F]{64}$/.test(seedHex)) {
      throw new Error(
        "tree_seed_hex must be 64 hex chars in deploy_config.json."
      );
    }
    const seed = Buffer.from(seedHex, "hex");

    const [treePda] = PublicKey.findProgramAddressSync(
      [Buffer.from("merkle"), seed],
      programId
    );
    console.log("Tree PDA:", treePda.toBase58());
    console.log("Seed (hex):", seedHex);

    // ---- Fetch account (confirmed)
    const acct = await (program.account as any).merkleTree.fetchNullable(
      treePda
    );
    if (!acct) {
      console.log(
        "Account not found at PDA (check program_id / seed / cluster)."
      );
      console.log("onchainRoot:", "0x" + Buffer.alloc(32, 0).toString("hex"));
      process.exit(0);
    }

    const rootArr: number[] | Uint8Array = acct.root;
    const root =
      rootArr && (rootArr as any).length === 32
        ? Buffer.from(rootArr as Uint8Array)
        : Buffer.alloc(32, 0);

    const authority = (acct.authority?.toBase58?.() ??
      String(acct.authority)) as string;

    console.log("Authority  :", authority);
    console.log("onchainRoot:", toHex(root));
  } catch (e: any) {
    console.error("read_onchain_root error:", e?.message || e);
    process.exit(1);
  }
})();
