// scripts/compute_and_update_roots.ts
import * as fs from "fs";
import * as path from "path";
import { createHash } from "crypto";
import MerkleTree from "merkletreejs";
import * as anchor from "@coral-xyz/anchor";
import { Program, AnchorProvider, Idl } from "@coral-xyz/anchor";
import { PublicKey, Connection, Keypair } from "@solana/web3.js";
import { MerkleRegistry } from "../target/types/merkle_registry";

// ---------- Hash + helpers ----------
const sha256 = (d: Buffer) => createHash("sha256").update(d).digest();
const ZERO32 = Buffer.alloc(32, 0);
const ZERO_HEX = "0x" + "0".repeat(64);

const toHex = (b: Buffer | null) =>
  b ? "0x" + Buffer.from(b).toString("hex") : "(null)";

function hex32(u: unknown): Buffer {
  if (typeof u !== "string") throw new Error("expect hex string");
  const s = u.startsWith("0x") ? u.slice(2) : u;
  if (s.length !== 64) throw new Error(`bad hex len ${s.length}: ${s}`);
  return Buffer.from(s.toLowerCase(), "hex");
}

function merkleRoot(hexLeaves: string[]): Buffer | null {
  if (!hexLeaves || hexLeaves.length === 0) return null; // delta-driven: aucun arbre local
  const leaves = hexLeaves.map(hex32);
  const tree = new MerkleTree(leaves, sha256, {
    hashLeaves: false,
    sortPairs: false,
  });
  const root = tree.getRoot();
  if (!root || root.equals(ZERO32)) return null;
  return root;
}

function loadJSON<T>(p: string): T | {} {
  if (!fs.existsSync(p)) return {};
  const raw = fs.readFileSync(p, "utf8");
  try {
    return JSON.parse(raw) as T;
  } catch (e) {
    console.warn(`Failed to parse JSON at ${p}:`, (e as Error)?.message || e);
    return {};
  }
}

function sleep(ms: number) {
  return new Promise((res) => setTimeout(res, ms));
}

// ---------- Config ----------
type DeployConfig = {
  program_id?: string;
  tree_seed_hex?: string;
  idl_path?: string;
  rpc_url?: string;
  authority_keypair?: string;
};

function resolveDeployConfigPath(): string {
  if (process.env.MERKLE_DEPLOY_CONFIG)
    return path.resolve(process.env.MERKLE_DEPLOY_CONFIG);
  const guess1 = path.resolve(__dirname, "../deploy_config.json");
  const guess2 = path.resolve(
    __dirname,
    "../../../contracts/solana/deploy_config.json"
  );
  if (fs.existsSync(guess1)) return guess1;
  if (fs.existsSync(guess2)) return guess2;
  return "/Users/alex/Desktop/incognito-protocol-1/contracts/solana/deploy_config.json";
}

// ---------- Core ----------
async function main() {
  // Config
  const cfgPath = resolveDeployConfigPath();
  if (!fs.existsSync(cfgPath))
    throw new Error(`Deploy config not found at ${cfgPath}`);
  const cfg = loadJSON<DeployConfig>(cfgPath) as DeployConfig;
  console.log("Using deploy_config.json:", cfgPath);

  const rpc =
    cfg.rpc_url ??
    process.env.SOLANA_RPC_URL ??
    process.env.RPC_URL ??
    "http://127.0.0.1:8899";

  const programIdStr = cfg.program_id ?? process.env.MERKLE_PROG_ID;
  if (!programIdStr)
    throw new Error("Missing program id (program_id or MERKLE_PROG_ID).");
  const programId = new PublicKey(programIdStr);

  const idlPath = cfg.idl_path ?? process.env.MERKLE_IDL_PATH;
  if (!idlPath || !fs.existsSync(idlPath))
    throw new Error(`Missing IDL file at ${idlPath || "(undefined)"}`);
  const idl = JSON.parse(fs.readFileSync(idlPath, "utf8")) as Idl;

  const seedHex =
    cfg.tree_seed_hex ??
    process.env.MERKLE_SEED_HEX ??
    process.env.TREE_SEED_HEX;
  if (!seedHex || !/^[0-9a-fA-F]{64}$/.test(seedHex)) {
    throw new Error("tree_seed_hex must be 64 hex chars.");
  }
  const seed = Buffer.from(seedHex, "hex");

  const authorityKeyPath =
    cfg.authority_keypair ??
    process.env.MERKLE_AUTHORITY ??
    path.join(process.env.HOME || "", ".config/solana/id.json");
  if (!fs.existsSync(authorityKeyPath))
    throw new Error(`Missing authority keypair at ${authorityKeyPath}`);

  // Provider
  const connection = new Connection(rpc, { commitment: "confirmed" });
  const secret = JSON.parse(fs.readFileSync(authorityKeyPath, "utf8"));
  const kp = Keypair.fromSecretKey(Uint8Array.from(secret));
  const wallet = new anchor.Wallet(kp);
  const provider = new AnchorProvider(connection, wallet, {
    commitment: "confirmed",
    preflightCommitment: "confirmed",
  });
  anchor.setProvider(provider);

  // Program (Anchor ≥ 0.30)
  const program = new Program(idl as Idl, provider) as Program<MerkleRegistry>;
  const idlAddr =
    (idl as any)?.address ?? (idl as any)?.metadata?.address ?? null;
  if (!idlAddr || idlAddr !== programId.toBase58()) {
    throw new Error(
      `Program address mismatch. IDL=${idlAddr} CFG=${programId.toBase58()}`
    );
  }

  // PDA
  const [treePda] = PublicKey.findProgramAddressSync(
    [Buffer.from("merkle"), seed],
    programId
  );
  console.log("Derived tree PDA:", treePda.toBase58());

  // Chargement état local
  const msPath = path.resolve(__dirname, "../../../merkle_state.json");
  const pmPath = path.resolve(__dirname, "../../../pool_merkle_state.json");
  const merkleState = loadJSON<any>(msPath) as any;
  const poolState = loadJSON<any>(pmPath) as any;

  const A: string[] = Array.from(new Set([...(merkleState.leaves ?? [])]));
  const B: string[] = Array.from(
    new Set(
      [...(merkleState.notes ?? []).map((n: any) => n.leaf)].filter(Boolean)
    )
  );
  const POOL: string[] = Array.from(
    new Set(
      [...(poolState.records ?? []).map((r: any) => r.commitment)].filter(
        Boolean
      )
    )
  );
  const NULLIF: string[] = Array.from(
    new Set([...(merkleState.nullifiers ?? [])])
  );

  const rootA = merkleRoot(A);
  const rootB = merkleRoot(B);
  const rootPool = merkleRoot(POOL);
  const rootNull = merkleRoot(NULLIF); // informatif

  console.log(
    "rootCommitmentsA:",
    toHex(rootA) === "(null)" ? ZERO_HEX : toHex(rootA)
  );
  console.log(
    "rootCommitmentsB:",
    toHex(rootB) === "(null)" ? ZERO_HEX : toHex(rootB)
  );
  console.log(
    "rootPool        :",
    toHex(rootPool) === "(null)" ? ZERO_HEX : toHex(rootPool)
  );
  console.log(
    "rootNullifiers  :",
    toHex(rootNull) === "(null)" ? ZERO_HEX : toHex(rootNull)
  );

  // Sélection locale (delta-driven)
  const choice = (process.env.CANONICAL_SET ?? "A").toUpperCase();
  const mergeUnique = (arrs: string[][]) => Array.from(new Set(arrs.flat()));
  const pick = (): Buffer | null => {
    switch (choice) {
      case "A":
        return rootA;
      case "B":
        return rootB;
      case "POOL":
        return rootPool;
      case "MERGE_AB":
        return merkleRoot(mergeUnique([A, B]));
      case "MERGE_A_POOL":
        return merkleRoot(mergeUnique([A, POOL]));
      case "MERGE_B_POOL":
        return merkleRoot(mergeUnique([B, POOL]));
      default:
        throw new Error(`Unknown CANONICAL_SET='${choice}'`);
    }
  };
  const chosen = pick();
  console.log(
    "Chosen local root:",
    chosen ? toHex(chosen) : ZERO_HEX,
    `(set=${choice})`
  );

  // Helpers de fetch
  async function fetchOnchain(
    commitment: "processed" | "confirmed" | "finalized"
  ) {
    try {
      const acc: any = await (program.account as any).merkleTree.fetch(
        treePda,
        commitment
      );
      const rootArr: Uint8Array | number[] | undefined = acc?.root;
      const auth: PublicKey | null = acc?.authority ?? null;
      const rootBuf =
        rootArr && (rootArr as any).length === 32
          ? Buffer.from(rootArr as Uint8Array)
          : null;
      return { exists: true, rootBuf, auth };
    } catch {
      return {
        exists: false,
        rootBuf: null as Buffer | null,
        auth: null as PublicKey | null,
      };
    }
  }

  async function confirmFinalized(sig: string) {
    const bh = await provider.connection.getLatestBlockhash("finalized");
    await provider.connection.confirmTransaction(
      {
        signature: sig,
        blockhash: bh.blockhash,
        lastValidBlockHeight: bh.lastValidBlockHeight,
      },
      "finalized"
    );
  }

  // Flux delta-driven
  const state = await fetchOnchain("confirmed");

  // Cas 0: aucun arbre on-chain ET aucune racine locale non nulle -> no-op silencieux
  if (!state.exists && !chosen) {
    console.log("No on-chain tree and no local root — nothing to do.");
    process.exit(0);
  }

  // Cas 1: init quand une racine locale apparaît
  if (!state.exists && chosen) {
    console.log("Initializing on-chain tree with first non-zero local root…");
    const sig = await program.methods
      .initTree(Array.from(seed) as any, Array.from(chosen) as any)
      .accounts({ tree: treePda, authority: wallet.publicKey } as any)
      .rpc();
    await confirmFinalized(sig);

    const verify = await fetchOnchain("finalized");
    if (!verify.exists || !verify.rootBuf || !verify.rootBuf.equals(chosen)) {
      throw new Error(
        `Init verification failed. On-chain=${toHex(
          verify.rootBuf
        )} Wanted=${toHex(chosen)}`
      );
    }
    console.log("Tree initialized. Root:", toHex(verify.rootBuf));
    process.exit(0);
  }

  // Cas 2: arbre existe. Vérifier autorité
  if (state.exists && state.auth && !wallet.publicKey.equals(state.auth)) {
    throw new Error(
      `Authority mismatch. On-chain=${state.auth.toBase58()} Signer=${wallet.publicKey.toBase58()}`
    );
  }

  // Cas 3: pas de racine locale non nulle -> aucune mise à jour (on conserve l’état on-chain)
  if (!chosen) {
    console.log("Local root is null — keeping current on-chain root as-is.");
    process.exit(0);
  }

  // Cas 4: racine locale identique -> no-op
  if (state.rootBuf && state.rootBuf.equals(chosen)) {
    console.log("On-chain root equals local root — nothing to do.");
    process.exit(0);
  }

  // Cas 5: update delta
  const maxAttempts = 3;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      console.log(
        `Updating on-chain root (attempt ${attempt}/${maxAttempts})…`
      );
      const sig = await program.methods
        .updateRoot(Array.from(chosen) as any)
        .accounts({ tree: treePda, authority: wallet.publicKey } as any)
        .rpc();
      await confirmFinalized(sig);

      const final = await fetchOnchain("finalized");
      if (!final.rootBuf || !final.rootBuf.equals(chosen)) {
        throw new Error(
          `Post-update verification failed. On-chain=${toHex(
            final.rootBuf
          )} Wanted=${toHex(chosen)}`
        );
      }
      console.log("Updated on-chain root to:", toHex(final.rootBuf));
      process.exit(0);
    } catch (e: any) {
      const msg = String(e?.message || e).toLowerCase();
      if (msg.includes("6000"))
        throw new Error("Program error 6000 (unauthorized).");
      console.warn(`updateRoot attempt ${attempt} failed:`, e?.message || e);
      if (attempt < maxAttempts) await sleep(750 * attempt);
      else throw e;
    }
  }
}

main().catch((e) => {
  console.error("Uncaught error:", e?.message || e);
  process.exit(1);
});
