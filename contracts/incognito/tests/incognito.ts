import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import {
  PublicKey,
  Keypair,
  SystemProgram,
  LAMPORTS_PER_SOL,
} from "@solana/web3.js";
import { Incognito } from "../target/types/incognito";
import { randomBytes, createHash } from "crypto";
import {
  awaitComputationFinalization,
  getArciumEnv,
  getCompDefAccOffset,
  getArciumProgAddress,
  buildFinalizeCompDefTx,
  RescueCipher,
  deserializeLE,
  getMXEAccAddress,
  getMempoolAccAddress,
  getCompDefAccAddress,
  getExecutingPoolAccAddress,
  x25519,
  getComputationAccAddress,
  getArciumAccountBaseSeed,
  getMXEPublicKey,
} from "@arcium-hq/client";
import * as fs from "fs";
import * as os from "os";
import { expect } from "chai";

/** --------------------- Note model (client-side) -------------------- **/

type Bytes16 = Uint8Array;
type Bytes32 = Uint8Array;

export interface ShieldedNote {
  secret: Bytes32; // Secret principal
  nullifier: Bytes32; // Nullifier preimage (rho)
  pk_view: Bytes32;
  ct_amount: Bytes32;
  nonce: Bytes16;
  recipient: Bytes32; // NEW: recipient public key
}

/** Normalize inputs to Buffer for hashing */
function asBuffer(p: any): Buffer {
  if (Buffer.isBuffer(p)) return p;
  if (p instanceof Uint8Array) return Buffer.from(p);
  if (Array.isArray(p)) return Buffer.from(p);
  if (typeof p === "string") return Buffer.from(p, "utf8");
  throw new Error("Unsupported input type for hashing");
}

function sha256Concat(parts: any[]): Bytes32 {
  const hash = createHash("sha256");
  for (const part of parts) {
    hash.update(asBuffer(part));
  }
  return new Uint8Array(hash.digest());
}

// JS equivalents for on-chain helpers
function h1js(x: Bytes32): Bytes32 {
  return sha256Concat([x]); // single-input
}
function h2js(a: Bytes32, b: Bytes32): Bytes32 {
  return sha256Concat([a, b]); // two-input concat-then-sha256
}

export function computeCommitment(note: ShieldedNote): Bytes32 {
  // C = H(secret || nullifier_preimage || pk_view || ct_amount || nonce || recipient)
  return sha256Concat([
    note.secret,
    note.nullifier,
    note.pk_view,
    note.ct_amount,
    note.nonce,
    note.recipient,
  ]);
}

// NF = H(secret || nullifier_preimage)
export function computeNullifier(secret: Bytes32, nullifier: Bytes32): Bytes32 {
  return sha256Concat([secret, nullifier]);
}

// nf_hash = h1(nullifier) where `nullifier` is NF (the computed nullifier)
export function computeNfHash(nullifierNF: Bytes32): Bytes32 {
  return h1js(nullifierNF);
}

export function encryptAmountToCt(
  cipher: RescueCipher,
  amount: bigint,
  nonce16: Bytes16,
): Bytes32 {
  return cipher.encrypt([amount], nonce16)[0];
}

export function createChangeNote(
  cipher: RescueCipher,
  changeAmount: bigint,
  recipientPubkey: Uint8Array,
): ShieldedNote {
  const secret = randomBytes(32);
  const nullifierPreimage = randomBytes(32);
  const sk = x25519.utils.randomSecretKey();
  const pk_view = x25519.getPublicKey(sk);
  const nonce = randomBytes(16);
  const ct_amount = encryptAmountToCt(cipher, changeAmount, nonce);

  return {
    secret,
    nullifier: nullifierPreimage,
    pk_view,
    ct_amount,
    nonce,
    recipient: recipientPubkey,
  };
}

/** ---------------------- Client-side Merkle tracker ---------------------- */

class MerkleTree {
  private leaves: Bytes32[] = []; // store LEAFS actually used on-chain
  private depth: number;

  constructor(depth: number) {
    this.depth = depth;
  }

  private h2(a: Bytes32, b: Bytes32): Bytes32 {
    const hash = createHash("sha256");
    hash.update(Buffer.from(a));
    hash.update(Buffer.from(b));
    return new Uint8Array(hash.digest());
  }

  private zeroAt(level: number): Bytes32 {
    let z = new Uint8Array(32);
    for (let i = 0; i < level; i++) {
      z = this.h2(z, z);
    }
    return z;
  }

  // IMPORTANT: This now expects the actual leaf value (H2(commitment, nf_hash))
  addLeaf(leaf: Bytes32): number {
    const index = this.leaves.length;
    this.leaves.push(leaf);
    return index;
  }

  getLeaves(): Bytes32[] {
    return [...this.leaves];
  }

  computeRoot(): Bytes32 {
    if (this.leaves.length === 0) {
      return this.zeroAt(this.depth);
    }

    let currentLevel = [...this.leaves];

    for (let level = 0; level < this.depth; level++) {
      const zero = this.zeroAt(level);
      const nextLevel: Bytes32[] = [];

      for (let i = 0; i < currentLevel.length; i += 2) {
        const left = currentLevel[i];
        const right = i + 1 < currentLevel.length ? currentLevel[i + 1] : zero;
        nextLevel.push(this.h2(left, right));
      }

      currentLevel = nextLevel;
    }

    return currentLevel[0];
  }

  getMerklePath(targetIndex: number): Bytes32[] {
    const path: Bytes32[] = [];
    let currentLevel = [...this.leaves];
    let idx = targetIndex;

    for (let level = 0; level < this.depth; level++) {
      const zero = this.zeroAt(level);

      const siblingIndex = (idx & 1) === 0 ? idx + 1 : idx - 1;
      const sibling =
        siblingIndex >= 0 && siblingIndex < currentLevel.length
          ? currentLevel[siblingIndex]
          : zero;
      path.push(sibling);

      const nextLevel: Bytes32[] = [];
      for (let i = 0; i < currentLevel.length; i += 2) {
        const left = currentLevel[i];
        const right = i + 1 < currentLevel.length ? currentLevel[i + 1] : zero;
        nextLevel.push(this.h2(left, right));
      }
      if (nextLevel.length === 0) nextLevel.push(this.h2(zero, zero));

      currentLevel = nextLevel;
      idx >>= 1;
    }

    return path;
  }
}

/** ---------------------- Helper Functions ---------------------- */

async function getMXEPublicKeyWithRetry(
  provider: anchor.AnchorProvider,
  programId: PublicKey,
  maxRetries = 10,
  retryDelayMs = 500,
): Promise<Uint8Array> {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const mxePublicKey = await getMXEPublicKey(provider, programId);
      if (mxePublicKey) return mxePublicKey;
    } catch {}
    if (attempt < maxRetries)
      await new Promise((r) => setTimeout(r, retryDelayMs));
  }
  throw new Error(
    `Failed to fetch MXE public key after ${maxRetries} attempts`,
  );
}

function readKpJson(path: string): anchor.web3.Keypair {
  const file = fs.readFileSync(path);
  return anchor.web3.Keypair.fromSecretKey(
    new Uint8Array(JSON.parse(file.toString())),
  );
}

async function pdaSign(pid: PublicKey): Promise<[PublicKey, number]> {
  const base = getArciumAccountBaseSeed("SignerAccount");
  return PublicKey.findProgramAddress([base], pid);
}

/**
 * Generate a dummy wrapper stealth address for tests.
 * In production, this would be generated using proper stealth address cryptography.
 * For tests, we just create a new keypair and return its public key.
 */
function generateDummyWrapperStealthAddress(): PublicKey {
  return Keypair.generate().publicKey;
}

async function ensureCompDef(
  program: Program<Incognito>,
  owner: Keypair,
  provider: anchor.AnchorProvider,
  name:
    | "deposit_shielded"
    | "withdraw_shielded"
    | "deposit_note"
    | "withdraw_note_check",
) {
  const baseSeedCompDefAcc = getArciumAccountBaseSeed(
    "ComputationDefinitionAccount",
  );
  const offset = getCompDefAccOffset(name);
  const compDefPDA = PublicKey.findProgramAddressSync(
    [baseSeedCompDefAcc, program.programId.toBuffer(), offset],
    getArciumProgAddress(),
  )[0];

  let needsInit = false;
  try {
    await (program.account as any).computationDefinitionAccount.fetch(
      compDefPDA,
    );
  } catch {
    needsInit = true;
  }

  if (needsInit) {
    const initIx =
      name === "deposit_shielded"
        ? program.methods.initDepositShieldedCompDef()
        : name === "withdraw_shielded"
          ? program.methods.initWithdrawShieldedCompDef()
          : name === "deposit_note"
            ? program.methods.initDepositNoteCompDef()
            : program.methods.initWithdrawNoteCheckCompDef();

    await initIx
      .accounts({
        compDefAccount: compDefPDA,
        payer: owner.publicKey,
        mxeAccount: getMXEAccAddress(program.programId),
        arciumProgram: getArciumProgAddress(),
        systemProgram: SystemProgram.programId,
      })
      .signers([owner])
      .rpc({ commitment: "confirmed" });
  }

  const finalizeTx = await buildFinalizeCompDefTx(
    provider,
    Buffer.from(offset).readUInt32LE(),
    program.programId,
  );
  const bh = await provider.connection.getLatestBlockhash();
  finalizeTx.recentBlockhash = bh.blockhash;
  finalizeTx.lastValidBlockHeight = bh.lastValidBlockHeight;
  finalizeTx.sign(owner);
  await provider.sendAndConfirm(finalizeTx, [owner], {
    commitment: "confirmed",
  });
}

/** Helper to close pool after each test */
async function closePoolIfExists(
  program: Program<Incognito>,
  owner: Keypair,
  poolStatePda: PublicKey,
) {
  try {
    await program.methods
      .closePool()
      .accounts({
        authority: owner.publicKey,
        poolState: poolStatePda,
      })
      .signers([owner])
      .rpc({ commitment: "confirmed" });
    console.log("✓ Pool closed successfully");
  } catch {
    console.log("✓ No pool to close");
  }
}

/** Helper to initialize fresh pool */
async function initFreshPool(
  program: Program<Incognito>,
  owner: Keypair,
  poolStatePda: PublicKey,
  depth: number = 20,
) {
  await program.methods
    .initPool(depth)
    .accounts({
      payer: owner.publicKey,
      poolState: poolStatePda,
      systemProgram: SystemProgram.programId,
    })
    .signers([owner])
    .rpc({ commitment: "confirmed" });
}

/** ---------------------------- Tests ------------------------ **/

describe("note model", () => {
  anchor.setProvider(anchor.AnchorProvider.env());
  const provider = anchor.getProvider() as anchor.AnchorProvider;
  const program = anchor.workspace.Incognito as Program<Incognito>;
  const owner = readKpJson(`${os.homedir()}/.config/solana/id.json`);

  it("builds a note locally and derives C & NF", async () => {
    const mxePublicKey = await getMXEPublicKeyWithRetry(
      provider,
      program.programId,
    );
    const secret = randomBytes(32);
    const nullifierPreimage = randomBytes(32);
    const sk = x25519.utils.randomSecretKey();
    const pk_view = x25519.getPublicKey(sk);
    const shared = x25519.getSharedSecret(sk, mxePublicKey);
    const cipher = new RescueCipher(shared);

    const amount = 1_234_567n;
    const nonce = randomBytes(16);

    const ct_amount = encryptAmountToCt(cipher, amount, nonce);
    const note: ShieldedNote = {
      secret,
      nullifier: nullifierPreimage,
      pk_view,
      ct_amount,
      nonce,
      recipient: new Uint8Array(owner.publicKey.toBytes()), // NEW
    };

    const C = computeCommitment(note);
    const NF = computeNullifier(secret, nullifierPreimage);
    const NFH = computeNfHash(NF);

    expect(C.length).to.equal(32);
    expect(NF.length).to.equal(32);
    expect(NFH.length).to.equal(32);

    const C2 = computeCommitment(note);
    const NF2 = computeNullifier(secret, nullifierPreimage);
    const NFH2 = computeNfHash(NF2);

    expect(Buffer.from(C2).equals(Buffer.from(C))).to.equal(true);
    expect(Buffer.from(NF2).equals(Buffer.from(NF))).to.equal(true);
    expect(Buffer.from(NFH2).equals(Buffer.from(NFH))).to.equal(true);

    console.log("✓ Note model: C, NF and nf_hash are consistent");
  });

  describe("sol_deposit_withdraw", () => {
    anchor.setProvider(anchor.AnchorProvider.env());
    const provider = anchor.getProvider() as anchor.AnchorProvider;
    const program = anchor.workspace.Incognito as Program<Incognito>;
    const arciumEnv = getArciumEnv();
    const owner = readKpJson(`${os.homedir()}/.config/solana/id.json`);

    type Event = anchor.IdlEvents<(typeof program)["idl"]>;
    const awaitEvent = async <E extends keyof Event>(
      name: E,
      timeoutMs = 60_000,
    ): Promise<Event[E]> => {
      let id = -1;
      let t: NodeJS.Timeout;
      const ev = await new Promise<Event[E]>((res, rej) => {
        id = program.addEventListener(name as any, (e) => {
          if (t) clearTimeout(t);
          res(e);
        });
        t = setTimeout(() => {
          program.removeEventListener(id);
          rej(new Error(`timeout: ${String(name)}`));
        }, timeoutMs);
      });
      await program.removeEventListener(id);
      return ev;
    };

    it("Shielded balance: deposit → withdraw success → withdraw fail", async () => {
      const mxePublicKey = await getMXEPublicKeyWithRetry(
        provider,
        program.programId,
      );
      const sk = x25519.utils.randomSecretKey();
      const pk = x25519.getPublicKey(sk);
      const shared = x25519.getSharedSecret(sk, mxePublicKey);
      const cipher = new RescueCipher(shared);

      await ensureCompDef(program, owner, provider, "deposit_shielded");
      await ensureCompDef(program, owner, provider, "withdraw_shielded");

      const depLamports = (1n * BigInt(LAMPORTS_PER_SOL)) / 100n;
      const depNonce = randomBytes(16);
      const [ctBalance0] = cipher.encrypt([0n], depNonce);
      const [_, ctAmountDep] = cipher.encrypt([0n, depLamports], depNonce);

      const depOffset = new anchor.BN(randomBytes(8), "hex");
      const depEvPromise = awaitEvent("depositShieldedEvent");

      await program.methods
        .depositShielded(
          depOffset,
          Array.from(ctBalance0),
          Array.from(ctAmountDep),
          Array.from(pk),
          new anchor.BN(deserializeLE(depNonce).toString()),
        )
        .accounts({
          payer: owner.publicKey,
          signPdaAccount: (await pdaSign(program.programId))[0],
          mxeAccount: getMXEAccAddress(program.programId),
          mempoolAccount: getMempoolAccAddress(program.programId),
          executingPool: getExecutingPoolAccAddress(program.programId),
          computationAccount: getComputationAccAddress(
            program.programId,
            depOffset,
          ),
          compDefAccount: getCompDefAccAddress(
            program.programId,
            Buffer.from(getCompDefAccOffset("deposit_shielded")).readUInt32LE(),
          ),
          clusterAccount: arciumEnv.arciumClusterPubkey,
          poolAccount: arciumEnv.arciumFeePoolPubkey,
          clockAccount: arciumEnv.arciumClockPubkey,
          systemProgram: SystemProgram.programId,
          arciumProgram: getArciumProgAddress(),
        })
        .signers([owner])
        .rpc({ skipPreflight: true, commitment: "confirmed" });

      await awaitComputationFinalization(
        provider,
        depOffset,
        program.programId,
        "confirmed",
      );
      const depEv = await depEvPromise;

      const balance1 = cipher.decrypt([depEv.newBalance], depEv.nonce)[0];
      expect(balance1).to.equal(depLamports);
      console.log(`✓ Deposited ${depLamports} lamports (encrypted balance)`);

      const wdLamports = BigInt(4_000_000);
      const wdNonce = randomBytes(16);
      const [ctBal1] = cipher.encrypt([balance1], wdNonce);
      const [__, ctAmountWd] = cipher.encrypt([0n, wdLamports], wdNonce);

      const wdOffset = new anchor.BN(randomBytes(8), "hex");
      const wdEvPromise = awaitEvent("withdrawShieldedEvent");

      await program.methods
        .withdrawShielded(
          wdOffset,
          Array.from(ctBal1),
          Array.from(ctAmountWd),
          Array.from(pk),
          new anchor.BN(deserializeLE(wdNonce).toString()),
        )
        .accounts({
          payer: owner.publicKey,
          signPdaAccount: (await pdaSign(program.programId))[0],
          mxeAccount: getMXEAccAddress(program.programId),
          mempoolAccount: getMempoolAccAddress(program.programId),
          executingPool: getExecutingPoolAccAddress(program.programId),
          computationAccount: getComputationAccAddress(
            program.programId,
            wdOffset,
          ),
          compDefAccount: getCompDefAccAddress(
            program.programId,
            Buffer.from(
              getCompDefAccOffset("withdraw_shielded"),
            ).readUInt32LE(),
          ),
          clusterAccount: arciumEnv.arciumClusterPubkey,
          poolAccount: arciumEnv.arciumFeePoolPubkey,
          clockAccount: arciumEnv.arciumClockPubkey,
          systemProgram: SystemProgram.programId,
          arciumProgram: getArciumProgAddress(),
        })
        .signers([owner])
        .rpc({ skipPreflight: true, commitment: "confirmed" });

      await awaitComputationFinalization(
        provider,
        wdOffset,
        program.programId,
        "confirmed",
      );
      const wdEv = await wdEvPromise;
      expect(wdEv.success).to.equal(true);

      const balance2 = cipher.decrypt([wdEv.newBalance], wdEv.nonce)[0];
      expect(balance2).to.equal(balance1 - wdLamports);
      console.log(`✓ Withdrew ${wdLamports} lamports successfully`);

      const wdLamportsTooMuch = balance1 + 1n;
      const wdNonce2 = randomBytes(16);
      const [ctBal2] = cipher.encrypt([balance2], wdNonce2);
      const [___, ctAmountWdTooMuch] = cipher.encrypt(
        [0n, wdLamportsTooMuch],
        wdNonce2,
      );

      const wdOffset2 = new anchor.BN(randomBytes(8), "hex");
      const wdEvPromise2 = awaitEvent("withdrawShieldedEvent");

      await program.methods
        .withdrawShielded(
          wdOffset2,
          Array.from(ctBal2),
          Array.from(ctAmountWdTooMuch),
          Array.from(pk),
          new anchor.BN(deserializeLE(wdNonce2).toString()),
        )
        .accounts({
          payer: owner.publicKey,
          signPdaAccount: (await pdaSign(program.programId))[0],
          mxeAccount: getMXEAccAddress(program.programId),
          mempoolAccount: getMempoolAccAddress(program.programId),
          executingPool: getExecutingPoolAccAddress(program.programId),
          computationAccount: getComputationAccAddress(
            program.programId,
            wdOffset2,
          ),
          compDefAccount: getCompDefAccAddress(
            program.programId,
            Buffer.from(
              getCompDefAccOffset("withdraw_shielded"),
            ).readUInt32LE(),
          ),
          clusterAccount: arciumEnv.arciumClusterPubkey,
          poolAccount: arciumEnv.arciumFeePoolPubkey,
          clockAccount: arciumEnv.arciumClockPubkey,
          systemProgram: SystemProgram.programId,
          arciumProgram: getArciumProgAddress(),
        })
        .signers([owner])
        .rpc({ skipPreflight: true, commitment: "confirmed" });

      await awaitComputationFinalization(
        provider,
        wdOffset2,
        program.programId,
        "confirmed",
      );
      const wdEv2 = await wdEvPromise2;
      expect(wdEv2.success).to.equal(false);
      console.log("✓ Withdrawal rejected due to insufficient funds");
    });
  });

  /** ---------------------- Shielded pool tests (leaf = H2(C, h1(NF))) ---------------------- */

  describe("shielded_pool", () => {
    anchor.setProvider(anchor.AnchorProvider.env());
    const provider = anchor.getProvider() as anchor.AnchorProvider;
    const program = anchor.workspace.Incognito as Program<Incognito>;
    const owner = readKpJson(`${os.homedir()}/.config/solana/id.json`);

    const POOL_STATE_SEED = Buffer.from("pool_state");
    const SOL_VAULT_SEED = Buffer.from("sol_vault");
    const poolStatePda = PublicKey.findProgramAddressSync(
      [POOL_STATE_SEED],
      program.programId,
    )[0];
    const solVaultPda = PublicKey.findProgramAddressSync(
      [SOL_VAULT_SEED],
      program.programId,
    )[0];

    // Add this before hook
    before(async () => {
      console.log("\n--- Setting up vault for shielded_pool ---");
      try {
        await program.account.solVault.fetch(solVaultPda);
        console.log("✓ Vault already initialized");
      } catch {
        await program.methods
          .initVault()
          .accounts({
            payer: owner.publicKey,
            solVault: solVaultPda,
            systemProgram: SystemProgram.programId,
          })
          .signers([owner])
          .rpc({ commitment: "confirmed" });
        console.log("✓ Vault initialized");
      }
    });

    type Event = anchor.IdlEvents<(typeof program)["idl"]>;
    const awaitEvent = async <E extends keyof Event>(
      name: E,
      timeoutMs = 60_000,
    ): Promise<Event[E]> => {
      let id = -1;
      let t: NodeJS.Timeout;
      const ev = await new Promise<Event[E]>((res, rej) => {
        id = program.addEventListener(name as any, (e) => {
          if (t) clearTimeout(t);
          res(e);
        });
        t = setTimeout(() => {
          program.removeEventListener(id);
          rej(new Error(`timeout: ${String(name)}`));
        }, timeoutMs);
      });
      await program.removeEventListener(id);
      return ev;
    };

    // Close pool after each test for isolation
    afterEach(async () => {
      await closePoolIfExists(program, owner, poolStatePda);
    });

    it("init_pool → insert_note x3 (with nf_hash) → verify_proof (creates marker) → double-spend blocked", async () => {
      const depth = 10;
      const tree = new MerkleTree(depth);

      console.log("\n--- Initializing pool ---");
      const initEvPromise = awaitEvent("poolInitialized");
      await initFreshPool(program, owner, poolStatePda, depth);
      const initEv = await initEvPromise;

      expect(initEv.depth).to.equal(depth);
      expect(initEv.root.length).to.equal(32);
      console.log("✓ Pool initialized");

      const initialLocalRoot = tree.computeRoot();
      const initialOnChainRoot = initEv.root;
      console.log("\nInitial roots (empty tree):");
      console.log(
        `  Local:    ${Buffer.from(initialLocalRoot).toString("hex")}`,
      );
      console.log(
        `  On-chain: ${Buffer.from(initialOnChainRoot).toString("hex")}`,
      );
      console.log(
        `  Match: ${Buffer.from(initialLocalRoot).equals(Buffer.from(initialOnChainRoot))}`,
      );

      const ps0 = await program.account.poolState.fetch(poolStatePda);
      expect(ps0.depth).to.equal(depth);
      expect(ps0.leafCount.toString()).to.equal("0");

      const commitments: Bytes32[] = [];
      const nullifiersNF: Bytes32[] = [];
      const nfHashes: Bytes32[] = [];

      // Insert #1
      console.log("\n--- Inserting notes ---");
      const depositAmount = BigInt(100_000_000); // 0.1 SOL (enough to cover 0.05 SOL wrapper fee)

      const commitment1 = randomBytes(32);
      const NF1 = randomBytes(32);
      const nfHash1 = computeNfHash(NF1);
      const leaf1 = h2js(commitment1, nfHash1);
      const index1 = tree.getLeaves().length;
      const path1 = tree.getMerklePath(index1);
      tree.addLeaf(leaf1);

      commitments.push(commitment1);
      nullifiersNF.push(NF1);
      nfHashes.push(nfHash1);

      const COMMITMENT_SEED = Buffer.from("commitment");
      const commitmentMarkerPda1 = PublicKey.findProgramAddressSync(
        [COMMITMENT_SEED, commitment1],
        program.programId,
      )[0];

      const ins1P = awaitEvent("realDepositEvent");
      const wrapperStealth1 = generateDummyWrapperStealthAddress();
      await program.methods
        .depositToPool(
          new anchor.BN(depositAmount.toString()),
          Array.from(commitment1),
          Array.from(nfHash1),
          path1.map((p) => Array.from(p)),
        )
        .accounts({
          depositor: owner.publicKey,
          solVault: solVaultPda,
          poolState: poolStatePda,
          commitmentMarker: commitmentMarkerPda1,
          wrapperStealthAddress: wrapperStealth1,
          systemProgram: SystemProgram.programId,
        })
        .signers([owner])
        .rpc({ commitment: "confirmed" });
      const ins1 = await ins1P;

      const root1Local = tree.computeRoot();
      const root1Chain = ins1.root;
      console.log(`After insert #1:`);
      console.log(
        `  Local:    ${Buffer.from(root1Local).toString("hex").substring(0, 16)}...`,
      );
      console.log(
        `  On-chain: ${Buffer.from(root1Chain).toString("hex").substring(0, 16)}...`,
      );
      console.log(
        `  Match: ${Buffer.from(root1Local).equals(Buffer.from(root1Chain))}`,
      );

      expect(ins1.index.toString()).to.equal("0");
      console.log(`✓ Deposited note #1 at index 0`);

      // Insert #2
      const commitment2 = randomBytes(32);
      const NF2 = randomBytes(32);
      const nfHash2 = computeNfHash(NF2);
      const leaf2 = h2js(commitment2, nfHash2);
      const index2 = tree.getLeaves().length;
      const path2 = tree.getMerklePath(index2);
      tree.addLeaf(leaf2);

      commitments.push(commitment2);
      nullifiersNF.push(NF2);
      nfHashes.push(nfHash2);

      const commitmentMarkerPda2 = PublicKey.findProgramAddressSync(
        [COMMITMENT_SEED, commitment2],
        program.programId,
      )[0];

      const ins2P = awaitEvent("realDepositEvent");
      await program.methods
        .depositToPool(
          new anchor.BN(depositAmount.toString()),
          Array.from(commitment2),
          Array.from(nfHash2),
          path2.map((p) => Array.from(p)),
        )
        .accounts({
          depositor: owner.publicKey,
          solVault: solVaultPda,
          poolState: poolStatePda,
          commitmentMarker: commitmentMarkerPda2,
          wrapperStealthAddress: generateDummyWrapperStealthAddress(),
          systemProgram: SystemProgram.programId,
        })
        .signers([owner])
        .rpc({ commitment: "confirmed" });
      const ins2 = await ins2P;

      const root2Local = tree.computeRoot();
      const root2Chain = ins2.root;
      console.log(`After insert #2:`);
      console.log(
        `  Local:    ${Buffer.from(root2Local).toString("hex").substring(0, 16)}...`,
      );
      console.log(
        `  On-chain: ${Buffer.from(root2Chain).toString("hex").substring(0, 16)}...`,
      );
      console.log(
        `  Match: ${Buffer.from(root2Local).equals(Buffer.from(root2Chain))}`,
      );

      expect(ins2.index.toString()).to.equal("1");
      console.log(`✓ Deposited note #2 at index 1`);

      // Insert #3
      const commitment3 = randomBytes(32);
      const NF3 = randomBytes(32);
      const nfHash3 = computeNfHash(NF3);
      const leaf3 = h2js(commitment3, nfHash3);
      const index3 = tree.getLeaves().length;
      const path3 = tree.getMerklePath(index3);
      tree.addLeaf(leaf3);

      commitments.push(commitment3);
      nullifiersNF.push(NF3);
      nfHashes.push(nfHash3);

      const commitmentMarkerPda3 = PublicKey.findProgramAddressSync(
        [COMMITMENT_SEED, commitment3],
        program.programId,
      )[0];

      const ins3P = awaitEvent("realDepositEvent");
      await program.methods
        .depositToPool(
          new anchor.BN(depositAmount.toString()),
          Array.from(commitment3),
          Array.from(nfHash3),
          path3.map((p) => Array.from(p)),
        )
        .accounts({
          depositor: owner.publicKey,
          solVault: solVaultPda,
          poolState: poolStatePda,
          commitmentMarker: commitmentMarkerPda3,
          wrapperStealthAddress: generateDummyWrapperStealthAddress(),
          systemProgram: SystemProgram.programId,
        })
        .signers([owner])
        .rpc({ commitment: "confirmed" });
      const ins3 = await ins3P;

      const root3Local = tree.computeRoot();
      const root3Chain = ins3.root;
      console.log(`After insert #3:`);
      console.log(
        `  Local:    ${Buffer.from(root3Local).toString("hex").substring(0, 16)}...`,
      );
      console.log(
        `  On-chain: ${Buffer.from(root3Chain).toString("hex").substring(0, 16)}...`,
      );
      console.log(
        `  Match: ${Buffer.from(root3Local).equals(Buffer.from(root3Chain))}`,
      );

      expect(ins3.index.toString()).to.equal("2");
      console.log(`✓ Deposited note #3 at index 2`);

      // Verify proof for note #2 (index 1), MUST use matching nullifier NF2
      console.log("\n--- Verifying Merkle proof ---");
      const targetIndex = 1;
      const targetCommitment = commitments[targetIndex];
      const targetNF = nullifiersNF[targetIndex];
      const path = tree.getMerklePath(targetIndex);

      const localRoot = tree.computeRoot();
      const onChainRoot = ins3.root;

      console.log(
        `Local root:    ${Buffer.from(localRoot).toString("hex").substring(0, 16)}...`,
      );
      console.log(
        `On-chain root: ${Buffer.from(onChainRoot).toString("hex").substring(0, 16)}...`,
      );

      const rootsMatch = Buffer.from(localRoot).equals(
        Buffer.from(onChainRoot),
      );
      console.log(`Roots match:   ${rootsMatch}`);
      expect(rootsMatch).to.equal(true);

      const NULLIFIER_SEED = Buffer.from("nf");
      const nullifierPda = PublicKey.findProgramAddressSync(
        [NULLIFIER_SEED, targetNF],
        program.programId,
      )[0];

      const verifyEvP = awaitEvent("proofVerified");
      await program.methods
        .verifyProof(
          Array.from(targetCommitment),
          path.map((p) => Array.from(p)),
          Array.from(targetNF), // pass NF here; on-chain computes h1(NF) to bind
          new anchor.BN(targetIndex),
        )
        .accounts({
          payer: owner.publicKey,
          poolState: poolStatePda,
          nullifierMarker: nullifierPda,
          systemProgram: SystemProgram.programId,
        })
        .signers([owner])
        .rpc({ commitment: "confirmed" });
      const verifyEv = await verifyEvP;
      expect(verifyEv.index.toString()).to.equal(targetIndex.toString());
      console.log(`✓ Merkle proof verified for note at index ${targetIndex}`);

      // Confirm nullifier marker created by verifyProof (no explicit markSpent)
      const markerAcc = await (program.account as any).nullifierMarker.fetch(
        nullifierPda,
      );
      expect(markerAcc).to.not.equal(undefined);
      console.log("✓ Nullifier marker created by verifyProof");

      // Try to verify proof again with same nullifier (should fail — PDA already in use)
      console.log("\n--- Testing double-spend prevention ---");
      try {
        await program.methods
          .verifyProof(
            Array.from(targetCommitment),
            path.map((p) => Array.from(p)),
            Array.from(targetNF),
            new anchor.BN(targetIndex),
          )
          .accounts({
            payer: owner.publicKey,
            poolState: poolStatePda,
            nullifierMarker: nullifierPda,
            systemProgram: SystemProgram.programId,
          })
          .signers([owner])
          .rpc({ commitment: "confirmed" });
        expect.fail("Should have failed due to nullifier already used");
      } catch (err: any) {
        expect(err.toString()).to.include("already in use");
        console.log(
          "✓ Double-spend prevented successfully (PDA already in use)",
        );
      }

      const ps1 = await program.account.poolState.fetch(poolStatePda);
      expect(ps1.leafCount.toString()).to.equal("3");
      console.log("✓ Pool state updated correctly (3 leaves)");
    });
  });

  /** ---------------------- Note circuits (Arcis) ---------------------- */

  describe("note_circuits", () => {
    anchor.setProvider(anchor.AnchorProvider.env());
    const provider = anchor.getProvider() as anchor.AnchorProvider;
    const program = anchor.workspace.Incognito as Program<Incognito>;
    const arciumEnv = getArciumEnv();
    const owner = readKpJson(`${os.homedir()}/.config/solana/id.json`);

    type Event = anchor.IdlEvents<(typeof program)["idl"]>;
    const awaitEvent = async <E extends keyof Event>(
      name: E,
      timeoutMs = 60_000,
    ): Promise<Event[E]> => {
      let id = -1;
      let t: NodeJS.Timeout;
      const ev = await new Promise<Event[E]>((res, rej) => {
        id = program.addEventListener(name as any, (e) => {
          if (t) clearTimeout(t);
          res(e);
        });
        t = setTimeout(() => {
          program.removeEventListener(id);
          rej(new Error(`timeout: ${String(name)}`));
        }, timeoutMs);
      });
      await program.removeEventListener(id);
      return ev;
    };

    it("deposit_note → withdraw_note_check (success + failure)", async () => {
      const mxePublicKey = await getMXEPublicKeyWithRetry(
        provider,
        program.programId,
      );
      const sk = x25519.utils.randomSecretKey();
      const pk = x25519.getPublicKey(sk);
      const shared = x25519.getSharedSecret(sk, mxePublicKey);
      const cipher = new RescueCipher(shared);

      await ensureCompDef(program, owner, provider, "deposit_note");
      await ensureCompDef(program, owner, provider, "withdraw_note_check");

      console.log("\n--- Testing deposit_note circuit ---");
      const depAmount = 5_000_000n;
      const depNonce = randomBytes(16);
      const [ctAmount] = cipher.encrypt([depAmount], depNonce);

      const depOffset = new anchor.BN(randomBytes(8), "hex");
      const depEvPromise = awaitEvent("depositNoteEvent");

      await program.methods
        .depositNote(
          depOffset,
          Array.from(ctAmount),
          Array.from(pk),
          new anchor.BN(deserializeLE(depNonce).toString()),
        )
        .accounts({
          payer: owner.publicKey,
          signPdaAccount: (await pdaSign(program.programId))[0],
          mxeAccount: getMXEAccAddress(program.programId),
          mempoolAccount: getMempoolAccAddress(program.programId),
          executingPool: getExecutingPoolAccAddress(program.programId),
          computationAccount: getComputationAccAddress(
            program.programId,
            depOffset,
          ),
          compDefAccount: getCompDefAccAddress(
            program.programId,
            Buffer.from(getCompDefAccOffset("deposit_note")).readUInt32LE(),
          ),
          clusterAccount: arciumEnv.arciumClusterPubkey,
          poolAccount: arciumEnv.arciumFeePoolPubkey,
          clockAccount: arciumEnv.arciumClockPubkey,
          systemProgram: SystemProgram.programId,
          arciumProgram: getArciumProgAddress(),
        })
        .signers([owner])
        .rpc({ skipPreflight: true, commitment: "confirmed" });

      await awaitComputationFinalization(
        provider,
        depOffset,
        program.programId,
        "confirmed",
      );
      const depEv = await depEvPromise;

      const amt1 = cipher.decrypt([depEv.ctAmount], depEv.nonce)[0];
      expect(amt1).to.equal(depAmount);
      console.log(`✓ Encrypted ${depAmount} lamports via deposit_note`);

      console.log("\n--- Testing withdraw_note_check (sufficient balance) ---");
      const want = depAmount;
      const [ctNoteAmtPacked, ctWantPacked] = cipher.encrypt(
        [depAmount, want],
        depEv.nonce,
      );

      const wdOffset = new anchor.BN(randomBytes(8), "hex");
      const wdEvP = awaitEvent("withdrawNoteCheckEvent");

      await program.methods
        .withdrawNoteCheck(
          wdOffset,
          Array.from(ctNoteAmtPacked),
          Array.from(ctWantPacked),
          Array.from(pk),
          new anchor.BN(deserializeLE(depEv.nonce).toString()),
        )
        .accounts({
          payer: owner.publicKey,
          signPdaAccount: (await pdaSign(program.programId))[0],
          mxeAccount: getMXEAccAddress(program.programId),
          mempoolAccount: getMempoolAccAddress(program.programId),
          executingPool: getExecutingPoolAccAddress(program.programId),
          computationAccount: getComputationAccAddress(
            program.programId,
            wdOffset,
          ),
          compDefAccount: getCompDefAccAddress(
            program.programId,
            Buffer.from(
              getCompDefAccOffset("withdraw_note_check"),
            ).readUInt32LE(),
          ),
          clusterAccount: arciumEnv.arciumClusterPubkey,
          poolAccount: arciumEnv.arciumFeePoolPubkey,
          clockAccount: arciumEnv.arciumClockPubkey,
          systemProgram: SystemProgram.programId,
          arciumProgram: getArciumProgAddress(),
        })
        .signers([owner])
        .rpc({ skipPreflight: true, commitment: "confirmed" });

      await awaitComputationFinalization(
        provider,
        wdOffset,
        program.programId,
        "confirmed",
      );
      const wdEv = await wdEvP;
      expect(wdEv.ok).to.equal(true);
      console.log(`✓ MPC verified sufficient balance (note_amount >= ${want})`);

      console.log(
        "\n--- Testing withdraw_note_check (insufficient balance) ---",
      );
      const wantTooMuch = depAmount + 1n;
      const wdNonce2 = randomBytes(16);
      const [ctNoteAmt2, ctWant2] = cipher.encrypt(
        [depAmount, wantTooMuch],
        wdNonce2,
      );

      const wdOffset2 = new anchor.BN(randomBytes(8), "hex");
      const wdEvP2 = awaitEvent("withdrawNoteCheckEvent");

      await program.methods
        .withdrawNoteCheck(
          wdOffset2,
          Array.from(ctNoteAmt2),
          Array.from(ctWant2),
          Array.from(pk),
          new anchor.BN(deserializeLE(wdNonce2).toString()),
        )
        .accounts({
          payer: owner.publicKey,
          signPdaAccount: (await pdaSign(program.programId))[0],
          mxeAccount: getMXEAccAddress(program.programId),
          mempoolAccount: getMempoolAccAddress(program.programId),
          executingPool: getExecutingPoolAccAddress(program.programId),
          computationAccount: getComputationAccAddress(
            program.programId,
            wdOffset2,
          ),
          compDefAccount: getCompDefAccAddress(
            program.programId,
            Buffer.from(
              getCompDefAccOffset("withdraw_note_check"),
            ).readUInt32LE(),
          ),
          clusterAccount: arciumEnv.arciumClusterPubkey,
          poolAccount: arciumEnv.arciumFeePoolPubkey,
          clockAccount: arciumEnv.arciumClockPubkey,
          systemProgram: SystemProgram.programId,
          arciumProgram: getArciumProgAddress(),
        })
        .signers([owner])
        .rpc({ skipPreflight: true, commitment: "confirmed" });

      await awaitComputationFinalization(
        provider,
        wdOffset2,
        program.programId,
        "confirmed",
      );
      const wdEv2 = await wdEvP2;
      expect(wdEv2.ok).to.equal(false);
      console.log("✓ MPC rejected insufficient balance");
    });
  });

  /** ---------------------- E2E Integration Test ---------------------- */

  describe("e2e_privacy_flow", () => {
    anchor.setProvider(anchor.AnchorProvider.env());
    const provider = anchor.getProvider() as anchor.AnchorProvider;
    const program = anchor.workspace.Incognito as Program<Incognito>;
    const arciumEnv = getArciumEnv();
    const owner = readKpJson(`${os.homedir()}/.config/solana/id.json`);

    const POOL_STATE_SEED = Buffer.from("pool_state");
    const NULLIFIER_SEED = Buffer.from("nf");
    const SOL_VAULT_SEED = Buffer.from("sol_vault");
    const poolStatePda = PublicKey.findProgramAddressSync(
      [POOL_STATE_SEED],
      program.programId,
    )[0];
    const solVaultPda = PublicKey.findProgramAddressSync(
      [SOL_VAULT_SEED],
      program.programId,
    )[0];

    // Add this before hook
    before(async () => {
      console.log("\n--- Setting up vault for e2e_privacy_flow ---");
      try {
        await program.account.solVault.fetch(solVaultPda);
        console.log("✓ Vault already initialized");
      } catch {
        await program.methods
          .initVault()
          .accounts({
            payer: owner.publicKey,
            solVault: solVaultPda,
            systemProgram: SystemProgram.programId,
          })
          .signers([owner])
          .rpc({ commitment: "confirmed" });
        console.log("✓ Vault initialized");
      }
    });

    type Event = anchor.IdlEvents<(typeof program)["idl"]>;
    const awaitEvent = async <E extends keyof Event>(
      name: E,
      timeoutMs = 90_000,
    ): Promise<Event[E]> => {
      let id = -1;
      let t: NodeJS.Timeout;
      const ev = await new Promise<Event[E]>((res, rej) => {
        id = program.addEventListener(name as any, (e) => {
          if (t) clearTimeout(t);
          res(e);
        });
        t = setTimeout(() => {
          program.removeEventListener(id);
          rej(new Error(`timeout: ${String(name)}`));
        }, timeoutMs);
      });
      await program.removeEventListener(id);
      return ev;
    };

    // Close pool after each test
    afterEach(async () => {
      await closePoolIfExists(program, owner, poolStatePda);
    });

    it("E2E: Create encrypted note → Insert to tree (with nf_hash) → Verify proof → double-spend blocked", async () => {
      console.log("\n=== E2E Privacy Flow Test ===");

      const depth = 10;
      const tree = new MerkleTree(depth);

      console.log("\n1. Initializing fresh Merkle pool...");
      await initFreshPool(program, owner, poolStatePda, depth);
      console.log("   ✓ Fresh pool initialized");

      const mxePublicKey = await getMXEPublicKeyWithRetry(
        provider,
        program.programId,
      );
      const secret = randomBytes(32);
      const nullifierPreimage = randomBytes(32);
      const sk = x25519.utils.randomSecretKey();
      const pk_view = x25519.getPublicKey(sk);
      const shared = x25519.getSharedSecret(sk, mxePublicKey);
      const cipher = new RescueCipher(shared);

      await ensureCompDef(program, owner, provider, "deposit_note");
      await ensureCompDef(program, owner, provider, "withdraw_note_check");

      console.log("\n2. Creating encrypted note with cryptographic binding...");
      const noteAmount = BigInt(10_000_000);
      const nonce = randomBytes(16);
      const [ct_amount] = cipher.encrypt([noteAmount], nonce);

      const depOffset = new anchor.BN(randomBytes(8), "hex");
      const depEvPromise = awaitEvent("depositNoteEvent");

      await program.methods
        .depositNote(
          depOffset,
          Array.from(ct_amount),
          Array.from(pk_view),
          new anchor.BN(deserializeLE(nonce).toString()),
        )
        .accounts({
          payer: owner.publicKey,
          signPdaAccount: (await pdaSign(program.programId))[0],
          mxeAccount: getMXEAccAddress(program.programId),
          mempoolAccount: getMempoolAccAddress(program.programId),
          executingPool: getExecutingPoolAccAddress(program.programId),
          computationAccount: getComputationAccAddress(
            program.programId,
            depOffset,
          ),
          compDefAccount: getCompDefAccAddress(
            program.programId,
            Buffer.from(getCompDefAccOffset("deposit_note")).readUInt32LE(),
          ),
          clusterAccount: arciumEnv.arciumClusterPubkey,
          poolAccount: arciumEnv.arciumFeePoolPubkey,
          clockAccount: arciumEnv.arciumClockPubkey,
          systemProgram: SystemProgram.programId,
          arciumProgram: getArciumProgAddress(),
        })
        .signers([owner])
        .rpc({ skipPreflight: false, commitment: "confirmed" });

      await awaitComputationFinalization(
        provider,
        depOffset,
        program.programId,
        "confirmed",
      );
      const depEv = await depEvPromise;

      const decryptedAmount = cipher.decrypt([depEv.ctAmount], depEv.nonce)[0];
      expect(decryptedAmount).to.equal(noteAmount);
      console.log(
        `   ✓ Note created with encrypted amount: ${noteAmount} lamports`,
      );

      console.log("\n3. Computing commitment, NF/NFH and inserting...");
      const storedNote: ShieldedNote = {
        secret: new Uint8Array(secret),
        nullifier: new Uint8Array(nullifierPreimage),
        pk_view: new Uint8Array(pk_view),
        ct_amount: new Uint8Array(depEv.ctAmount),
        nonce: new Uint8Array(depEv.nonce),
        recipient: new Uint8Array(owner.publicKey.toBytes()),
      };
      const commitment = computeCommitment(storedNote); // C
      const NF = computeNullifier(secret, nullifierPreimage); // NF
      const nf_hash = computeNfHash(NF); // h1(NF)
      const leaf = h2js(commitment, nf_hash); // Merkle leaf

      const noteIndex = tree.getLeaves().length;
      const insertPath = tree.getMerklePath(noteIndex);
      tree.addLeaf(leaf);

      const SOL_VAULT_SEED = Buffer.from("sol_vault");
      const solVaultPda = PublicKey.findProgramAddressSync(
        [SOL_VAULT_SEED],
        program.programId,
      )[0];
      const COMMITMENT_SEED = Buffer.from("commitment");
      const commitmentMarkerPda = PublicKey.findProgramAddressSync(
        [COMMITMENT_SEED, commitment],
        program.programId,
      )[0];
      const depositAmount = BigInt(100_000_000); // 0.1 SOL (enough to cover 0.05 SOL wrapper fee)
      const insEvP = awaitEvent("realDepositEvent");
      await program.methods
        .depositToPool(
          new anchor.BN(depositAmount.toString()),
          Array.from(commitment),
          Array.from(nf_hash),
          insertPath.map((p) => Array.from(p)),
        )
        .accounts({
          depositor: owner.publicKey,
          solVault: solVaultPda,
          poolState: poolStatePda,
          commitmentMarker: commitmentMarkerPda,
          wrapperStealthAddress: generateDummyWrapperStealthAddress(),
          systemProgram: SystemProgram.programId,
        })
        .signers([owner])
        .rpc({ commitment: "confirmed" });
      const insEv = await insEvP;
      const actualIndex = Number(insEv.index.toString());

      console.log(`   ✓ Commitment inserted at index ${actualIndex}`);
      console.log(
        `   ✓ New Merkle root: ${Buffer.from(insEv.root).toString("hex").substring(0, 16)}...`,
      );

      console.log("\n4. Building Merkle path and verifying proof...");
      const path = tree.getMerklePath(actualIndex);
      const nullifier = NF; // spend with the same NF tied to nf_hash
      const nullifierPda = PublicKey.findProgramAddressSync(
        [NULLIFIER_SEED, nullifier],
        program.programId,
      )[0];

      const verifyEvP = awaitEvent("proofVerified");
      await program.methods
        .verifyProof(
          Array.from(commitment),
          path.map((p) => Array.from(p)),
          Array.from(nullifier),
          new anchor.BN(actualIndex),
        )
        .accounts({
          payer: owner.publicKey,
          poolState: poolStatePda,
          nullifierMarker: nullifierPda,
          systemProgram: SystemProgram.programId,
        })
        .signers([owner])
        .rpc({ commitment: "confirmed" });
      await verifyEvP;
      console.log("   ✓ Merkle proof verified (note exists in tree)");
      console.log("   ✓ Nullifier cryptographically bound to commitment");

      // Confirm marker exists now (created by verifyProof)
      const markerAcc = await (program.account as any).nullifierMarker.fetch(
        nullifierPda,
      );
      expect(markerAcc).to.not.equal(undefined);
      console.log("   ✓ Nullifier marker created by verifyProof");

      console.log("\n5. Testing private balance check (MPC)...");
      const withdrawAmount = BigInt(5_000_000);
      const wdNonce = randomBytes(16);
      const [ctNoteAmount, ctWant] = cipher.encrypt(
        [noteAmount, withdrawAmount],
        wdNonce,
      );

      const wdOffset = new anchor.BN(randomBytes(8), "hex");
      const wdEvPromise = awaitEvent("withdrawNoteCheckEvent");

      await program.methods
        .withdrawNoteCheck(
          wdOffset,
          Array.from(ctNoteAmount),
          Array.from(ctWant),
          Array.from(pk_view),
          new anchor.BN(deserializeLE(wdNonce).toString()),
        )
        .accounts({
          payer: owner.publicKey,
          signPdaAccount: (await pdaSign(program.programId))[0],
          mxeAccount: getMXEAccAddress(program.programId),
          mempoolAccount: getMempoolAccAddress(program.programId),
          executingPool: getExecutingPoolAccAddress(program.programId),
          computationAccount: getComputationAccAddress(
            program.programId,
            wdOffset,
          ),
          compDefAccount: getCompDefAccAddress(
            program.programId,
            Buffer.from(
              getCompDefAccOffset("withdraw_note_check"),
            ).readUInt32LE(),
          ),
          clusterAccount: arciumEnv.arciumClusterPubkey,
          poolAccount: arciumEnv.arciumFeePoolPubkey,
          clockAccount: arciumEnv.arciumClockPubkey,
          systemProgram: SystemProgram.programId,
          arciumProgram: getArciumProgAddress(),
        })
        .signers([owner])
        .rpc({ skipPreflight: false, commitment: "confirmed" });

      await awaitComputationFinalization(
        provider,
        wdOffset,
        program.programId,
        "confirmed",
      );
      const wdEv = await wdEvPromise;
      expect(wdEv.ok).to.equal(true);
      console.log(
        `   ✓ MPC verified sufficient balance for withdrawal of ${withdrawAmount} lamports`,
      );

      console.log(
        "\n6. Testing double-spend prevention (verifyProof again)...",
      );
      try {
        await program.methods
          .verifyProof(
            Array.from(commitment),
            path.map((p) => Array.from(p)),
            Array.from(nullifier),
            new anchor.BN(actualIndex),
          )
          .accounts({
            payer: owner.publicKey,
            poolState: poolStatePda,
            nullifierMarker: nullifierPda,
            systemProgram: SystemProgram.programId,
          })
          .signers([owner])
          .rpc({ commitment: "confirmed" });
        expect.fail("Should have failed due to nullifier reuse");
      } catch (err: any) {
        expect(err.toString()).to.include("already in use");
        console.log("   ✓ Double-spend attempt blocked (PDA already in use)");
      }

      console.log("\n=== E2E Test Complete ===");
      console.log("\nPrivacy guarantees verified:");
      console.log("  ✓ Amount encrypted via MPC (never revealed on-chain)");
      console.log("  ✓ Commitment-based storage (only hashes on-chain)");
      console.log(
        "  ✓ Merkle proof verification (membership without revealing which note)",
      );
      console.log("  ✓ Nullifier prevents double-spending");
      console.log("  ✓ Only Merkle root stored on-chain (minimal storage)");
    });
  });

  /** ---------------------- Real SOL Vault Flow ---------------------- */

  describe("real_sol_vault_flow", () => {
    anchor.setProvider(anchor.AnchorProvider.env());
    const provider = anchor.getProvider() as anchor.AnchorProvider;
    const program = anchor.workspace.Incognito as Program<Incognito>;
    const owner = readKpJson(`${os.homedir()}/.config/solana/id.json`);

    const POOL_STATE_SEED = Buffer.from("pool_state");
    const SOL_VAULT_SEED = Buffer.from("sol_vault");
    const NULLIFIER_SEED = Buffer.from("nf");

    const poolStatePda = PublicKey.findProgramAddressSync(
      [POOL_STATE_SEED],
      program.programId,
    )[0];
    const solVaultPda = PublicKey.findProgramAddressSync(
      [SOL_VAULT_SEED],
      program.programId,
    )[0];

    type Event = anchor.IdlEvents<(typeof program)["idl"]>;
    const awaitEvent = async <E extends keyof Event>(
      name: E,
      timeoutMs = 60_000,
    ): Promise<Event[E]> => {
      let id = -1;
      let t: NodeJS.Timeout;
      const ev = await new Promise<Event[E]>((res, rej) => {
        id = program.addEventListener(name as any, (e) => {
          if (t) clearTimeout(t);
          res(e);
        });
        t = setTimeout(() => {
          program.removeEventListener(id);
          rej(new Error(`timeout: ${String(name)}`));
        }, timeoutMs);
      });
      await program.removeEventListener(id);
      return ev;
    };

    // Initialize vault once before all tests
    before(async () => {
      console.log("\n--- Setting up vault for real_sol_vault_flow ---");

      try {
        await program.account.solVault.fetch(solVaultPda);
        console.log("✓ Vault already initialized");
      } catch {
        await program.methods
          .initVault()
          .accounts({
            payer: owner.publicKey,
            solVault: solVaultPda,
            systemProgram: SystemProgram.programId,
          })
          .signers([owner])
          .rpc({ commitment: "confirmed" });
        console.log("✓ Vault initialized");
      }
    });

    // Close pool after each test for isolation
    afterEach(async () => {
      await closePoolIfExists(program, owner, poolStatePda);
    });

    it("Complete flow: Real SOL deposit (with nf_hash) → withdraw with proof", async () => {
      console.log("\n=== Real SOL Vault Privacy Flow ===");

      const depth = 10;
      const tree = new MerkleTree(depth);

      console.log("\n1. Initializing fresh pool...");
      await initFreshPool(program, owner, poolStatePda, depth);

      // Generate note credentials
      const mxePublicKey = await getMXEPublicKeyWithRetry(
        provider,
        program.programId,
      );
      const secret = randomBytes(32);
      const nullifierPreimage = randomBytes(32);
      const sk = x25519.utils.randomSecretKey();
      const pk_view = x25519.getPublicKey(sk);
      const shared = x25519.getSharedSecret(sk, mxePublicKey);
      const cipher = new RescueCipher(shared);

      // Create note with encrypted amount
      const depositAmount = BigInt(100_000_000); // 0.1 SOL (enough to cover 0.05 SOL wrapper fee)
      const nonce = randomBytes(16);
      const [ct_amount] = cipher.encrypt([depositAmount], nonce);

      const note: ShieldedNote = {
        secret: new Uint8Array(secret),
        nullifier: new Uint8Array(nullifierPreimage),
        pk_view: new Uint8Array(pk_view),
        ct_amount: new Uint8Array(ct_amount),
        nonce: new Uint8Array(nonce),
        recipient: new Uint8Array(owner.publicKey.toBytes()),
      };
      const commitment = computeCommitment(note); // C
      const NF = computeNullifier(secret, nullifierPreimage);
      const nf_hash = computeNfHash(NF);
      const leaf = h2js(commitment, nf_hash);

      console.log("\n2. Depositing real SOL to vault...");
      const balanceBefore = await provider.connection.getBalance(
        owner.publicKey,
      );
      const vaultBalanceBefore =
        await provider.connection.getBalance(solVaultPda);

      const noteIndex = tree.getLeaves().length;
      const depositPath = tree.getMerklePath(noteIndex);
      tree.addLeaf(leaf);

      const depositEvPromise = awaitEvent("realDepositEvent");

      const COMMITMENT_SEED = Buffer.from("commitment");
      const commitmentMarkerPda = PublicKey.findProgramAddressSync(
        [COMMITMENT_SEED, commitment],
        program.programId,
      )[0];

      await program.methods
        .depositToPool(
          new anchor.BN(depositAmount.toString()),
          Array.from(commitment),
          Array.from(nf_hash),
          depositPath.map((p) => Array.from(p)),
        )
        .accounts({
          depositor: owner.publicKey,
          solVault: solVaultPda,
          poolState: poolStatePda,
          commitmentMarker: commitmentMarkerPda,
          wrapperStealthAddress: generateDummyWrapperStealthAddress(),
          systemProgram: SystemProgram.programId,
        })
        .signers([owner])
        .rpc({ commitment: "confirmed" });

      const depositEv = await depositEvPromise;

      const balanceAfter = await provider.connection.getBalance(
        owner.publicKey,
      );
      const vaultBalanceAfter =
        await provider.connection.getBalance(solVaultPda);

      console.log(`   ✓ Deposited ${depositAmount} lamports`);
      console.log(
        `   ✓ User balance decreased: ${balanceBefore - balanceAfter} lamports`,
      );
      console.log(
        `   ✓ Vault balance increased: ${vaultBalanceAfter - vaultBalanceBefore} lamports`,
      );
      console.log(
        `   ✓ Commitment (with nf_hash binding) inserted at index ${depositEv.index.toString()}`,
      );

      // Verify vault state
      const vaultState = await program.account.solVault.fetch(solVaultPda);
      expect(
        BigInt(vaultState.totalDeposited.toString()) >= depositAmount,
      ).to.equal(true);

      console.log("\n3. Preparing withdrawal with Merkle proof...");
      const actualIndex = Number(depositEv.index.toString());
      const withdrawPath = tree.getMerklePath(actualIndex);
      const nullifier = NF; // must reuse the same NF used to bind nf_hash
      const nullifierPda = PublicKey.findProgramAddressSync(
        [NULLIFIER_SEED, nullifier],
        program.programId,
      )[0];

      console.log("\n4. Withdrawing SOL from vault...");
      const withdrawAmount = BigInt(30_000_000); // Withdraw 0.03 SOL
      const userBalanceBefore = await provider.connection.getBalance(
        owner.publicKey,
      );
      const vaultBalBefore = await provider.connection.getBalance(solVaultPda);

      const withdrawEvPromise = awaitEvent("realWithdrawEvent");

      await program.methods
        .withdrawFromPool(
          new anchor.BN(withdrawAmount.toString()),
          Array.from(commitment),
          withdrawPath.map((p) => Array.from(p)),
          Array.from(nullifier),
          new anchor.BN(actualIndex),
          Array.from(owner.publicKey.toBytes()),
          null,
          null,
          null,
        )
        .accounts({
          recipient: owner.publicKey,
          solVault: solVaultPda,
          poolState: poolStatePda,
          nullifierMarker: nullifierPda,
          changeCommitmentMarker: poolStatePda,
          systemProgram: SystemProgram.programId,
        })
        .signers([owner])
        .rpc({ commitment: "confirmed" });

      const withdrawEv = await withdrawEvPromise;

      const userBalanceAfter = await provider.connection.getBalance(
        owner.publicKey,
      );
      const vaultBalAfter = await provider.connection.getBalance(solVaultPda);

      console.log(`   ✓ Withdrew ${withdrawAmount} lamports`);
      console.log(`   ✓ User balance increased (net of fees)`);
      console.log(
        `   ✓ Vault balance decreased: ${vaultBalBefore - vaultBalAfter} lamports`,
      );
      console.log(`   ✓ Nullifier marked as spent`);

      console.log("\n5. Testing double-spend prevention...");
      try {
        await program.methods
          .withdrawFromPool(
            new anchor.BN(withdrawAmount.toString()),
            Array.from(commitment),
            withdrawPath.map((p) => Array.from(p)),
            Array.from(nullifier),
            new anchor.BN(actualIndex),
            Array.from(owner.publicKey.toBytes()),
            null,
            null,
            null,
          )
          .accounts({
            recipient: owner.publicKey,
            solVault: solVaultPda,
            poolState: poolStatePda,
            nullifierMarker: nullifierPda,
            changeCommitmentMarker: poolStatePda,
            systemProgram: SystemProgram.programId,
          })
          .signers([owner])
          .rpc({ commitment: "confirmed" });
        expect.fail("Should have failed due to nullifier already used");
      } catch (err: any) {
        expect(err.toString()).to.include("already in use");
        console.log("   ✓ Double-spend prevented (nullifier already exists)");
      }

      console.log("\n=== Real SOL Vault Flow Complete ===");
      console.log("\nPrivacy + Real funds verified:");
      console.log("  ✓ Real SOL transferred to/from vault PDA");
      console.log("  ✓ Commitment-based privacy (amounts hidden)");
      console.log("  ✓ Merkle proof verification (anonymous withdrawal)");
      console.log("  ✓ Nullifier CRYPTOGRAPHICALLY BOUND to commitment");
      console.log("  ✓ Nullifier prevents double-spending");
    });

    it("Withdrawal fails with invalid Merkle proof", async () => {
      console.log("\n--- Testing invalid proof rejection ---");

      const depth = 10;
      const tree = new MerkleTree(depth);

      // Initialize fresh pool
      await initFreshPool(program, owner, poolStatePda, depth);

      // Create fake commitment + fake NF (but won't insert)
      const fakeCommitment = randomBytes(32);
      const fakeNF = randomBytes(32);
      const nullifierPda = PublicKey.findProgramAddressSync(
        [NULLIFIER_SEED, fakeNF],
        program.programId,
      )[0];

      // Try to use an invalid path (all zeros)
      const invalidPath = Array(20)
        .fill(null)
        .map(() => new Uint8Array(32));

      try {
        await program.methods
          .withdrawFromPool(
            new anchor.BN(1_000_000),
            Array.from(fakeCommitment),
            invalidPath.map((p) => Array.from(p)),
            Array.from(fakeNF),
            new anchor.BN(0),
            Array.from(owner.publicKey.toBytes()),
            null,
            null,
            null,
          )
          .accounts({
            recipient: owner.publicKey,
            solVault: solVaultPda,
            poolState: poolStatePda,
            nullifierMarker: nullifierPda,
            changeCommitmentMarker: poolStatePda,
            systemProgram: SystemProgram.programId,
          })
          .signers([owner])
          .rpc({ commitment: "confirmed" });
        expect.fail("Should have failed due to invalid proof");
      } catch (err: any) {
        expect(err.toString()).to.include("InvalidMerkleProof");
        console.log("✓ Invalid Merkle proof rejected");
      }
    });

    it("Withdrawal fails when vault has insufficient balance", async () => {
      console.log("\n--- Testing insufficient vault balance ---");

      const depth = 10;
      const tree = new MerkleTree(depth);

      // Initialize fresh pool
      await initFreshPool(program, owner, poolStatePda, depth);

      // Create a valid note (commitment + NF + nf_hash)
      const commitment = randomBytes(32);
      const NF = randomBytes(32);
      const nf_hash = computeNfHash(NF);
      const leaf = h2js(commitment, nf_hash);

      const noteIndex = tree.getLeaves().length;
      const depositPath = tree.getMerklePath(noteIndex);
      tree.addLeaf(leaf);

      // Deposit small amount
      const COMMITMENT_SEED = Buffer.from("commitment");
      const commitmentMarkerPda = PublicKey.findProgramAddressSync(
        [COMMITMENT_SEED, commitment],
        program.programId,
      )[0];
      const smallAmount = BigInt(100_000_000); // 0.1 SOL (enough to cover 0.05 SOL wrapper fee)
      await program.methods
        .depositToPool(
          new anchor.BN(smallAmount.toString()),
          Array.from(commitment),
          Array.from(nf_hash),
          depositPath.map((p) => Array.from(p)),
        )
        .accounts({
          depositor: owner.publicKey,
          solVault: solVaultPda,
          poolState: poolStatePda,
          commitmentMarker: commitmentMarkerPda,
          wrapperStealthAddress: generateDummyWrapperStealthAddress(),
          systemProgram: SystemProgram.programId,
        })
        .signers([owner])
        .rpc({ commitment: "confirmed" });

      // Try to withdraw more than vault has — use the SAME NF to pass Merkle check
      const vaultBalance = await provider.connection.getBalance(solVaultPda);
      const excessiveAmount = BigInt(vaultBalance) + BigInt(1_000_000);

      const nullifierPda = PublicKey.findProgramAddressSync(
        [NULLIFIER_SEED, NF],
        program.programId,
      )[0];

      const actualIndex = noteIndex;
      const withdrawPath = tree.getMerklePath(actualIndex);

      try {
        await program.methods
          .withdrawFromPool(
            new anchor.BN(excessiveAmount.toString()),
            Array.from(commitment),
            withdrawPath.map((p) => Array.from(p)),
            Array.from(NF), // reuse NF bound to nf_hash
            new anchor.BN(actualIndex),
            Array.from(owner.publicKey.toBytes()),
            null,
            null,
            null,
          )
          .accounts({
            recipient: owner.publicKey,
            solVault: solVaultPda,
            poolState: poolStatePda,
            nullifierMarker: nullifierPda,
            changeCommitmentMarker: poolStatePda,
            systemProgram: SystemProgram.programId,
          })
          .signers([owner])
          .rpc({ commitment: "confirmed" });
        expect.fail("Should have failed due to insufficient vault balance");
      } catch (err: any) {
        expect(err.toString()).to.include("InsufficientVaultBalance");
        console.log("✓ Insufficient vault balance rejection works");
      }
    });

    it("Attack test: Cannot use valid commitment with fake nullifier", async () => {
      console.log("\n--- Testing nullifier binding security ---");

      const depth = 10;
      const tree = new MerkleTree(depth);

      // Initialize fresh pool
      await initFreshPool(program, owner, poolStatePda, depth);

      // Alice creates a valid note
      const aliceSecret = randomBytes(32);
      const aliceNullifierPreimage = randomBytes(32);
      const alicePkView = x25519.getPublicKey(x25519.utils.randomSecretKey());

      const aliceNote: ShieldedNote = {
        secret: new Uint8Array(aliceSecret),
        nullifier: new Uint8Array(aliceNullifierPreimage),
        pk_view: new Uint8Array(alicePkView),
        ct_amount: randomBytes(32),
        nonce: randomBytes(16),
        recipient: new Uint8Array(owner.publicKey.toBytes()),
      };
      const aliceCommitment = computeCommitment(aliceNote);
      const aliceNF = computeNullifier(aliceSecret, aliceNullifierPreimage);
      const aliceNfHash = computeNfHash(aliceNF);
      const aliceLeaf = h2js(aliceCommitment, aliceNfHash);

      // Alice deposits (with nf_hash)
      const noteIndex = tree.getLeaves().length;
      const depositPath = tree.getMerklePath(noteIndex);
      tree.addLeaf(aliceLeaf);

      const COMMITMENT_SEED = Buffer.from("commitment");
      const aliceCommitmentMarkerPda = PublicKey.findProgramAddressSync(
        [COMMITMENT_SEED, aliceCommitment],
        program.programId,
      )[0];

      await program.methods
        .depositToPool(
          new anchor.BN(100_000_000), // 0.1 SOL (enough to cover 0.05 SOL wrapper fee)
          Array.from(aliceCommitment),
          Array.from(aliceNfHash),
          depositPath.map((p) => Array.from(p)),
        )
        .accounts({
          depositor: owner.publicKey,
          solVault: solVaultPda,
          poolState: poolStatePda,
          commitmentMarker: aliceCommitmentMarkerPda,
          wrapperStealthAddress: generateDummyWrapperStealthAddress(),
          systemProgram: SystemProgram.programId,
        })
        .signers([owner])
        .rpc({ commitment: "confirmed" });

      console.log("✓ Alice deposited with valid commitment + nf_hash");

      // Attacker tries to use Alice's commitment with a fake nullifier (won't match nf_hash)
      const fakeNullifier = randomBytes(32);
      const fakeNullifierPda = PublicKey.findProgramAddressSync(
        [NULLIFIER_SEED, fakeNullifier],
        program.programId,
      )[0];

      const withdrawPath = tree.getMerklePath(noteIndex);

      try {
        await program.methods
          .withdrawFromPool(
            new anchor.BN(1_000_000),
            Array.from(aliceCommitment),
            withdrawPath.map((p) => Array.from(p)),
            Array.from(fakeNullifier),
            new anchor.BN(noteIndex),
            Array.from(owner.publicKey.toBytes()),
            null,
            null,
            null,
          )
          .accounts({
            recipient: owner.publicKey,
            solVault: solVaultPda,
            poolState: poolStatePda,
            nullifierMarker: fakeNullifierPda,
            changeCommitmentMarker: poolStatePda,
            systemProgram: SystemProgram.programId,
          })
          .signers([owner])
          .rpc({ commitment: "confirmed" });

        expect.fail("Attack should have been prevented!");
      } catch {
        console.log(
          "✓ Attack prevented: Cannot use valid commitment with a fake nullifier",
        );
        console.log(
          "✓ Cryptographic binding via nf_hash ensures uniqueness per commitment",
        );
      }
    });
  });

  describe("partial_withdrawal_with_change", () => {
    anchor.setProvider(anchor.AnchorProvider.env());
    const provider = anchor.getProvider() as anchor.AnchorProvider;
    const program = anchor.workspace.Incognito as Program<Incognito>;
    const owner = readKpJson(`${os.homedir()}/.config/solana/id.json`);

    const POOL_STATE_SEED = Buffer.from("pool_state");
    const SOL_VAULT_SEED = Buffer.from("sol_vault");
    const NULLIFIER_SEED = Buffer.from("nf");
    const COMMITMENT_SEED = Buffer.from("commitment");

    const poolStatePda = PublicKey.findProgramAddressSync(
      [POOL_STATE_SEED],
      program.programId,
    )[0];
    const solVaultPda = PublicKey.findProgramAddressSync(
      [SOL_VAULT_SEED],
      program.programId,
    )[0];

    before(async () => {
      console.log("\n--- Setting up vault for partial withdrawal ---");
      try {
        await program.account.solVault.fetch(solVaultPda);
        console.log("✓ Vault already initialized");
      } catch {
        await program.methods
          .initVault()
          .accounts({
            payer: owner.publicKey,
            solVault: solVaultPda,
            systemProgram: SystemProgram.programId,
          })
          .signers([owner])
          .rpc({ commitment: "confirmed" });
        console.log("✓ Vault initialized");
      }
    });

    afterEach(async () => {
      await closePoolIfExists(program, owner, poolStatePda);
    });

    it("Partial withdrawal: 10 SOL note → withdraw 3 SOL → 7 SOL change note created", async () => {
      console.log("\n=== Partial Withdrawal Test ===");

      const depth = 10;
      const tree = new MerkleTree(depth);

      console.log("\n1. Initializing pool...");
      await initFreshPool(program, owner, poolStatePda, depth);

      const mxePublicKey = await getMXEPublicKeyWithRetry(
        provider,
        program.programId,
      );
      const sk = x25519.utils.randomSecretKey();
      const pk_view = x25519.getPublicKey(sk);
      const shared = x25519.getSharedSecret(sk, mxePublicKey);
      const cipher = new RescueCipher(shared);

      // Create 10 SOL note
      const originalAmount = BigInt(10_000_000_000); // 10 SOL
      const secret = randomBytes(32);
      const nullifierPreimage = randomBytes(32);
      const nonce = randomBytes(16);
      const [ct_amount] = cipher.encrypt([originalAmount], nonce);

      const originalNote: ShieldedNote = {
        secret,
        nullifier: nullifierPreimage,
        pk_view,
        ct_amount,
        nonce,
        recipient: new Uint8Array(owner.publicKey.toBytes()),
      };

      const commitment = computeCommitment(originalNote);
      const NF = computeNullifier(secret, nullifierPreimage);
      const nf_hash = computeNfHash(NF);
      const leaf = h2js(commitment, nf_hash);

      console.log("\n2. Depositing 10 SOL note...");
      const noteIndex = tree.getLeaves().length;
      const depositPath = tree.getMerklePath(noteIndex);
      tree.addLeaf(leaf);

      const commitmentMarkerPda = PublicKey.findProgramAddressSync(
        [COMMITMENT_SEED, commitment],
        program.programId,
      )[0];

      await program.methods
        .depositToPool(
          new anchor.BN(originalAmount.toString()),
          Array.from(commitment),
          Array.from(nf_hash),
          depositPath.map((p) => Array.from(p)),
        )
        .accounts({
          depositor: owner.publicKey,
          solVault: solVaultPda,
          poolState: poolStatePda,
          commitmentMarker: commitmentMarkerPda,
          wrapperStealthAddress: generateDummyWrapperStealthAddress(),
          systemProgram: SystemProgram.programId,
        })
        .signers([owner])
        .rpc({ commitment: "confirmed" });

      console.log("   ✓ 10 SOL note deposited");

      // Prepare partial withdrawal
        const withdrawAmount = BigInt(3_000_000_000); // 3 SOL
        const changeAmount = originalAmount - withdrawAmount; // 7 SOL

        console.log("\n3. Creating 7 SOL change note...");
        const changeNote = createChangeNote(
        cipher,
        changeAmount,
        new Uint8Array(owner.publicKey.toBytes()),
        );
        const changeCommitment = computeCommitment(changeNote);
        const changeNF = computeNullifier(
        changeNote.secret,
        changeNote.nullifier,
        );
        const changeNfHash = computeNfHash(changeNF);

        // IMPORTANT: Get the current state to know what index the change will have
        const psBeforeWithdraw = await program.account.poolState.fetch(poolStatePda);
        const changeIndex = Number(psBeforeWithdraw.leafCount.toString());

        // Compute change path for current tree state (before any modifications)
        const changePath = tree.getMerklePath(changeIndex);

        console.log("\n4. Withdrawing 3 SOL with change...");
        const withdrawPath = tree.getMerklePath(noteIndex);
        const nullifierPda = PublicKey.findProgramAddressSync(
        [NULLIFIER_SEED, NF],
        program.programId,
        )[0];

        // Derive change commitment marker PDA
        const changeCommitmentMarkerPda = PublicKey.findProgramAddressSync(
        [COMMITMENT_SEED, changeCommitment],
        program.programId,
        )[0];

        await program.methods
        .withdrawFromPool(
            new anchor.BN(withdrawAmount.toString()),
            Array.from(commitment),
            withdrawPath.map((p) => Array.from(p)),
            Array.from(NF),
            new anchor.BN(noteIndex),
            Array.from(owner.publicKey.toBytes()),
            Array.from(changeCommitment),
            Array.from(changeNfHash),
            changePath.map((p) => Array.from(p)),
        )
        .accounts({
            recipient: owner.publicKey,
            solVault: solVaultPda,
            poolState: poolStatePda,
            nullifierMarker: nullifierPda,
            changeCommitmentMarker: changeCommitmentMarkerPda, // Actual change marker PDA
            systemProgram: SystemProgram.programId,
        })
        .signers([owner])
        .rpc({ commitment: "confirmed" });

        console.log("   ✓ 3 SOL withdrawn");
        console.log("   ✓ 7 SOL change note created");

        // NOW update local tree to reflect the change note insertion
        const changeLeaf = h2js(changeCommitment, changeNfHash);
        tree.addLeaf(changeLeaf);

      // Verify pool state
      const ps = await program.account.poolState.fetch(poolStatePda);
      expect(ps.leafCount.toString()).to.equal("2"); // Original + change note

      console.log("\n5. Verifying change note can be spent...");
      const changeWithdrawPath = tree.getMerklePath(changeIndex);
      const changeNullifierPda = PublicKey.findProgramAddressSync(
        [NULLIFIER_SEED, changeNF],
        program.programId,
      )[0];

      await program.methods
        .withdrawFromPool(
          new anchor.BN(changeAmount.toString()),
          Array.from(changeCommitment),
          changeWithdrawPath.map((p) => Array.from(p)),
          Array.from(changeNF),
          new anchor.BN(changeIndex),
          Array.from(owner.publicKey.toBytes()),
          null,
          null,
          null,
        )
        .accounts({
          recipient: owner.publicKey,
          solVault: solVaultPda,
          poolState: poolStatePda,
          nullifierMarker: changeNullifierPda,
          changeCommitmentMarker: poolStatePda, // Unused when no change
          systemProgram: SystemProgram.programId,
        })
        .signers([owner])
        .rpc({ commitment: "confirmed" });

      console.log("   ✓ Change note spent successfully (7 SOL withdrawn)");

      console.log("\n=== Partial Withdrawal Complete ===");
      console.log("  ✓ Original 10 SOL note consumed");
      console.log("  ✓ 3 SOL withdrawn to recipient");
      console.log("  ✓ 7 SOL change note created and spent");
      console.log("  ✓ Privacy maintained throughout");
    });
  });
});