import { PublicKey } from "@solana/web3.js";

// Program ID (le tien)
const programId = new PublicKey("5U8nyhH1iAEvsZBkraQRd5ft5UsrzZbMgrRnPBbqf2Mc");

// ⚠️ Remplace par le même seed de 32 octets que tu avais utilisé à l'initTree
// Exemple: si tu avais mis Buffer.alloc(32, 1), alors:
const seed = Buffer.alloc(32, 1);

const [treePda] = PublicKey.findProgramAddressSync(
  [Buffer.from("merkle"), seed],
  programId
);

console.log("Tree PDA:", treePda.toBase58());
