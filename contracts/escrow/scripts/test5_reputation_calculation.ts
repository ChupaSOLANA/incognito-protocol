#!/usr/bin/env ts-node

// Test 5: Reputation score calculation - Fetch and verify reputation scores

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey } from "@solana/web3.js";
import { Escrow } from "../target/types/escrow";
import { initializeTestAccounts, outputResult } from "./common";

async function main() {
  try {
    // Configure the client
    anchor.setProvider(anchor.AnchorProvider.env());
    const program = anchor.workspace.Escrow as Program<Escrow>;
    const provider = anchor.getProvider() as anchor.AnchorProvider;

    // Initialize test accounts
    const accounts = await initializeTestAccounts(program, provider);

    const [buyerRepPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("reputation"), accounts.buyer.publicKey.toBuffer()],
      program.programId
    );

    const [sellerRepPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("reputation"), accounts.seller.publicKey.toBuffer()],
      program.programId
    );

    // Fetch reputations
    let buyerRep, sellerRep;

    try {
      buyerRep = await program.account.userReputation.fetch(buyerRepPDA);
    } catch (error) {
      console.log("Buyer reputation not found (not initialized yet)");
      buyerRep = null;
    }

    try {
      sellerRep = await program.account.userReputation.fetch(sellerRepPDA);
    } catch (error) {
      console.log("Seller reputation not found (not initialized yet)");
      sellerRep = null;
    }

    const calculateExpectedScore = (
      total: number,
      successful: number,
      won: number,
      lost: number
    ): number => {
      if (total === 0) {
        return 500;
      }
      const successRate = (successful * 100) / total;
      const penalty = lost * 50;
      const bonus = won * 10;
      return Math.min(1000, Math.max(0, successRate + bonus - penalty));
    };

    const formatReputation = (rep: any) => {
      if (!rep) {
        return {
          initialized: false,
          note: "Reputation not initialized yet",
        };
      }

      const total = rep.totalOrders.toNumber();
      const successful = rep.successfulOrders.toNumber();
      const won = rep.disputesWon.toNumber();
      const lost = rep.disputesLost.toNumber();
      const score = rep.reputationScore.toNumber();
      const expectedScore = calculateExpectedScore(total, successful, won, lost);

      return {
        initialized: true,
        totalOrders: total,
        successfulOrders: successful,
        disputesWon: won,
        disputesLost: lost,
        reputationScore: score,
        expectedScore: expectedScore,
        scoreMatch: Math.abs(score - expectedScore) <= 1,
      };
    };

    const buyerRepData = formatReputation(buyerRep);
    const sellerRepData = formatReputation(sellerRep);

    console.log("\nBuyer reputation:", JSON.stringify(buyerRepData, null, 2));
    console.log("\nSeller reputation:", JSON.stringify(sellerRepData, null, 2));

    outputResult(true, {
      buyer: {
        address: accounts.buyer.publicKey.toBase58(),
        reputation: buyerRepData,
      },
      seller: {
        address: accounts.seller.publicKey.toBase58(),
        reputation: sellerRepData,
      },
      note: "Run dispute tests first to see reputation changes",
    });
  } catch (error) {
    console.error("Error:", error);
    outputResult(false, { error: error.message });
  }
}

main();
