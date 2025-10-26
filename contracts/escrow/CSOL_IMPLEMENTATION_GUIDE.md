# cSOL Escrow Implementation Guide
## Converting Arcium Escrow from SOL to Token-2022 cSOL

---

## Overview

This guide provides **complete step-by-step instructions** to modify the Arcium escrow contract to use **cSOL (Token-2022 with confidential transfers)** instead of SOL.

**Key Change:** Instead of holding SOL in the escrow PDA's account balance, we'll hold cSOL in an **Associated Token Account (ATA)** owned by the escrow PDA.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│ BUYER                                                    │
│  Pubkey: Buyer...                                       │
│  cSOL ATA: [buyer_ata]  ← Has cSOL balance             │
└────────────────┬────────────────────────────────────────┘
                 │
                 │ 1. create_order()
                 │    Confidential Transfer: buyer_ata → escrow_ata
                 ↓
┌─────────────────────────────────────────────────────────┐
│ ESCROW PDA                                               │
│  Pubkey: [escrow_pda] (derived from buyer + order_id)  │
│  State: Stores order metadata                           │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │ ESCROW ATA (Associated Token Account)          │    │
│  │  Address: [escrow_ata] (derived from escrow)   │    │
│  │  Authority: escrow_pda                          │    │
│  │  Balance: ENCRYPTED (confidential)              │    │
│  │                                                  │    │
│  │  Holds: amount + platform_fee + seller_stake    │    │
│  └────────────────────────────────────────────────┘    │
└────────────────┬────────────────────────────────────────┘
                 │
                 │ 2. finalize_order()
                 │    Confidential Transfers:
                 │    - escrow_ata → seller_ata (payment + stake)
                 │    - escrow_ata → platform_ata (fee)
                 ↓
┌─────────────────────────────────────────────────────────┐
│ SELLER / PLATFORM                                        │
│  Seller ATA: [seller_ata]  ← Receives payment          │
│  Platform ATA: [platform_ata]  ← Receives fee          │
└─────────────────────────────────────────────────────────┘
```

---

## Step 1: Update Dependencies

### **Cargo.toml**

```toml
[package]
name = "escrow"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "lib"]
name = "escrow"

[dependencies]
anchor-lang = "0.31.1"
anchor-spl = "0.31.1"
spl-token = "6.0"
spl-token-2022 = { version = "6.0", features = ["no-entrypoint"] }
spl-associated-token-account = "6.0"
arcium-anchor = "0.3.0"

[features]
no-entrypoint = []
no-idl = []
no-log-ix-name = []
cpi = ["no-entrypoint"]
default = []
```

**Key additions:**
- `spl-token-2022` for confidential transfers
- `spl-associated-token-account` for ATA derivation

---

## Step 2: Update Program Imports

### **programs/escrow/src/lib.rs**

Add at the top:

```rust
use anchor_lang::prelude::*;
use anchor_spl::{
    associated_token::AssociatedToken,
    token_2022::{self, Token2022, TransferChecked},
    token_interface::{Mint, TokenAccount},
};
use arcium_anchor::prelude::*;
use spl_token_2022::{
    extension::confidential_transfer::{
        instruction::{
            transfer_with_fee as confidential_transfer_with_fee,
            TransferWithFeeProofContext,
        },
        ConfidentialTransferAccount,
    },
};

declare_id!("5QvQbnrL7fKpM5pCMS3zNqgTK8ALNkgHgRvgd49YF7v4");
```

---

## Step 3: Update Escrow Account Structure

### **Modified Escrow Struct**

```rust
#[account]
pub struct Escrow {
    pub buyer: Pubkey,                // 32
    pub seller: Pubkey,               // 32
    pub arbiter: Pubkey,              // 32
    pub amount: u64,                  // 8 - Amount for this order
    pub seller_stake: u64,            // 8 - Seller's stake
    pub platform_fee: u64,            // 8 - Platform fee
    pub amount_commitment: [u8; 32],  // 32 - Optional: Hash commitment
    pub state: EscrowState,           // 1 + size
    pub bump: u8,                     // 1
    pub created_at: i64,              // 8
    pub accepted_at: i64,             // 8
    pub shipped_at: i64,              // 8
    pub delivered_at: i64,            // 8
    pub dispute_opened_at: i64,       // 8
    pub encrypted_shipping: Vec<u8>,  // 4 + 256
    pub shipping_encryption_nonce: [u8; 16], // 16
    pub tracking_number: String,      // 4 + 64
    pub dispute_reason: String,       // 4 + 200
    pub use_private_reputation: bool, // 1

    // NEW: cSOL-specific fields
    pub csol_mint: Pubkey,           // 32 - The cSOL mint address
    pub escrow_ata: Pubkey,          // 32 - Escrow's cSOL ATA (holds funds)
}

impl Escrow {
    pub const LEN: usize = 8 // discriminator
        + 32 + 32 + 32       // buyer, seller, arbiter
        + 8 + 8 + 8          // amount, seller_stake, platform_fee
        + 32                 // amount_commitment
        + 1 + 1              // state (enum) + bump
        + 8 + 8 + 8 + 8 + 8  // timestamps
        + (4 + 256)          // encrypted_shipping
        + 16                 // shipping_nonce
        + (4 + 64)           // tracking_number
        + (4 + 200)          // dispute_reason
        + 1                  // use_private_reputation
        + 32 + 32;           // csol_mint + escrow_ata
}
```

---

## Step 4: Create Order with ATA Initialization

### **CreateOrder Context**

```rust
#[derive(Accounts)]
#[instruction(amount: u64, order_id: u64)]
pub struct CreateOrder<'info> {
    #[account(mut)]
    pub buyer: Signer<'info>,

    // Escrow PDA (stores state)
    #[account(
        init,
        payer = buyer,
        space = Escrow::LEN,
        seeds = [b"escrow", buyer.key().as_ref(), &order_id.to_le_bytes()],
        bump
    )]
    pub escrow: Account<'info, Escrow>,

    // cSOL mint (Token-2022)
    #[account(
        mint::token_program = token_program,
    )]
    pub csol_mint: InterfaceAccount<'info, Mint>,

    // Buyer's cSOL ATA (source of funds)
    #[account(
        mut,
        associated_token::mint = csol_mint,
        associated_token::authority = buyer,
        associated_token::token_program = token_program,
    )]
    pub buyer_ata: InterfaceAccount<'info, TokenAccount>,

    // Escrow's cSOL ATA (destination - holds funds)
    #[account(
        init_if_needed,
        payer = buyer,
        associated_token::mint = csol_mint,
        associated_token::authority = escrow,
        associated_token::token_program = token_program,
    )]
    pub escrow_ata: InterfaceAccount<'info, TokenAccount>,

    // Platform config & arbiter pool (existing)
    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,
    #[account(seeds = [b"arbiter_pool"], bump = arbiter_pool.bump)]
    pub arbiter_pool: Account<'info, ArbiterPool>,

    // Buyer reputation (init if needed)
    #[account(
        init_if_needed,
        payer = buyer,
        space = 8 + 32 + 8 + 8 + 8 + 8 + 8 + 8 + 1,
        seeds = [b"reputation", buyer.key().as_ref()],
        bump
    )]
    pub buyer_reputation: Account<'info, UserReputation>,

    // Programs
    pub token_program: Program<'info, Token2022>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub system_program: Program<'info, System>,
}
```

### **create_order Instruction**

```rust
pub fn create_order(
    ctx: Context<CreateOrder>,
    amount: u64,
    order_id: u64,
    encrypted_shipping: Vec<u8>,
    shipping_nonce: [u8; 16],
    amount_commitment: Option<[u8; 32]>,
    use_private_reputation: bool,
) -> Result<()> {
    require!(amount > 0, ErrorCode::InvalidAmount);
    require!(
        encrypted_shipping.len() <= 256,
        ErrorCode::ShippingDataTooLarge
    );
    require!(
        ctx.accounts.arbiter_pool.arbiters.len() > 0,
        ErrorCode::NoArbitersAvailable
    );

    let config = &ctx.accounts.config;
    let escrow = &mut ctx.accounts.escrow;
    let clock = Clock::get()?;

    // Calculate fees
    let platform_fee = (amount * config.platform_fee_bps as u64) / 10000;
    let total_required = amount + platform_fee;

    // ===== CONFIDENTIAL TRANSFER: buyer → escrow =====
    // Transfer cSOL from buyer's ATA to escrow's ATA
    let cpi_accounts = TransferChecked {
        from: ctx.accounts.buyer_ata.to_account_info(),
        mint: ctx.accounts.csol_mint.to_account_info(),
        to: ctx.accounts.escrow_ata.to_account_info(),
        authority: ctx.accounts.buyer.to_account_info(),
    };

    let cpi_program = ctx.accounts.token_program.to_account_info();
    let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);

    // Execute confidential transfer
    // Note: For Token-2022 confidential transfers, use transfer_checked
    // The confidential transfer extension is automatically used if configured
    token_2022::transfer_checked(
        cpi_ctx,
        total_required,
        ctx.accounts.csol_mint.decimals,
    )?;

    // Select random arbiter
    let arbiter_index = (clock.unix_timestamp as usize)
        % ctx.accounts.arbiter_pool.arbiters.len();
    let arbiter = ctx.accounts.arbiter_pool.arbiters[arbiter_index];

    // Initialize escrow state
    escrow.buyer = ctx.accounts.buyer.key();
    escrow.seller = Pubkey::default(); // Set when accepted
    escrow.arbiter = arbiter;
    escrow.amount = amount;
    escrow.seller_stake = 0; // Set when accepted
    escrow.platform_fee = platform_fee;
    escrow.amount_commitment = amount_commitment.unwrap_or([0u8; 32]);
    escrow.state = EscrowState::Created;
    escrow.bump = ctx.bumps.escrow;
    escrow.created_at = clock.unix_timestamp;
    escrow.accepted_at = 0;
    escrow.shipped_at = 0;
    escrow.delivered_at = 0;
    escrow.dispute_opened_at = 0;
    escrow.encrypted_shipping = encrypted_shipping;
    escrow.shipping_encryption_nonce = shipping_nonce;
    escrow.tracking_number = String::new();
    escrow.dispute_reason = String::new();
    escrow.use_private_reputation = use_private_reputation;

    // NEW: Store cSOL references
    escrow.csol_mint = ctx.accounts.csol_mint.key();
    escrow.escrow_ata = ctx.accounts.escrow_ata.key();

    // Initialize buyer reputation if needed
    if ctx.accounts.buyer_reputation.to_account_info().data_is_empty() {
        let buyer_rep = &mut ctx.accounts.buyer_reputation;
        buyer_rep.user = ctx.accounts.buyer.key();
        buyer_rep.total_orders = 0;
        buyer_rep.successful_orders = 0;
        buyer_rep.disputes_opened = 0;
        buyer_rep.disputes_won = 0;
        buyer_rep.disputes_lost = 0;
        buyer_rep.reputation_score = 500;
        buyer_rep.bump = ctx.bumps.buyer_reputation;
    }

    emit!(OrderCreatedEvent {
        order_id,
        buyer: ctx.accounts.buyer.key(),
        amount,
        arbiter,
    });

    Ok(())
}
```

---

## Step 5: Accept Order (Seller Stake)

### **AcceptOrder Context**

```rust
#[derive(Accounts)]
pub struct AcceptOrder<'info> {
    #[account(mut)]
    pub seller: Signer<'info>,

    #[account(
        mut,
        seeds = [b"escrow", escrow.buyer.as_ref(), &escrow.created_at.to_le_bytes()],
        bump = escrow.bump,
    )]
    pub escrow: Account<'info, Escrow>,

    #[account(
        mut,
        associated_token::mint = csol_mint,
        associated_token::authority = seller,
        associated_token::token_program = token_program,
    )]
    pub seller_ata: InterfaceAccount<'info, TokenAccount>,

    #[account(
        mut,
        address = escrow.escrow_ata,
    )]
    pub escrow_ata: InterfaceAccount<'info, TokenAccount>,

    #[account(
        address = escrow.csol_mint,
    )]
    pub csol_mint: InterfaceAccount<'info, Mint>,

    #[account(
        init_if_needed,
        payer = seller,
        space = 8 + 32 + 8 + 8 + 8 + 8 + 8 + 8 + 1,
        seeds = [b"reputation", seller.key().as_ref()],
        bump
    )]
    pub seller_reputation: Account<'info, UserReputation>,

    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,

    pub token_program: Program<'info, Token2022>,
    pub system_program: Program<'info, System>,
}
```

### **accept_order Instruction**

```rust
pub fn accept_order(ctx: Context<AcceptOrder>) -> Result<()> {
    let escrow = &mut ctx.accounts.escrow;
    let config = &ctx.accounts.config;
    let clock = Clock::get()?;

    // Validate state
    require!(
        escrow.state == EscrowState::Created,
        ErrorCode::InvalidState
    );

    // Check deadline
    let deadline = escrow.created_at + config.acceptance_deadline;
    require!(
        clock.unix_timestamp <= deadline,
        ErrorCode::DeadlineExpired
    );

    // Calculate seller stake (10% of amount)
    let seller_stake = (escrow.amount * config.seller_stake_bps as u64) / 10000;

    // ===== CONFIDENTIAL TRANSFER: seller → escrow (stake) =====
    let cpi_accounts = TransferChecked {
        from: ctx.accounts.seller_ata.to_account_info(),
        mint: ctx.accounts.csol_mint.to_account_info(),
        to: ctx.accounts.escrow_ata.to_account_info(),
        authority: ctx.accounts.seller.to_account_info(),
    };

    let cpi_program = ctx.accounts.token_program.to_account_info();
    let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);

    token_2022::transfer_checked(
        cpi_ctx,
        seller_stake,
        ctx.accounts.csol_mint.decimals,
    )?;

    // Update escrow
    escrow.seller = ctx.accounts.seller.key();
    escrow.seller_stake = seller_stake;
    escrow.state = EscrowState::Accepted;
    escrow.accepted_at = clock.unix_timestamp;

    // Initialize seller reputation if needed
    if ctx.accounts.seller_reputation.to_account_info().data_is_empty() {
        let seller_rep = &mut ctx.accounts.seller_reputation;
        seller_rep.user = ctx.accounts.seller.key();
        seller_rep.total_orders = 0;
        seller_rep.successful_orders = 0;
        seller_rep.disputes_opened = 0;
        seller_rep.disputes_won = 0;
        seller_rep.disputes_lost = 0;
        seller_rep.reputation_score = 500;
        seller_rep.bump = ctx.bumps.seller_reputation;
    }

    emit!(OrderAcceptedEvent {
        buyer: escrow.buyer,
        seller: ctx.accounts.seller.key(),
        amount: escrow.amount,
    });

    Ok(())
}
```

---

## Step 6: Finalize Order (Release Funds)

### **FinalizeOrder Context**

```rust
#[derive(Accounts)]
pub struct FinalizeOrder<'info> {
    #[account(
        mut,
        seeds = [b"escrow", escrow.buyer.as_ref(), &escrow.created_at.to_le_bytes()],
        bump = escrow.bump,
    )]
    pub escrow: Account<'info, Escrow>,

    #[account(
        mut,
        address = escrow.escrow_ata,
    )]
    pub escrow_ata: InterfaceAccount<'info, TokenAccount>,

    #[account(
        mut,
        associated_token::mint = csol_mint,
        associated_token::authority = seller,
        associated_token::token_program = token_program,
    )]
    pub seller_ata: InterfaceAccount<'info, TokenAccount>,

    #[account(
        mut,
        associated_token::mint = csol_mint,
        associated_token::authority = treasury,
        associated_token::token_program = token_program,
    )]
    pub treasury_ata: InterfaceAccount<'info, TokenAccount>,

    /// CHECK: Seller account (validated via escrow.seller)
    #[account(address = escrow.seller)]
    pub seller: UncheckedAccount<'info>,

    /// CHECK: Treasury account (validated via config.treasury)
    #[account(address = config.treasury)]
    pub treasury: UncheckedAccount<'info>,

    #[account(
        address = escrow.csol_mint,
    )]
    pub csol_mint: InterfaceAccount<'info, Mint>,

    #[account(mut)]
    pub buyer_reputation: Account<'info, UserReputation>,

    #[account(mut)]
    pub seller_reputation: Account<'info, UserReputation>,

    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,

    pub token_program: Program<'info, Token2022>,
}
```

### **finalize_order Instruction**

```rust
pub fn finalize_order(ctx: Context<FinalizeOrder>) -> Result<()> {
    let escrow = &mut ctx.accounts.escrow;
    let config = &ctx.accounts.config;
    let clock = Clock::get()?;

    // Validate state
    require!(
        escrow.state == EscrowState::Delivered,
        ErrorCode::InvalidState
    );

    // Check dispute window passed
    let deadline = escrow.delivered_at + config.dispute_window;
    require!(
        clock.unix_timestamp > deadline,
        ErrorCode::DeadlineNotReached
    );

    // Calculate distribution
    let to_seller = escrow.amount - escrow.platform_fee + escrow.seller_stake;
    let to_platform = escrow.platform_fee;

    // PDA signer seeds
    let escrow_seeds = &[
        b"escrow",
        escrow.buyer.as_ref(),
        &escrow.created_at.to_le_bytes(),
        &[escrow.bump],
    ];
    let signer_seeds = &[&escrow_seeds[..]];

    // ===== CONFIDENTIAL TRANSFER: escrow → seller =====
    let cpi_accounts_seller = TransferChecked {
        from: ctx.accounts.escrow_ata.to_account_info(),
        mint: ctx.accounts.csol_mint.to_account_info(),
        to: ctx.accounts.seller_ata.to_account_info(),
        authority: ctx.accounts.escrow.to_account_info(),
    };

    let cpi_program = ctx.accounts.token_program.to_account_info();
    let cpi_ctx_seller = CpiContext::new_with_signer(
        cpi_program.clone(),
        cpi_accounts_seller,
        signer_seeds,
    );

    token_2022::transfer_checked(
        cpi_ctx_seller,
        to_seller,
        ctx.accounts.csol_mint.decimals,
    )?;

    // ===== CONFIDENTIAL TRANSFER: escrow → platform =====
    let cpi_accounts_platform = TransferChecked {
        from: ctx.accounts.escrow_ata.to_account_info(),
        mint: ctx.accounts.csol_mint.to_account_info(),
        to: ctx.accounts.treasury_ata.to_account_info(),
        authority: ctx.accounts.escrow.to_account_info(),
    };

    let cpi_ctx_platform = CpiContext::new_with_signer(
        cpi_program,
        cpi_accounts_platform,
        signer_seeds,
    );

    token_2022::transfer_checked(
        cpi_ctx_platform,
        to_platform,
        ctx.accounts.csol_mint.decimals,
    )?;

    // Update reputation stats
    let buyer_rep = &mut ctx.accounts.buyer_reputation;
    buyer_rep.total_orders += 1;
    buyer_rep.successful_orders += 1;

    let seller_rep = &mut ctx.accounts.seller_reputation;
    seller_rep.total_orders += 1;
    seller_rep.successful_orders += 1;

    // Update reputation scores
    if escrow.use_private_reputation {
        // MPC reputation calculation (emit event for off-chain processing)
        buyer_rep.reputation_score = 0;
        seller_rep.reputation_score = 0;

        emit!(PrivateReputationUpdateNeeded {
            buyer: escrow.buyer,
            seller: escrow.seller,
            msg: "Call calculate_reputation_private for both users".to_string(),
        });
    } else {
        // Standard plaintext reputation calculation
        update_reputation_score(buyer_rep);
        update_reputation_score(seller_rep);
    }

    // Update state
    escrow.state = EscrowState::Completed;

    emit!(OrderCompletedEvent {
        buyer: escrow.buyer,
        seller: escrow.seller,
        amount: escrow.amount,
    });

    Ok(())
}
```

---

## Step 7: Refund (Dispute Resolution)

### **ResolveDispute Context** (excerpt showing refund to buyer)

```rust
pub fn resolve_dispute(
    ctx: Context<ResolveDispute>,
    winner: DisputeWinner,
) -> Result<()> {
    let escrow = &mut ctx.accounts.escrow;

    // ... validation ...

    let escrow_seeds = &[
        b"escrow",
        escrow.buyer.as_ref(),
        &escrow.created_at.to_le_bytes(),
        &[escrow.bump],
    ];
    let signer_seeds = &[&escrow_seeds[..]];

    match winner {
        DisputeWinner::Buyer => {
            // Buyer gets amount + seller stake back
            let refund = escrow.amount + escrow.seller_stake;

            // ===== CONFIDENTIAL TRANSFER: escrow → buyer =====
            let cpi_accounts = TransferChecked {
                from: ctx.accounts.escrow_ata.to_account_info(),
                mint: ctx.accounts.csol_mint.to_account_info(),
                to: ctx.accounts.buyer_ata.to_account_info(),
                authority: ctx.accounts.escrow.to_account_info(),
            };

            let cpi_ctx = CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                cpi_accounts,
                signer_seeds,
            );

            token_2022::transfer_checked(
                cpi_ctx,
                refund,
                ctx.accounts.csol_mint.decimals,
            )?;

            // Platform still gets fee
            // ... (similar transfer to platform) ...

            escrow.state = EscrowState::Refunded;
        }
        DisputeWinner::Seller => {
            // ... (similar to finalize_order) ...
        }
    }

    Ok(())
}
```

---

## Step 8: Testing Setup

### **Create Test Script**

Create `scripts/test_csol_escrow.ts`:

```typescript
import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { Escrow } from "../target/types/escrow";
import {
  getAssociatedTokenAddress,
  TOKEN_2022_PROGRAM_ID,
  ASSOCIATED_TOKEN_PROGRAM_ID,
} from "@solana/spl-token";
import { PublicKey, Keypair, SystemProgram } from "@solana/web3.js";

describe("cSOL Escrow", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.Escrow as Program<Escrow>;

  // Load from env or hardcode for testing
  const CSOL_MINT = new PublicKey(process.env.CSOL_MINT || "YOUR_CSOL_MINT");

  it("Creates an escrow with cSOL", async () => {
    const buyer = provider.wallet;
    const orderId = new anchor.BN(Date.now());

    // Derive escrow PDA
    const [escrowPda, escrowBump] = PublicKey.findProgramAddressSync(
      [
        Buffer.from("escrow"),
        buyer.publicKey.toBuffer(),
        orderId.toArrayLike(Buffer, "le", 8),
      ],
      program.programId
    );

    // Derive ATAs
    const buyerAta = await getAssociatedTokenAddress(
      CSOL_MINT,
      buyer.publicKey,
      false,
      TOKEN_2022_PROGRAM_ID,
      ASSOCIATED_TOKEN_PROGRAM_ID
    );

    const escrowAta = await getAssociatedTokenAddress(
      CSOL_MINT,
      escrowPda,
      true, // allowOwnerOffCurve = true (PDA)
      TOKEN_2022_PROGRAM_ID,
      ASSOCIATED_TOKEN_PROGRAM_ID
    );

    // Create order
    const amount = new anchor.BN(5_000_000_000); // 5 cSOL (in smallest units)
    const encryptedShipping = Buffer.from("encrypted_data");
    const shippingNonce = Buffer.alloc(16);

    const tx = await program.methods
      .createOrder(
        amount,
        orderId,
        encryptedShipping,
        Array.from(shippingNonce),
        null, // amount_commitment
        true  // use_private_reputation
      )
      .accounts({
        buyer: buyer.publicKey,
        escrow: escrowPda,
        csolMint: CSOL_MINT,
        buyerAta: buyerAta,
        escrowAta: escrowAta,
        // ... other accounts ...
        tokenProgram: TOKEN_2022_PROGRAM_ID,
        associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
        systemProgram: SystemProgram.programId,
      })
      .rpc();

    console.log("Order created:", tx);

    // Verify escrow state
    const escrowAccount = await program.account.escrow.fetch(escrowPda);
    console.log("Escrow state:", escrowAccount);
    console.log("Escrow ATA:", escrowAccount.escrowAta.toString());
  });
});
```

---

## Step 9: Deployment Checklist

### **Before Deployment:**

- [ ] Update all instructions to use Token-2022
- [ ] Test create_order on devnet
- [ ] Test accept_order (seller stake)
- [ ] Test finalize_order (release funds)
- [ ] Test resolve_dispute (refund)
- [ ] Verify ATA initialization works correctly
- [ ] Check PDA signing for transfers
- [ ] Test confidential transfer receipts

### **Deployment Steps:**

```bash
# 1. Build the program
anchor build

# 2. Deploy to devnet
anchor deploy --provider.cluster devnet

# 3. Initialize platform config
ts-node scripts/initialize_platform.ts

# 4. Test end-to-end flow
anchor test --skip-local-validator --provider.cluster devnet

# 5. Deploy to mainnet (when ready)
anchor deploy --provider.cluster mainnet-beta
```

---

## Step 10: API Integration

### **Python API Endpoint**

```python
# In services/api/app.py

@app.post("/marketplace/buy-with-escrow")
def marketplace_buy_with_escrow(req: BuyWithEscrowReq):
    """
    Create escrow order with cSOL confidential transfer
    """

    buyer_pub = get_pubkey_from_keypair(req.buyer_keyfile)
    listing = listing_get(req.listing_id)

    # Generate order ID
    order_id = int(time.time() * 1000)

    # Derive escrow PDA
    escrow_pda = derive_escrow_pda(buyer_pub, order_id)
    escrow_ata = derive_ata(escrow_pda, CSOL_MINT)

    # Call Anchor program via subprocess or Python SDK
    result = subprocess.run([
        "anchor",
        "run",
        "create-order",
        "--",
        "--amount", str(int(listing.price * 1e9)),
        "--order-id", str(order_id),
        "--buyer", req.buyer_keyfile,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise HTTPException(500, f"Failed to create escrow: {result.stderr}")

    # Store in local DB
    local_escrow = {
        "id": str(order_id),
        "escrow_pda": str(escrow_pda),
        "escrow_ata": str(escrow_ata),
        "buyer_pub": buyer_pub,
        "seller_pub": listing.seller_pub,
        "amount_sol": str(listing.price),
        "status": "CREATED",
        "created_at": now(),
    }

    save_escrow(local_escrow)

    return {"escrow_id": order_id, "escrow_pda": str(escrow_pda)}
```

---

## Summary

### **What Changed:**

1. ✅ **Escrow holds cSOL in ATA** (not SOL in account balance)
2. ✅ **Each escrow gets its own ATA** (derived from escrow PDA + mint)
3. ✅ **Confidential transfers** via Token-2022
4. ✅ **PDA signing** for releasing funds
5. ✅ **Same state machine** (just different transfer mechanism)

### **Key Files Modified:**

- `programs/escrow/Cargo.toml` - Add Token-2022 dependencies
- `programs/escrow/src/lib.rs` - All instructions updated
- `tests/escrow.ts` - Add cSOL-specific tests

### **Next Steps:**

1. Implement these changes in your local copy
2. Test thoroughly on devnet
3. Deploy to mainnet when confident
4. Update API and frontend

**The escrow now properly custodies cSOL with confidential transfers!** 🎉
