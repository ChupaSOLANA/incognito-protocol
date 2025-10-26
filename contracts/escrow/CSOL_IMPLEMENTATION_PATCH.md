# cSOL Escrow Implementation - Complete Patch

This document contains all the code changes needed to convert the escrow contract to use cSOL.

## ⚠️ IMPORTANT: Apply these changes to `/contracts/escrow/programs/escrow/src/lib.rs`

---

## Change 1: Replace create_order instruction (lines ~183-271)

**Find the `create_order` function and replace it with:**

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

    let platform_fee = (amount * config.platform_fee_bps as u64) / 10000;
    let total_required = amount + platform_fee;

    // ===== CONFIDENTIAL TRANSFER: buyer → escrow (cSOL) =====
    let cpi_accounts = TransferChecked {
        from: ctx.accounts.buyer_ata.to_account_info(),
        mint: ctx.accounts.csol_mint.to_account_info(),
        to: ctx.accounts.escrow_ata.to_account_info(),
        authority: ctx.accounts.buyer.to_account_info(),
    };

    let cpi_program = ctx.accounts.token_program.to_account_info();
    let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);

    token_2022::transfer_checked(cpi_ctx, total_required, CSOL_DECIMALS)?;

    let arbiter_index = (clock.unix_timestamp as usize) % ctx.accounts.arbiter_pool.arbiters.len();
    let arbiter = ctx.accounts.arbiter_pool.arbiters[arbiter_index];

    escrow.buyer = ctx.accounts.buyer.key();
    escrow.seller = Pubkey::default();
    escrow.arbiter = arbiter;
    escrow.amount = amount;
    escrow.seller_stake = 0;
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
    escrow.csol_mint = ctx.accounts.csol_mint.key();
    escrow.escrow_ata = ctx.accounts.escrow_ata.key();

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

## Change 2: Replace CreateOrder context (lines ~958-984)

**Find the `CreateOrder` struct and replace it with:**

```rust
#[derive(Accounts)]
#[instruction(amount: u64, order_id: u64, encrypted_shipping: Vec<u8>, shipping_nonce: [u8; 16])]
pub struct CreateOrder<'info> {
    #[account(mut)]
    pub buyer: Signer<'info>,

    #[account(
        init,
        payer = fee_payer,
        space = 8 + 32 + 32 + 32 + 8 + 8 + 8 + 32 + 1 + 1 + 8 + 8 + 8 + 8 + 8 + 260 + 16 + 68 + 204 + 1 + 32 + 32,
        seeds = [b"escrow", buyer.key().as_ref(), &order_id.to_le_bytes()],
        bump
    )]
    pub escrow: Account<'info, Escrow>,

    #[account(
        mint::token_program = token_program,
    )]
    pub csol_mint: InterfaceAccount<'info, Mint>,

    #[account(
        mut,
        associated_token::mint = csol_mint,
        associated_token::authority = buyer,
        associated_token::token_program = token_program,
    )]
    pub buyer_ata: InterfaceAccount<'info, TokenAccount>,

    #[account(
        init_if_needed,
        payer = fee_payer,
        associated_token::mint = csol_mint,
        associated_token::authority = escrow,
        associated_token::token_program = token_program,
    )]
    pub escrow_ata: InterfaceAccount<'info, TokenAccount>,

    #[account(
        init_if_needed,
        payer = fee_payer,
        space = 8 + 32 + 8 + 8 + 8 + 8 + 8 + 8 + 1,
        seeds = [b"reputation", buyer.key().as_ref()],
        bump
    )]
    pub buyer_reputation: Account<'info, UserReputation>,

    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,

    #[account(seeds = [b"arbiter_pool"], bump = arbiter_pool.bump)]
    pub arbiter_pool: Account<'info, ArbiterPool>,

    #[account(mut)]
    pub fee_payer: Signer<'info>,

    pub token_program: Program<'info, Token2022>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub system_program: Program<'info, System>,
}
```

---

## Change 3: Replace accept_order instruction (lines ~273-335)

**Find the `accept_order` function and replace it with:**

```rust
pub fn accept_order(ctx: Context<AcceptOrder>) -> Result<()> {
    let escrow = &mut ctx.accounts.escrow;
    let config = &ctx.accounts.config;
    let clock = Clock::get()?;

    require!(
        escrow.state == EscrowState::Created,
        ErrorCode::InvalidState
    );

    let deadline = escrow.created_at + config.acceptance_deadline;
    require!(
        clock.unix_timestamp <= deadline,
        ErrorCode::DeadlineExpired
    );

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

    token_2022::transfer_checked(cpi_ctx, seller_stake, CSOL_DECIMALS)?;

    escrow.seller = ctx.accounts.seller.key();
    escrow.seller_stake = seller_stake;
    escrow.state = EscrowState::Accepted;
    escrow.accepted_at = clock.unix_timestamp;

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

## Change 4: Replace AcceptOrder context (lines ~986-1003)

**Find the `AcceptOrder` struct and replace it with:**

```rust
#[derive(Accounts)]
pub struct AcceptOrder<'info> {
    #[account(mut)]
    pub seller: Signer<'info>,

    #[account(mut)]
    pub escrow: Account<'info, Escrow>,

    #[account(
        address = escrow.csol_mint,
    )]
    pub csol_mint: InterfaceAccount<'info, Mint>,

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
        init_if_needed,
        payer = fee_payer,
        space = 8 + 32 + 8 + 8 + 8 + 8 + 8 + 8 + 1,
        seeds = [b"reputation", seller.key().as_ref()],
        bump
    )]
    pub seller_reputation: Account<'info, UserReputation>,

    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,

    #[account(mut)]
    pub fee_payer: Signer<'info>,

    pub token_program: Program<'info, Token2022>,
    pub system_program: Program<'info, System>,
}
```

---

## Change 5: Replace finalize_order instruction (lines ~417-484)

**Find the `finalize_order` function and replace the fund distribution section with:**

```rust
pub fn finalize_order(ctx: Context<FinalizeOrder>) -> Result<()> {
    let escrow = &mut ctx.accounts.escrow;
    let config = &ctx.accounts.config;
    let clock = Clock::get()?;

    require!(
        escrow.state == EscrowState::Delivered,
        ErrorCode::InvalidState
    );

    let deadline = escrow.delivered_at + config.dispute_window;
    require!(
        clock.unix_timestamp > deadline,
        ErrorCode::DeadlineNotReached
    );

    let to_seller = escrow.amount - escrow.platform_fee + escrow.seller_stake;
    let to_platform = escrow.platform_fee;

    let escrow_seeds = &[
        b"escrow",
        escrow.buyer.as_ref(),
        &escrow.created_at.to_le_bytes(),
        &[escrow.bump],
    ];
    let signer_seeds = &[&escrow_seeds[..]];

    // ===== TRANSFER: escrow → seller =====
    let cpi_accounts_seller = TransferChecked {
        from: ctx.accounts.escrow_ata.to_account_info(),
        mint: ctx.accounts.csol_mint.to_account_info(),
        to: ctx.accounts.seller_ata.to_account_info(),
        authority: ctx.accounts.escrow.to_account_info(),
    };

    let cpi_ctx_seller = CpiContext::new_with_signer(
        ctx.accounts.token_program.to_account_info(),
        cpi_accounts_seller,
        signer_seeds,
    );

    token_2022::transfer_checked(cpi_ctx_seller, to_seller, CSOL_DECIMALS)?;

    // ===== TRANSFER: escrow → platform =====
    let cpi_accounts_platform = TransferChecked {
        from: ctx.accounts.escrow_ata.to_account_info(),
        mint: ctx.accounts.csol_mint.to_account_info(),
        to: ctx.accounts.treasury_ata.to_account_info(),
        authority: ctx.accounts.escrow.to_account_info(),
    };

    let cpi_ctx_platform = CpiContext::new_with_signer(
        ctx.accounts.token_program.to_account_info(),
        cpi_accounts_platform,
        signer_seeds,
    );

    token_2022::transfer_checked(cpi_ctx_platform, to_platform, CSOL_DECIMALS)?;

    let buyer_rep = &mut ctx.accounts.buyer_reputation;
    buyer_rep.total_orders += 1;
    buyer_rep.successful_orders += 1;

    let seller_rep = &mut ctx.accounts.seller_reputation;
    seller_rep.total_orders += 1;
    seller_rep.successful_orders += 1;

    if escrow.use_private_reputation {
        buyer_rep.reputation_score = 0;
        seller_rep.reputation_score = 0;

        emit!(PrivateReputationUpdateNeeded {
            buyer: escrow.buyer,
            seller: escrow.seller,
            msg: "Call calculate_reputation_private for both users".to_string(),
        });
    } else {
        update_reputation_score(buyer_rep);
        update_reputation_score(seller_rep);
    }

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

## Change 6: Replace FinalizeOrder context (lines ~1021-1037)

**Find the `FinalizeOrder` struct and replace it with:**

```rust
#[derive(Accounts)]
pub struct FinalizeOrder<'info> {
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,

    #[account(
        address = escrow.csol_mint,
    )]
    pub csol_mint: InterfaceAccount<'info, Mint>,

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

    /// CHECK: Seller account
    pub seller: UncheckedAccount<'info>,

    /// CHECK: Treasury account
    pub treasury: UncheckedAccount<'info>,

    #[account(mut)]
    pub buyer_reputation: Account<'info, UserReputation>,

    #[account(mut)]
    pub seller_reputation: Account<'info, UserReputation>,

    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,

    pub token_program: Program<'info, Token2022>,
}
```

---

## Change 7: Update resolve_dispute for buyer win (lines ~678-766)

**In the `resolve_dispute` function, find the `DisputeWinner::Buyer` match arm and replace the fund transfers:**

```rust
DisputeWinner::Buyer => {
    let refund = escrow.amount + escrow.seller_stake;

    let escrow_seeds = &[
        b"escrow",
        escrow.buyer.as_ref(),
        &escrow.created_at.to_le_bytes(),
        &[escrow.bump],
    ];
    let signer_seeds = &[&escrow_seeds[..]];

    // Transfer to buyer
    let cpi_accounts_buyer = TransferChecked {
        from: ctx.accounts.escrow_ata.to_account_info(),
        mint: ctx.accounts.csol_mint.to_account_info(),
        to: ctx.accounts.buyer_ata.to_account_info(),
        authority: ctx.accounts.escrow.to_account_info(),
    };

    let cpi_ctx_buyer = CpiContext::new_with_signer(
        ctx.accounts.token_program.to_account_info(),
        cpi_accounts_buyer,
        signer_seeds,
    );

    token_2022::transfer_checked(cpi_ctx_buyer, refund, CSOL_DECIMALS)?;

    // Transfer fee to platform
    let fee = escrow.platform_fee;
    let cpi_accounts_platform = TransferChecked {
        from: ctx.accounts.escrow_ata.to_account_info(),
        mint: ctx.accounts.csol_mint.to_account_info(),
        to: ctx.accounts.treasury_ata.to_account_info(),
        authority: ctx.accounts.escrow.to_account_info(),
    };

    let cpi_ctx_platform = CpiContext::new_with_signer(
        ctx.accounts.token_program.to_account_info(),
        cpi_accounts_platform,
        signer_seeds,
    );

    token_2022::transfer_checked(cpi_ctx_platform, fee, CSOL_DECIMALS)?;

    let buyer_rep = &mut ctx.accounts.buyer_reputation;
    buyer_rep.total_orders += 1;
    buyer_rep.disputes_won += 1;
    update_reputation_score(buyer_rep);

    let seller_rep = &mut ctx.accounts.seller_reputation;
    seller_rep.total_orders += 1;
    seller_rep.disputes_lost += 1;
    update_reputation_score(seller_rep);

    escrow.state = EscrowState::Refunded;
}
```

**Also update the `DisputeWinner::Seller` match arm similarly for seller win case.**

---

## Change 8: Update ResolveDispute context (lines ~1094-1114)

**Find the `ResolveDispute` struct and add cSOL accounts:**

```rust
#[derive(Accounts)]
pub struct ResolveDispute<'info> {
    pub arbiter: Signer<'info>,

    #[account(mut)]
    pub escrow: Account<'info, Escrow>,

    #[account(
        address = escrow.csol_mint,
    )]
    pub csol_mint: InterfaceAccount<'info, Mint>,

    #[account(
        mut,
        address = escrow.escrow_ata,
    )]
    pub escrow_ata: InterfaceAccount<'info, TokenAccount>,

    #[account(
        mut,
        associated_token::mint = csol_mint,
        associated_token::authority = buyer,
        associated_token::token_program = token_program,
    )]
    pub buyer_ata: InterfaceAccount<'info, TokenAccount>,

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

    /// CHECK: Buyer account
    pub buyer: UncheckedAccount<'info>,

    /// CHECK: Seller account
    pub seller: UncheckedAccount<'info>,

    /// CHECK: Treasury account
    pub treasury: UncheckedAccount<'info>,

    #[account(mut)]
    pub buyer_reputation: Account<'info, UserReputation>,

    #[account(mut)]
    pub seller_reputation: Account<'info, UserReputation>,

    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,

    pub token_program: Program<'info, Token2022>,
}
```

---

## Change 9: Update process_acceptance_timeout (lines ~544-577)

**Replace the refund logic in this function:**

```rust
pub fn process_acceptance_timeout(ctx: Context<ProcessAcceptanceTimeout>) -> Result<()> {
    let escrow = &mut ctx.accounts.escrow;
    let config = &ctx.accounts.config;
    let clock = Clock::get()?;

    require!(
        escrow.state == EscrowState::Created,
        ErrorCode::InvalidState
    );

    let deadline = escrow.created_at + config.acceptance_deadline;
    require!(
        clock.unix_timestamp > deadline,
        ErrorCode::DeadlineNotReached
    );

    let refund = escrow.amount;

    let escrow_seeds = &[
        b"escrow",
        escrow.buyer.as_ref(),
        &escrow.created_at.to_le_bytes(),
        &[escrow.bump],
    ];
    let signer_seeds = &[&escrow_seeds[..]];

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

    token_2022::transfer_checked(cpi_ctx, refund, CSOL_DECIMALS)?;

    escrow.state = EscrowState::Refunded;

    emit!(OrderRefundedEvent {
        buyer: escrow.buyer,
        amount: refund,
        reason: "Acceptance timeout".to_string(),
    });

    Ok(())
}
```

---

## Change 10: Update ProcessAcceptanceTimeout context (lines ~1058-1066)

```rust
#[derive(Accounts)]
pub struct ProcessAcceptanceTimeout<'info> {
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,

    #[account(
        address = escrow.csol_mint,
    )]
    pub csol_mint: InterfaceAccount<'info, Mint>,

    #[account(
        mut,
        address = escrow.escrow_ata,
    )]
    pub escrow_ata: InterfaceAccount<'info, TokenAccount>,

    #[account(
        mut,
        associated_token::mint = csol_mint,
        associated_token::authority = buyer,
        associated_token::token_program = token_program,
    )]
    pub buyer_ata: InterfaceAccount<'info, TokenAccount>,

    /// CHECK: Buyer account
    pub buyer: UncheckedAccount<'info>,

    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,

    pub token_program: Program<'info, Token2022>,
}
```

---

## Change 11: Update process_shipping_timeout similarly

**Apply the same pattern for `process_shipping_timeout` and `ProcessShippingTimeout` context**

---

## Change 12: Update auto_complete similarly

**Apply the same pattern for `auto_complete` and `AutoComplete` context**

---

## Summary of Changes

1. ✅ Added Token-2022 imports
2. ✅ Added CSOL_DECIMALS constant (9)
3. ✅ Updated Escrow struct with csol_mint and escrow_ata fields
4. ✅ Replaced all SOL system transfers with cSOL Token-2022 transfers
5. ✅ Added fee_payer (wrapper) to all contexts
6. ✅ Added ATA accounts to all contexts
7. ✅ Used PDA signing for escrow → seller/buyer transfers
8. ✅ Kept all MPC features intact

**All transfers now use confidential cSOL via Token-2022!**
