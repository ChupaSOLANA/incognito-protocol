use anchor_lang::prelude::*;
use arcium_anchor::prelude::*;

const COMP_DEF_OFFSET_ENCRYPT_SHIPPING: u32 = comp_def_offset("encrypt_shipping_address");
const COMP_DEF_OFFSET_CALC_REPUTATION: u32 = comp_def_offset("calculate_reputation_score");
const COMP_DEF_OFFSET_VERIFY_AMOUNT: u32 = comp_def_offset("verify_escrow_amount");
const COMP_DEF_OFFSET_CALC_STAKE: u32 = comp_def_offset("calculate_seller_stake");
const COMP_DEF_OFFSET_CALC_FEE: u32 = comp_def_offset("calculate_platform_fee");
const COMP_DEF_OFFSET_CALC_REFUND: u32 = comp_def_offset("calculate_refund_amount");
const COMP_DEF_OFFSET_CALC_COMPLETION: u32 = comp_def_offset("calculate_completion_distribution");
const COMP_DEF_OFFSET_CALC_BUYER_WIN: u32 = comp_def_offset("calculate_buyer_dispute_win");
const COMP_DEF_OFFSET_CALC_SELLER_WIN: u32 = comp_def_offset("calculate_seller_dispute_win");
const COMP_DEF_OFFSET_VERIFY_BALANCE: u32 = comp_def_offset("verify_sufficient_balance");
const COMP_DEF_OFFSET_CHECK_TIMEOUT: u32 = comp_def_offset("check_timeout");

const CSOL_DECIMALS: u8 = 9;

declare_id!("5QvQbnrL7fKpM5pCMS3zNqgTK8ALNkgHgRvgd49YF7v4");

#[arcium_program]
pub mod escrow {
    use super::*;

    pub fn init_encrypt_shipping_comp_def(ctx: Context<InitEncryptShippingCompDef>) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn init_reputation_calc_comp_def(ctx: Context<InitReputationCalcCompDef>) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn init_verify_amount_comp_def(ctx: Context<InitVerifyAmountCompDef>) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn init_calc_stake_comp_def(ctx: Context<InitCalcStakeCompDef>) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn init_calc_fee_comp_def(ctx: Context<InitCalcFeeCompDef>) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn init_calc_refund_comp_def(ctx: Context<InitCalcRefundCompDef>) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn init_calc_completion_comp_def(ctx: Context<InitCalcCompletionCompDef>) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn init_calc_buyer_win_comp_def(ctx: Context<InitCalcBuyerWinCompDef>) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn init_calc_seller_win_comp_def(ctx: Context<InitCalcSellerWinCompDef>) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn init_verify_balance_comp_def(ctx: Context<InitVerifyBalanceCompDef>) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn init_check_timeout_comp_def(ctx: Context<InitCheckTimeoutCompDef>) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn initialize_platform(
        ctx: Context<InitializePlatform>,
        treasury: Pubkey,
    ) -> Result<()> {
        let config = &mut ctx.accounts.config;
        config.authority = ctx.accounts.authority.key();
        config.treasury = treasury;
        config.platform_fee_bps = 200;
        config.seller_stake_bps = 1000;
        config.acceptance_deadline = 86400;
        config.shipping_deadline = 604800;
        config.delivery_deadline = 1209600;
        config.dispute_window = 604800;
        config.arbiter_deadline = 259200;
        config.bump = ctx.bumps.config;
        Ok(())
    }

    pub fn initialize_arbiter_pool(ctx: Context<InitializeArbiterPool>) -> Result<()> {
        let pool = &mut ctx.accounts.arbiter_pool;
        pool.authority = ctx.accounts.authority.key();
        pool.arbiters = Vec::new();
        pool.arbiter_stakes = Vec::new();
        pool.bump = ctx.bumps.arbiter_pool;
        Ok(())
    }

    pub fn add_arbiter(
        ctx: Context<AddArbiter>,
        arbiter: Pubkey,
        stake: u64,
    ) -> Result<()> {
        let pool = &mut ctx.accounts.arbiter_pool;

        require!(
            ctx.accounts.authority.key() == pool.authority,
            ErrorCode::Unauthorized
        );
        require!(stake > 0, ErrorCode::InvalidAmount);
        require!(
            !pool.arbiters.contains(&arbiter),
            ErrorCode::ArbiterAlreadyExists
        );

        pool.arbiters.push(arbiter);
        pool.arbiter_stakes.push(stake);
        Ok(())
    }

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

        let arbiter_index = (clock.unix_timestamp as usize) % ctx.accounts.arbiter_pool.arbiters.len();
        let arbiter = ctx.accounts.arbiter_pool.arbiters[arbiter_index];

        escrow.buyer = ctx.accounts.buyer.key();
        escrow.seller = Pubkey::default();
        escrow.arbiter = arbiter;
        escrow.amount = amount;
        escrow.seller_stake = 0;
        escrow.platform_fee = 0;
        escrow.amount_commitment = amount_commitment.unwrap_or([0u8; 32]);
        escrow.state = EscrowState::Created;
        escrow.bump = ctx.bumps.escrow;
        escrow.order_id = order_id;
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
        escrow.csol_mint = Pubkey::default();
        escrow.escrow_ata = Pubkey::default();

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

    pub fn mark_shipped(
        ctx: Context<MarkShipped>,
        tracking_number: String,
    ) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;
        let config = &ctx.accounts.config;
        let clock = Clock::get()?;

        require!(
            escrow.state == EscrowState::Accepted,
            ErrorCode::InvalidState
        );

        require!(
            ctx.accounts.seller.key() == escrow.seller,
            ErrorCode::Unauthorized
        );

        let deadline = escrow.accepted_at + config.shipping_deadline;
        require!(
            clock.unix_timestamp <= deadline,
            ErrorCode::DeadlineExpired
        );

        require!(
            tracking_number.len() > 0 && tracking_number.len() <= 64,
            ErrorCode::InvalidTracking
        );

        escrow.tracking_number = tracking_number.clone();
        escrow.state = EscrowState::Shipped;
        escrow.shipped_at = clock.unix_timestamp;

        emit!(OrderShippedEvent {
            buyer: escrow.buyer,
            seller: escrow.seller,
            tracking_number,
        });

        Ok(())
    }

    pub fn confirm_delivery(ctx: Context<ConfirmDelivery>) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;
        let clock = Clock::get()?;

        require!(
            escrow.state == EscrowState::Shipped,
            ErrorCode::InvalidState
        );

        require!(
            ctx.accounts.buyer.key() == escrow.buyer,
            ErrorCode::Unauthorized
        );

        escrow.state = EscrowState::Delivered;
        escrow.delivered_at = clock.unix_timestamp;

        emit!(OrderDeliveredEvent {
            buyer: escrow.buyer,
            seller: escrow.seller,
        });

        Ok(())
    }

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

    pub fn buyer_finalize_early(ctx: Context<BuyerFinalizeEarly>) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;

        require!(
            escrow.state == EscrowState::Delivered,
            ErrorCode::InvalidState
        );

        require!(
            ctx.accounts.buyer_signer.key() == escrow.buyer,
            ErrorCode::Unauthorized
        );

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

    pub fn auto_complete(ctx: Context<AutoComplete>) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;
        let config = &ctx.accounts.config;
        let clock = Clock::get()?;

        require!(
            escrow.state == EscrowState::Shipped,
            ErrorCode::InvalidState
        );

        let deadline = escrow.shipped_at + config.delivery_deadline;
        require!(
            clock.unix_timestamp > deadline,
            ErrorCode::DeadlineNotReached
        );

        escrow.state = EscrowState::Delivered;
        escrow.delivered_at = clock.unix_timestamp;

        let buyer_rep = &mut ctx.accounts.buyer_reputation;
        buyer_rep.total_orders += 1;
        buyer_rep.successful_orders += 1;
        update_reputation_score(buyer_rep);

        let seller_rep = &mut ctx.accounts.seller_reputation;
        seller_rep.total_orders += 1;
        seller_rep.successful_orders += 1;
        update_reputation_score(seller_rep);

        escrow.state = EscrowState::Completed;

        emit!(OrderAutoCompletedEvent {
            buyer: escrow.buyer,
            seller: escrow.seller,
        });

        Ok(())
    }

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

        escrow.state = EscrowState::Refunded;

        emit!(OrderRefundedEvent {
            buyer: escrow.buyer,
            amount: escrow.amount,
            reason: "Acceptance timeout".to_string(),
        });

        Ok(())
    }

    pub fn process_shipping_timeout(ctx: Context<ProcessShippingTimeout>) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;
        let config = &ctx.accounts.config;
        let clock = Clock::get()?;

        require!(
            escrow.state == EscrowState::Accepted,
            ErrorCode::InvalidState
        );

        let deadline = escrow.accepted_at + config.shipping_deadline;
        require!(
            clock.unix_timestamp > deadline,
            ErrorCode::DeadlineNotReached
        );

        let refund = escrow.amount + escrow.seller_stake;

        let buyer_rep = &mut ctx.accounts.buyer_reputation;
        buyer_rep.total_orders += 1;
        buyer_rep.successful_orders += 1;
        update_reputation_score(buyer_rep);

        let seller_rep = &mut ctx.accounts.seller_reputation;
        seller_rep.total_orders += 1;
        seller_rep.disputes_lost += 1;
        update_reputation_score(seller_rep);

        escrow.state = EscrowState::Refunded;

        emit!(OrderRefundedEvent {
            buyer: escrow.buyer,
            amount: refund,
            reason: "Shipping timeout".to_string(),
        });

        Ok(())
    }

    pub fn open_dispute(
        ctx: Context<OpenDispute>,
        reason: String,
    ) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;
        let config = &ctx.accounts.config;
        let clock = Clock::get()?;

        require!(
            escrow.state == EscrowState::Delivered,
            ErrorCode::InvalidState
        );

        require!(
            ctx.accounts.buyer.key() == escrow.buyer,
            ErrorCode::Unauthorized
        );

        let deadline = escrow.delivered_at + config.dispute_window;
        require!(
            clock.unix_timestamp <= deadline,
            ErrorCode::DeadlineExpired
        );

        require!(
            reason.len() > 0 && reason.len() <= 200,
            ErrorCode::InvalidReason
        );

        escrow.dispute_reason = reason.clone();
        escrow.dispute_opened_at = clock.unix_timestamp;
        escrow.state = EscrowState::Disputed;

        let buyer_rep = &mut ctx.accounts.buyer_reputation;
        buyer_rep.disputes_opened += 1;

        emit!(DisputeOpenedEvent {
            buyer: escrow.buyer,
            seller: escrow.seller,
            reason,
        });

        Ok(())
    }

    pub fn resolve_dispute(
        ctx: Context<ResolveDispute>,
        winner: DisputeWinner,
    ) -> Result<()> {
        let escrow = &mut ctx.accounts.escrow;
        let config = &ctx.accounts.config;
        let clock = Clock::get()?;

        require!(
            escrow.state == EscrowState::Disputed,
            ErrorCode::InvalidState
        );

        require!(
            ctx.accounts.arbiter.key() == escrow.arbiter,
            ErrorCode::Unauthorized
        );

        let deadline = escrow.dispute_opened_at + config.arbiter_deadline;
        require!(
            clock.unix_timestamp <= deadline,
            ErrorCode::DeadlineExpired
        );

        match winner {
            DisputeWinner::Buyer => {
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
            DisputeWinner::Seller => {
                let buyer_rep = &mut ctx.accounts.buyer_reputation;
                buyer_rep.total_orders += 1;
                buyer_rep.disputes_lost += 1;
                update_reputation_score(buyer_rep);

                let seller_rep = &mut ctx.accounts.seller_reputation;
                seller_rep.total_orders += 1;
                seller_rep.disputes_won += 1;
                update_reputation_score(seller_rep);

                escrow.state = EscrowState::Completed;
            }
        }

        emit!(DisputeResolvedEvent {
            buyer: escrow.buyer,
            seller: escrow.seller,
            winner: match winner {
                DisputeWinner::Buyer => "Buyer".to_string(),
                DisputeWinner::Seller => "Seller".to_string(),
            },
        });

        Ok(())
    }

    pub fn initialize_reputation(ctx: Context<InitializeReputation>) -> Result<()> {
        let reputation = &mut ctx.accounts.reputation;
        reputation.user = ctx.accounts.user.key();
        reputation.total_orders = 0;
        reputation.successful_orders = 0;
        reputation.disputes_opened = 0;
        reputation.disputes_won = 0;
        reputation.disputes_lost = 0;
        reputation.reputation_score = 500;
        reputation.bump = ctx.bumps.reputation;
        Ok(())
    }

    pub fn calculate_reputation_private(
        ctx: Context<CalculateReputationPrivate>,
        computation_offset: u64,
        encrypted_stats: [[u8; 32]; 4],
        pub_key: [u8; 32],
        nonce: u128,
    ) -> Result<()> {
        ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;

        let args = vec![
            Argument::ArcisPubkey(pub_key),
            Argument::PlaintextU128(nonce),
            Argument::EncryptedU8(encrypted_stats[0]),
            Argument::EncryptedU8(encrypted_stats[1]),
            Argument::EncryptedU8(encrypted_stats[2]),
            Argument::EncryptedU8(encrypted_stats[3]),
        ];

        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            None,
            vec![CalculateReputationScoreCallback::callback_ix(&[])],
        )?;

        Ok(())
    }

    #[arcium_callback(encrypted_ix = "calculate_reputation_score")]
    pub fn calculate_reputation_score_callback(
        ctx: Context<CalculateReputationScoreCallback>,
        output: ComputationOutputs<CalculateReputationScoreOutput>,
    ) -> Result<()> {
        let encrypted_score = match output {
            ComputationOutputs::Success(CalculateReputationScoreOutput { field_0 }) => field_0,
            _ => return Err(ErrorCode::AbortedComputation.into()),
        };

        emit!(ReputationCalculatedPrivateEvent {
            encrypted_score: encrypted_score.ciphertexts[0],
            nonce: encrypted_score.nonce.to_le_bytes(),
        });

        Ok(())
    }

    pub fn verify_escrow_amount_private(
        ctx: Context<VerifyEscrowAmount>,
        computation_offset: u64,
        encrypted_amount: [u8; 32],
        encrypted_min_amount: [u8; 32],
        pub_key: [u8; 32],
        nonce: u128,
    ) -> Result<()> {
        ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;

        let args = vec![
            Argument::ArcisPubkey(pub_key),
            Argument::PlaintextU128(nonce),
            Argument::EncryptedU64(encrypted_amount),
            Argument::EncryptedU64(encrypted_min_amount),
        ];

        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            None,
            vec![VerifyEscrowAmountCallback::callback_ix(&[])],
        )?;

        Ok(())
    }

    #[arcium_callback(encrypted_ix = "verify_escrow_amount")]
    pub fn verify_escrow_amount_callback(
        ctx: Context<VerifyEscrowAmountCallback>,
        output: ComputationOutputs<VerifyEscrowAmountOutput>,
    ) -> Result<()> {
        let result = match output {
            ComputationOutputs::Success(VerifyEscrowAmountOutput { field_0 }) => field_0,
            _ => return Err(ErrorCode::AbortedComputation.into()),
        };

        emit!(AmountVerifiedEvent {
            encrypted_amount: result.field_0.ciphertexts[0],
            nonce: result.field_0.nonce.to_le_bytes(),
            is_valid: result.field_1,
        });

        Ok(())
    }

    pub fn calculate_seller_stake_private(
        ctx: Context<CalculateSellerStake>,
        computation_offset: u64,
        encrypted_order_amount: [u8; 32],
        encrypted_stake_percentage_bps: [u8; 32],
        pub_key: [u8; 32],
        nonce: u128,
    ) -> Result<()> {
        ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;

        let args = vec![
            Argument::ArcisPubkey(pub_key),
            Argument::PlaintextU128(nonce),
            Argument::EncryptedU64(encrypted_order_amount),
            Argument::EncryptedU64(encrypted_stake_percentage_bps),
        ];

        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            None,
            vec![CalculateSellerStakeCallback::callback_ix(&[])],
        )?;

        Ok(())
    }

    #[arcium_callback(encrypted_ix = "calculate_seller_stake")]
    pub fn calculate_seller_stake_callback(
        ctx: Context<CalculateSellerStakeCallback>,
        output: ComputationOutputs<CalculateSellerStakeOutput>,
    ) -> Result<()> {
        let encrypted_stake = match output {
            ComputationOutputs::Success(CalculateSellerStakeOutput { field_0 }) => field_0,
            _ => return Err(ErrorCode::AbortedComputation.into()),
        };

        emit!(SellerStakeCalculatedEvent {
            encrypted_stake: encrypted_stake.ciphertexts[0],
            nonce: encrypted_stake.nonce.to_le_bytes(),
        });

        Ok(())
    }

    pub fn calculate_platform_fee_private(
        ctx: Context<CalculatePlatformFee>,
        computation_offset: u64,
        encrypted_order_amount: [u8; 32],
        encrypted_fee_percentage_bps: [u8; 32],
        pub_key: [u8; 32],
        nonce: u128,
    ) -> Result<()> {
        ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;

        let args = vec![
            Argument::ArcisPubkey(pub_key),
            Argument::PlaintextU128(nonce),
            Argument::EncryptedU64(encrypted_order_amount),
            Argument::EncryptedU64(encrypted_fee_percentage_bps),
        ];

        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            None,
            vec![CalculatePlatformFeeCallback::callback_ix(&[])],
        )?;

        Ok(())
    }

    #[arcium_callback(encrypted_ix = "calculate_platform_fee")]
    pub fn calculate_platform_fee_callback(
        ctx: Context<CalculatePlatformFeeCallback>,
        output: ComputationOutputs<CalculatePlatformFeeOutput>,
    ) -> Result<()> {
        let encrypted_fee = match output {
            ComputationOutputs::Success(CalculatePlatformFeeOutput { field_0 }) => field_0,
            _ => return Err(ErrorCode::AbortedComputation.into()),
        };

        emit!(PlatformFeeCalculatedEvent {
            encrypted_fee: encrypted_fee.ciphertexts[0],
            nonce: encrypted_fee.nonce.to_le_bytes(),
        });

        Ok(())
    }

    pub fn calculate_refund_amount_private(
        ctx: Context<CalculateRefundAmount>,
        computation_offset: u64,
        encrypted_order_amount: [u8; 32],
        encrypted_seller_stake: [u8; 32],
        include_penalty: bool,
        pub_key: [u8; 32],
        nonce: u128,
    ) -> Result<()> {
        ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;

        let args = vec![
            Argument::ArcisPubkey(pub_key),
            Argument::PlaintextU128(nonce),
            Argument::EncryptedU64(encrypted_order_amount),
            Argument::EncryptedU64(encrypted_seller_stake),
            Argument::PlaintextBool(include_penalty),
        ];

        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            None,
            vec![CalculateRefundAmountCallback::callback_ix(&[])],
        )?;

        Ok(())
    }

    #[arcium_callback(encrypted_ix = "calculate_refund_amount")]
    pub fn calculate_refund_amount_callback(
        ctx: Context<CalculateRefundAmountCallback>,
        output: ComputationOutputs<CalculateRefundAmountOutput>,
    ) -> Result<()> {
        let encrypted_refund = match output {
            ComputationOutputs::Success(CalculateRefundAmountOutput { field_0 }) => field_0,
            _ => return Err(ErrorCode::AbortedComputation.into()),
        };

        emit!(RefundAmountCalculatedEvent {
            encrypted_refund: encrypted_refund.ciphertexts[0],
            nonce: encrypted_refund.nonce.to_le_bytes(),
        });

        Ok(())
    }

    pub fn calculate_completion_distribution_private(
        ctx: Context<CalculateCompletionDistribution>,
        computation_offset: u64,
        encrypted_order_amount: [u8; 32],
        encrypted_seller_stake: [u8; 32],
        encrypted_platform_fee: [u8; 32],
        pub_key: [u8; 32],
        nonce: u128,
    ) -> Result<()> {
        ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;

        let args = vec![
            Argument::ArcisPubkey(pub_key),
            Argument::PlaintextU128(nonce),
            Argument::EncryptedU64(encrypted_order_amount),
            Argument::EncryptedU64(encrypted_seller_stake),
            Argument::EncryptedU64(encrypted_platform_fee),
        ];

        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            None,
            vec![CalculateCompletionDistributionCallback::callback_ix(&[])],
        )?;

        Ok(())
    }

    #[arcium_callback(encrypted_ix = "calculate_completion_distribution")]
    pub fn calculate_completion_distribution_callback(
        ctx: Context<CalculateCompletionDistributionCallback>,
        output: ComputationOutputs<CalculateCompletionDistributionOutput>,
    ) -> Result<()> {
        let encrypted_distribution = match output {
            ComputationOutputs::Success(CalculateCompletionDistributionOutput { field_0 }) => field_0,
            _ => return Err(ErrorCode::AbortedComputation.into()),
        };

        emit!(CompletionDistributionCalculatedEvent {
            encrypted_to_seller: encrypted_distribution.ciphertexts[0],
            encrypted_to_platform: encrypted_distribution.ciphertexts[1],
            encrypted_to_buyer: encrypted_distribution.ciphertexts[2],
            nonce: encrypted_distribution.nonce.to_le_bytes(),
        });

        Ok(())
    }

    pub fn calculate_buyer_dispute_win_private(
        ctx: Context<CalculateBuyerDisputeWin>,
        computation_offset: u64,
        encrypted_order_amount: [u8; 32],
        encrypted_seller_stake: [u8; 32],
        encrypted_platform_fee: [u8; 32],
        pub_key: [u8; 32],
        nonce: u128,
    ) -> Result<()> {
        ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;

        let args = vec![
            Argument::ArcisPubkey(pub_key),
            Argument::PlaintextU128(nonce),
            Argument::EncryptedU64(encrypted_order_amount),
            Argument::EncryptedU64(encrypted_seller_stake),
            Argument::EncryptedU64(encrypted_platform_fee),
        ];

        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            None,
            vec![CalculateBuyerDisputeWinCallback::callback_ix(&[])],
        )?;

        Ok(())
    }

    #[arcium_callback(encrypted_ix = "calculate_buyer_dispute_win")]
    pub fn calculate_buyer_dispute_win_callback(
        ctx: Context<CalculateBuyerDisputeWinCallback>,
        output: ComputationOutputs<CalculateBuyerDisputeWinOutput>,
    ) -> Result<()> {
        let encrypted_distribution = match output {
            ComputationOutputs::Success(CalculateBuyerDisputeWinOutput { field_0 }) => field_0,
            _ => return Err(ErrorCode::AbortedComputation.into()),
        };

        emit!(BuyerDisputeWinDistributionEvent {
            encrypted_to_seller: encrypted_distribution.ciphertexts[0],
            encrypted_to_platform: encrypted_distribution.ciphertexts[1],
            encrypted_to_buyer: encrypted_distribution.ciphertexts[2],
            nonce: encrypted_distribution.nonce.to_le_bytes(),
        });

        Ok(())
    }

    pub fn calculate_seller_dispute_win_private(
        ctx: Context<CalculateSellerDisputeWin>,
        computation_offset: u64,
        encrypted_order_amount: [u8; 32],
        encrypted_seller_stake: [u8; 32],
        encrypted_platform_fee: [u8; 32],
        pub_key: [u8; 32],
        nonce: u128,
    ) -> Result<()> {
        ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;

        let args = vec![
            Argument::ArcisPubkey(pub_key),
            Argument::PlaintextU128(nonce),
            Argument::EncryptedU64(encrypted_order_amount),
            Argument::EncryptedU64(encrypted_seller_stake),
            Argument::EncryptedU64(encrypted_platform_fee),
        ];

        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            None,
            vec![CalculateSellerDisputeWinCallback::callback_ix(&[])],
        )?;

        Ok(())
    }

    #[arcium_callback(encrypted_ix = "calculate_seller_dispute_win")]
    pub fn calculate_seller_dispute_win_callback(
        ctx: Context<CalculateSellerDisputeWinCallback>,
        output: ComputationOutputs<CalculateSellerDisputeWinOutput>,
    ) -> Result<()> {
        let encrypted_distribution = match output {
            ComputationOutputs::Success(CalculateSellerDisputeWinOutput { field_0 }) => field_0,
            _ => return Err(ErrorCode::AbortedComputation.into()),
        };

        emit!(SellerDisputeWinDistributionEvent {
            encrypted_to_seller: encrypted_distribution.ciphertexts[0],
            encrypted_to_platform: encrypted_distribution.ciphertexts[1],
            encrypted_to_buyer: encrypted_distribution.ciphertexts[2],
            nonce: encrypted_distribution.nonce.to_le_bytes(),
        });

        Ok(())
    }
}

#[account]
pub struct PlatformConfig {
    pub authority: Pubkey,
    pub treasury: Pubkey,
    pub platform_fee_bps: u16,
    pub seller_stake_bps: u16,
    pub acceptance_deadline: i64,
    pub shipping_deadline: i64,
    pub delivery_deadline: i64,
    pub dispute_window: i64,
    pub arbiter_deadline: i64,
    pub bump: u8,
}

#[account]
pub struct ArbiterPool {
    pub authority: Pubkey,
    pub arbiters: Vec<Pubkey>,
    pub arbiter_stakes: Vec<u64>,
    pub bump: u8,
}

#[account]
pub struct Escrow {
    pub buyer: Pubkey,
    pub seller: Pubkey,
    pub arbiter: Pubkey,
    pub amount: u64,
    pub seller_stake: u64,
    pub platform_fee: u64,
    pub amount_commitment: [u8; 32],
    pub state: EscrowState,
    pub bump: u8,
    pub order_id: u64,
    pub created_at: i64,
    pub accepted_at: i64,
    pub shipped_at: i64,
    pub delivered_at: i64,
    pub dispute_opened_at: i64,
    pub encrypted_shipping: Vec<u8>,
    pub shipping_encryption_nonce: [u8; 16],
    pub tracking_number: String,
    pub dispute_reason: String,
    pub use_private_reputation: bool,
    pub csol_mint: Pubkey,
    pub escrow_ata: Pubkey,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum EscrowState {
    Created,
    Accepted,
    Shipped,
    Delivered,
    Disputed,
    Completed,
    Refunded,
}

#[account]
pub struct UserReputation {
    pub user: Pubkey,
    pub total_orders: u64,
    pub successful_orders: u64,
    pub disputes_opened: u64,
    pub disputes_won: u64,
    pub disputes_lost: u64,
    pub reputation_score: u64,
    pub bump: u8,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub enum DisputeWinner {
    Buyer,
    Seller,
}

#[derive(Accounts)]
pub struct InitializePlatform<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,
    #[account(
        init,
        payer = authority,
        space = 8 + 32 + 32 + 2 + 2 + 8 + 8 + 8 + 8 + 8 + 1,
        seeds = [b"config"],
        bump
    )]
    pub config: Account<'info, PlatformConfig>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct InitializeArbiterPool<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,
    #[account(
        init,
        payer = authority,
        space = 8 + 32 + 4 + (32 * 10) + 4 + (8 * 10) + 1,
        seeds = [b"arbiter_pool"],
        bump
    )]
    pub arbiter_pool: Account<'info, ArbiterPool>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct AddArbiter<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,
    #[account(mut)]
    pub arbiter_pool: Account<'info, ArbiterPool>,
}

#[derive(Accounts)]
#[instruction(amount: u64, order_id: u64, encrypted_shipping: Vec<u8>, shipping_nonce: [u8; 16])]
pub struct CreateOrder<'info> {
    #[account(mut)]
    pub buyer: Signer<'info>,

    #[account(
        init,
        payer = fee_payer,
        space = 8 + 32 + 32 + 32 + 8 + 8 + 8 + 32 + 1 + 1 + 8 + 8 + 8 + 8 + 8 + 8 + 260 + 16 + 68 + 204 + 1 + 32 + 32,
        seeds = [b"escrow", buyer.key().as_ref(), &order_id.to_le_bytes()],
        bump
    )]
    pub escrow: Account<'info, Escrow>,

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

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct AcceptOrder<'info> {
    #[account(mut)]
    pub seller: Signer<'info>,

    #[account(mut)]
    pub escrow: Account<'info, Escrow>,

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

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct MarkShipped<'info> {
    pub seller: Signer<'info>,
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,
    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,
}

#[derive(Accounts)]
pub struct ConfirmDelivery<'info> {
    pub buyer: Signer<'info>,
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,
}

#[derive(Accounts)]
pub struct FinalizeOrder<'info> {
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,

    /// CHECK: Buyer pubkey verified against escrow.buyer
    pub buyer: UncheckedAccount<'info>,

    /// CHECK: Seller pubkey verified against escrow.seller
    pub seller: UncheckedAccount<'info>,

    #[account(mut)]
    pub buyer_reputation: Account<'info, UserReputation>,

    #[account(mut)]
    pub seller_reputation: Account<'info, UserReputation>,

    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,
}

#[derive(Accounts)]
pub struct BuyerFinalizeEarly<'info> {
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,

    pub buyer_signer: Signer<'info>,

    /// CHECK: Buyer pubkey verified against escrow.buyer
    pub buyer: UncheckedAccount<'info>,

    /// CHECK: Seller pubkey verified against escrow.seller
    pub seller: UncheckedAccount<'info>,

    #[account(mut)]
    pub buyer_reputation: Account<'info, UserReputation>,

    #[account(mut)]
    pub seller_reputation: Account<'info, UserReputation>,

    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,
}

#[derive(Accounts)]
pub struct AutoComplete<'info> {
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,

    /// CHECK: Buyer pubkey verified against escrow.buyer
    pub buyer: UncheckedAccount<'info>,

    /// CHECK: Seller pubkey verified against escrow.seller
    pub seller: UncheckedAccount<'info>,

    #[account(mut)]
    pub buyer_reputation: Account<'info, UserReputation>,

    #[account(mut)]
    pub seller_reputation: Account<'info, UserReputation>,

    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,
}

#[derive(Accounts)]
pub struct ProcessAcceptanceTimeout<'info> {
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,

    /// CHECK: Buyer pubkey verified against escrow.buyer
    pub buyer: UncheckedAccount<'info>,

    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,
}

#[derive(Accounts)]
pub struct ProcessShippingTimeout<'info> {
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,

    /// CHECK: Buyer pubkey verified against escrow.buyer
    pub buyer: UncheckedAccount<'info>,

    /// CHECK: Seller pubkey verified against escrow.seller
    pub seller: UncheckedAccount<'info>,

    #[account(mut)]
    pub buyer_reputation: Account<'info, UserReputation>,

    #[account(mut)]
    pub seller_reputation: Account<'info, UserReputation>,

    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,
}

#[derive(Accounts)]
pub struct OpenDispute<'info> {
    pub buyer: Signer<'info>,
    #[account(mut)]
    pub escrow: Account<'info, Escrow>,
    #[account(mut)]
    pub buyer_reputation: Account<'info, UserReputation>,
    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,
}

#[derive(Accounts)]
pub struct ResolveDispute<'info> {
    pub arbiter: Signer<'info>,

    #[account(mut)]
    pub escrow: Account<'info, Escrow>,

    /// CHECK: Buyer pubkey verified against escrow.buyer
    pub buyer: UncheckedAccount<'info>,

    /// CHECK: Seller pubkey verified against escrow.seller
    pub seller: UncheckedAccount<'info>,

    #[account(mut)]
    pub buyer_reputation: Account<'info, UserReputation>,

    #[account(mut)]
    pub seller_reputation: Account<'info, UserReputation>,

    #[account(seeds = [b"config"], bump = config.bump)]
    pub config: Account<'info, PlatformConfig>,
}

#[derive(Accounts)]
pub struct InitializeReputation<'info> {
    #[account(mut)]
    pub user: Signer<'info>,
    #[account(
        init,
        payer = user,
        space = 8 + 32 + 8 + 8 + 8 + 8 + 8 + 8 + 1,
        seeds = [b"reputation", user.key().as_ref()],
        bump
    )]
    pub reputation: Account<'info, UserReputation>,
    pub system_program: Program<'info, System>,
}

#[queue_computation_accounts("calculate_reputation_score", payer)]
#[derive(Accounts)]
#[instruction(computation_offset: u64)]
pub struct CalculateReputationPrivate<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(
        init_if_needed,
        space = 9,
        payer = payer,
        seeds = [&SIGN_PDA_SEED],
        bump,
        address = derive_sign_pda!(),
    )]
    pub sign_pda_account: Account<'info, SignerAccount>,
    #[account(
        address = derive_mxe_pda!()
    )]
    pub mxe_account: Account<'info, MXEAccount>,
    /// CHECK: Mempool account managed by Arcium program
    #[account(
        mut,
        address = derive_mempool_pda!()
    )]
    pub mempool_account: UncheckedAccount<'info>,
    /// CHECK: Execution pool account managed by Arcium program
    #[account(
        mut,
        address = derive_execpool_pda!()
    )]
    pub executing_pool: UncheckedAccount<'info>,
    /// CHECK: Computation account managed by Arcium program
    #[account(
        mut,
        address = derive_comp_pda!(computation_offset)
    )]
    pub computation_account: UncheckedAccount<'info>,
    #[account(
        address = derive_comp_def_pda!(COMP_DEF_OFFSET_CALC_REPUTATION)
    )]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    #[account(
        mut,
        address = derive_cluster_pda!(mxe_account)
    )]
    pub cluster_account: Account<'info, Cluster>,
    #[account(
        mut,
        address = ARCIUM_FEE_POOL_ACCOUNT_ADDRESS,
    )]
    pub pool_account: Account<'info, FeePool>,
    #[account(
        address = ARCIUM_CLOCK_ACCOUNT_ADDRESS
    )]
    pub clock_account: Account<'info, ClockAccount>,
    pub system_program: Program<'info, System>,
    pub arcium_program: Program<'info, Arcium>,
}

#[callback_accounts("calculate_reputation_score")]
#[derive(Accounts)]
pub struct CalculateReputationScoreCallback<'info> {
    pub arcium_program: Program<'info, Arcium>,
    #[account(
        address = derive_comp_def_pda!(COMP_DEF_OFFSET_CALC_REPUTATION)
    )]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    /// CHECK: Instructions sysvar account
    #[account(address = ::anchor_lang::solana_program::sysvar::instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

#[init_computation_definition_accounts("encrypt_shipping_address", payer)]
#[derive(Accounts)]
pub struct InitEncryptShippingCompDef<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(
        mut,
        address = derive_mxe_pda!()
    )]
    pub mxe_account: Box<Account<'info, MXEAccount>>,
    /// CHECK: Computation definition account managed by Arcium program
    #[account(mut)]
    pub comp_def_account: UncheckedAccount<'info>,
    pub arcium_program: Program<'info, Arcium>,
    pub system_program: Program<'info, System>,
}

#[init_computation_definition_accounts("calculate_reputation_score", payer)]
#[derive(Accounts)]
pub struct InitReputationCalcCompDef<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(
        mut,
        address = derive_mxe_pda!()
    )]
    pub mxe_account: Box<Account<'info, MXEAccount>>,
    /// CHECK: Computation definition account managed by Arcium program
    #[account(mut)]
    pub comp_def_account: UncheckedAccount<'info>,
    pub arcium_program: Program<'info, Arcium>,
    pub system_program: Program<'info, System>,
}

#[init_computation_definition_accounts("verify_escrow_amount", payer)]
#[derive(Accounts)]
pub struct InitVerifyAmountCompDef<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(mut, address = derive_mxe_pda!())]
    pub mxe_account: Box<Account<'info, MXEAccount>>,
    /// CHECK: Computation definition account managed by Arcium program
    #[account(mut)]
    pub comp_def_account: UncheckedAccount<'info>,
    pub arcium_program: Program<'info, Arcium>,
    pub system_program: Program<'info, System>,
}

#[init_computation_definition_accounts("calculate_seller_stake", payer)]
#[derive(Accounts)]
pub struct InitCalcStakeCompDef<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(mut, address = derive_mxe_pda!())]
    pub mxe_account: Box<Account<'info, MXEAccount>>,
    /// CHECK: Computation definition account managed by Arcium program
    #[account(mut)]
    pub comp_def_account: UncheckedAccount<'info>,
    pub arcium_program: Program<'info, Arcium>,
    pub system_program: Program<'info, System>,
}

#[init_computation_definition_accounts("calculate_platform_fee", payer)]
#[derive(Accounts)]
pub struct InitCalcFeeCompDef<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(mut, address = derive_mxe_pda!())]
    pub mxe_account: Box<Account<'info, MXEAccount>>,
    /// CHECK: Computation definition account managed by Arcium program
    #[account(mut)]
    pub comp_def_account: UncheckedAccount<'info>,
    pub arcium_program: Program<'info, Arcium>,
    pub system_program: Program<'info, System>,
}

#[init_computation_definition_accounts("calculate_refund_amount", payer)]
#[derive(Accounts)]
pub struct InitCalcRefundCompDef<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(mut, address = derive_mxe_pda!())]
    pub mxe_account: Box<Account<'info, MXEAccount>>,
    /// CHECK: Computation definition account managed by Arcium program
    #[account(mut)]
    pub comp_def_account: UncheckedAccount<'info>,
    pub arcium_program: Program<'info, Arcium>,
    pub system_program: Program<'info, System>,
}

#[init_computation_definition_accounts("calculate_completion_distribution", payer)]
#[derive(Accounts)]
pub struct InitCalcCompletionCompDef<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(mut, address = derive_mxe_pda!())]
    pub mxe_account: Box<Account<'info, MXEAccount>>,
    /// CHECK: Computation definition account managed by Arcium program
    #[account(mut)]
    pub comp_def_account: UncheckedAccount<'info>,
    pub arcium_program: Program<'info, Arcium>,
    pub system_program: Program<'info, System>,
}

#[init_computation_definition_accounts("calculate_buyer_dispute_win", payer)]
#[derive(Accounts)]
pub struct InitCalcBuyerWinCompDef<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(mut, address = derive_mxe_pda!())]
    pub mxe_account: Box<Account<'info, MXEAccount>>,
    /// CHECK: Computation definition account managed by Arcium program
    #[account(mut)]
    pub comp_def_account: UncheckedAccount<'info>,
    pub arcium_program: Program<'info, Arcium>,
    pub system_program: Program<'info, System>,
}

#[init_computation_definition_accounts("calculate_seller_dispute_win", payer)]
#[derive(Accounts)]
pub struct InitCalcSellerWinCompDef<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(mut, address = derive_mxe_pda!())]
    pub mxe_account: Box<Account<'info, MXEAccount>>,
    /// CHECK: Computation definition account managed by Arcium program
    #[account(mut)]
    pub comp_def_account: UncheckedAccount<'info>,
    pub arcium_program: Program<'info, Arcium>,
    pub system_program: Program<'info, System>,
}

#[init_computation_definition_accounts("verify_sufficient_balance", payer)]
#[derive(Accounts)]
pub struct InitVerifyBalanceCompDef<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(mut, address = derive_mxe_pda!())]
    pub mxe_account: Box<Account<'info, MXEAccount>>,
    /// CHECK: Computation definition account managed by Arcium program
    #[account(mut)]
    pub comp_def_account: UncheckedAccount<'info>,
    pub arcium_program: Program<'info, Arcium>,
    pub system_program: Program<'info, System>,
}

#[init_computation_definition_accounts("check_timeout", payer)]
#[derive(Accounts)]
pub struct InitCheckTimeoutCompDef<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(mut, address = derive_mxe_pda!())]
    pub mxe_account: Box<Account<'info, MXEAccount>>,
    /// CHECK: Computation definition account managed by Arcium program
    #[account(mut)]
    pub comp_def_account: UncheckedAccount<'info>,
    pub arcium_program: Program<'info, Arcium>,
    pub system_program: Program<'info, System>,
}

// Queue computation account structs
#[queue_computation_accounts("verify_escrow_amount", payer)]
#[derive(Accounts)]
#[instruction(computation_offset: u64)]
pub struct VerifyEscrowAmount<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(init_if_needed, space = 9, payer = payer, seeds = [&SIGN_PDA_SEED], bump, address = derive_sign_pda!())]
    pub sign_pda_account: Account<'info, SignerAccount>,
    #[account(address = derive_mxe_pda!())]
    pub mxe_account: Account<'info, MXEAccount>,
    /// CHECK: Mempool account managed by Arcium program
    #[account(mut, address = derive_mempool_pda!())]
    pub mempool_account: UncheckedAccount<'info>,
    /// CHECK: Execution pool account managed by Arcium program
    #[account(mut, address = derive_execpool_pda!())]
    pub executing_pool: UncheckedAccount<'info>,
    /// CHECK: Computation account managed by Arcium program
    #[account(mut, address = derive_comp_pda!(computation_offset))]
    pub computation_account: UncheckedAccount<'info>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_VERIFY_AMOUNT))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    #[account(mut, address = derive_cluster_pda!(mxe_account))]
    pub cluster_account: Account<'info, Cluster>,
    #[account(mut, address = ARCIUM_FEE_POOL_ACCOUNT_ADDRESS)]
    pub pool_account: Account<'info, FeePool>,
    #[account(address = ARCIUM_CLOCK_ACCOUNT_ADDRESS)]
    pub clock_account: Account<'info, ClockAccount>,
    pub system_program: Program<'info, System>,
    pub arcium_program: Program<'info, Arcium>,
}

#[callback_accounts("verify_escrow_amount")]
#[derive(Accounts)]
pub struct VerifyEscrowAmountCallback<'info> {
    pub arcium_program: Program<'info, Arcium>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_VERIFY_AMOUNT))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    /// CHECK: Instructions sysvar account
    #[account(address = ::anchor_lang::solana_program::sysvar::instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

#[queue_computation_accounts("calculate_seller_stake", payer)]
#[derive(Accounts)]
#[instruction(computation_offset: u64)]
pub struct CalculateSellerStake<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(init_if_needed, space = 9, payer = payer, seeds = [&SIGN_PDA_SEED], bump, address = derive_sign_pda!())]
    pub sign_pda_account: Account<'info, SignerAccount>,
    #[account(address = derive_mxe_pda!())]
    pub mxe_account: Account<'info, MXEAccount>,
    /// CHECK: Mempool account managed by Arcium program
    #[account(mut, address = derive_mempool_pda!())]
    pub mempool_account: UncheckedAccount<'info>,
    /// CHECK: Execution pool account managed by Arcium program
    #[account(mut, address = derive_execpool_pda!())]
    pub executing_pool: UncheckedAccount<'info>,
    /// CHECK: Computation account managed by Arcium program
    #[account(mut, address = derive_comp_pda!(computation_offset))]
    pub computation_account: UncheckedAccount<'info>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CALC_STAKE))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    #[account(mut, address = derive_cluster_pda!(mxe_account))]
    pub cluster_account: Account<'info, Cluster>,
    #[account(mut, address = ARCIUM_FEE_POOL_ACCOUNT_ADDRESS)]
    pub pool_account: Account<'info, FeePool>,
    #[account(address = ARCIUM_CLOCK_ACCOUNT_ADDRESS)]
    pub clock_account: Account<'info, ClockAccount>,
    pub system_program: Program<'info, System>,
    pub arcium_program: Program<'info, Arcium>,
}

#[callback_accounts("calculate_seller_stake")]
#[derive(Accounts)]
pub struct CalculateSellerStakeCallback<'info> {
    pub arcium_program: Program<'info, Arcium>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CALC_STAKE))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    /// CHECK: Instructions sysvar account
    #[account(address = ::anchor_lang::solana_program::sysvar::instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

#[queue_computation_accounts("calculate_platform_fee", payer)]
#[derive(Accounts)]
#[instruction(computation_offset: u64)]
pub struct CalculatePlatformFee<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(init_if_needed, space = 9, payer = payer, seeds = [&SIGN_PDA_SEED], bump, address = derive_sign_pda!())]
    pub sign_pda_account: Account<'info, SignerAccount>,
    #[account(address = derive_mxe_pda!())]
    pub mxe_account: Account<'info, MXEAccount>,
    /// CHECK: Mempool account managed by Arcium program
    #[account(mut, address = derive_mempool_pda!())]
    pub mempool_account: UncheckedAccount<'info>,
    /// CHECK: Execution pool account managed by Arcium program
    #[account(mut, address = derive_execpool_pda!())]
    pub executing_pool: UncheckedAccount<'info>,
    /// CHECK: Computation account managed by Arcium program
    #[account(mut, address = derive_comp_pda!(computation_offset))]
    pub computation_account: UncheckedAccount<'info>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CALC_FEE))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    #[account(mut, address = derive_cluster_pda!(mxe_account))]
    pub cluster_account: Account<'info, Cluster>,
    #[account(mut, address = ARCIUM_FEE_POOL_ACCOUNT_ADDRESS)]
    pub pool_account: Account<'info, FeePool>,
    #[account(address = ARCIUM_CLOCK_ACCOUNT_ADDRESS)]
    pub clock_account: Account<'info, ClockAccount>,
    pub system_program: Program<'info, System>,
    pub arcium_program: Program<'info, Arcium>,
}

#[callback_accounts("calculate_platform_fee")]
#[derive(Accounts)]
pub struct CalculatePlatformFeeCallback<'info> {
    pub arcium_program: Program<'info, Arcium>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CALC_FEE))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    /// CHECK: Instructions sysvar account
    #[account(address = ::anchor_lang::solana_program::sysvar::instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

#[queue_computation_accounts("calculate_refund_amount", payer)]
#[derive(Accounts)]
#[instruction(computation_offset: u64)]
pub struct CalculateRefundAmount<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(init_if_needed, space = 9, payer = payer, seeds = [&SIGN_PDA_SEED], bump, address = derive_sign_pda!())]
    pub sign_pda_account: Account<'info, SignerAccount>,
    #[account(address = derive_mxe_pda!())]
    pub mxe_account: Account<'info, MXEAccount>,
    /// CHECK: Mempool account managed by Arcium program
    #[account(mut, address = derive_mempool_pda!())]
    pub mempool_account: UncheckedAccount<'info>,
    /// CHECK: Execution pool account managed by Arcium program
    #[account(mut, address = derive_execpool_pda!())]
    pub executing_pool: UncheckedAccount<'info>,
    /// CHECK: Computation account managed by Arcium program
    #[account(mut, address = derive_comp_pda!(computation_offset))]
    pub computation_account: UncheckedAccount<'info>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CALC_REFUND))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    #[account(mut, address = derive_cluster_pda!(mxe_account))]
    pub cluster_account: Account<'info, Cluster>,
    #[account(mut, address = ARCIUM_FEE_POOL_ACCOUNT_ADDRESS)]
    pub pool_account: Account<'info, FeePool>,
    #[account(address = ARCIUM_CLOCK_ACCOUNT_ADDRESS)]
    pub clock_account: Account<'info, ClockAccount>,
    pub system_program: Program<'info, System>,
    pub arcium_program: Program<'info, Arcium>,
}

#[callback_accounts("calculate_refund_amount")]
#[derive(Accounts)]
pub struct CalculateRefundAmountCallback<'info> {
    pub arcium_program: Program<'info, Arcium>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CALC_REFUND))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    /// CHECK: Instructions sysvar account
    #[account(address = ::anchor_lang::solana_program::sysvar::instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

#[queue_computation_accounts("calculate_completion_distribution", payer)]
#[derive(Accounts)]
#[instruction(computation_offset: u64)]
pub struct CalculateCompletionDistribution<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(init_if_needed, space = 9, payer = payer, seeds = [&SIGN_PDA_SEED], bump, address = derive_sign_pda!())]
    pub sign_pda_account: Account<'info, SignerAccount>,
    #[account(address = derive_mxe_pda!())]
    pub mxe_account: Account<'info, MXEAccount>,
    /// CHECK: Mempool account managed by Arcium program
    #[account(mut, address = derive_mempool_pda!())]
    pub mempool_account: UncheckedAccount<'info>,
    /// CHECK: Execution pool account managed by Arcium program
    #[account(mut, address = derive_execpool_pda!())]
    pub executing_pool: UncheckedAccount<'info>,
    /// CHECK: Computation account managed by Arcium program
    #[account(mut, address = derive_comp_pda!(computation_offset))]
    pub computation_account: UncheckedAccount<'info>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CALC_COMPLETION))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    #[account(mut, address = derive_cluster_pda!(mxe_account))]
    pub cluster_account: Account<'info, Cluster>,
    #[account(mut, address = ARCIUM_FEE_POOL_ACCOUNT_ADDRESS)]
    pub pool_account: Account<'info, FeePool>,
    #[account(address = ARCIUM_CLOCK_ACCOUNT_ADDRESS)]
    pub clock_account: Account<'info, ClockAccount>,
    pub system_program: Program<'info, System>,
    pub arcium_program: Program<'info, Arcium>,
}

#[callback_accounts("calculate_completion_distribution")]
#[derive(Accounts)]
pub struct CalculateCompletionDistributionCallback<'info> {
    pub arcium_program: Program<'info, Arcium>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CALC_COMPLETION))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    /// CHECK: Instructions sysvar account
    #[account(address = ::anchor_lang::solana_program::sysvar::instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

#[queue_computation_accounts("calculate_buyer_dispute_win", payer)]
#[derive(Accounts)]
#[instruction(computation_offset: u64)]
pub struct CalculateBuyerDisputeWin<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(init_if_needed, space = 9, payer = payer, seeds = [&SIGN_PDA_SEED], bump, address = derive_sign_pda!())]
    pub sign_pda_account: Account<'info, SignerAccount>,
    #[account(address = derive_mxe_pda!())]
    pub mxe_account: Account<'info, MXEAccount>,
    /// CHECK: Mempool account managed by Arcium program
    #[account(mut, address = derive_mempool_pda!())]
    pub mempool_account: UncheckedAccount<'info>,
    /// CHECK: Execution pool account managed by Arcium program
    #[account(mut, address = derive_execpool_pda!())]
    pub executing_pool: UncheckedAccount<'info>,
    /// CHECK: Computation account managed by Arcium program
    #[account(mut, address = derive_comp_pda!(computation_offset))]
    pub computation_account: UncheckedAccount<'info>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CALC_BUYER_WIN))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    #[account(mut, address = derive_cluster_pda!(mxe_account))]
    pub cluster_account: Account<'info, Cluster>,
    #[account(mut, address = ARCIUM_FEE_POOL_ACCOUNT_ADDRESS)]
    pub pool_account: Account<'info, FeePool>,
    #[account(address = ARCIUM_CLOCK_ACCOUNT_ADDRESS)]
    pub clock_account: Account<'info, ClockAccount>,
    pub system_program: Program<'info, System>,
    pub arcium_program: Program<'info, Arcium>,
}

#[callback_accounts("calculate_buyer_dispute_win")]
#[derive(Accounts)]
pub struct CalculateBuyerDisputeWinCallback<'info> {
    pub arcium_program: Program<'info, Arcium>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CALC_BUYER_WIN))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    /// CHECK: Instructions sysvar account
    #[account(address = ::anchor_lang::solana_program::sysvar::instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

#[queue_computation_accounts("calculate_seller_dispute_win", payer)]
#[derive(Accounts)]
#[instruction(computation_offset: u64)]
pub struct CalculateSellerDisputeWin<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(init_if_needed, space = 9, payer = payer, seeds = [&SIGN_PDA_SEED], bump, address = derive_sign_pda!())]
    pub sign_pda_account: Account<'info, SignerAccount>,
    #[account(address = derive_mxe_pda!())]
    pub mxe_account: Account<'info, MXEAccount>,
    /// CHECK: Mempool account managed by Arcium program
    #[account(mut, address = derive_mempool_pda!())]
    pub mempool_account: UncheckedAccount<'info>,
    /// CHECK: Execution pool account managed by Arcium program
    #[account(mut, address = derive_execpool_pda!())]
    pub executing_pool: UncheckedAccount<'info>,
    /// CHECK: Computation account managed by Arcium program
    #[account(mut, address = derive_comp_pda!(computation_offset))]
    pub computation_account: UncheckedAccount<'info>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CALC_SELLER_WIN))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    #[account(mut, address = derive_cluster_pda!(mxe_account))]
    pub cluster_account: Account<'info, Cluster>,
    #[account(mut, address = ARCIUM_FEE_POOL_ACCOUNT_ADDRESS)]
    pub pool_account: Account<'info, FeePool>,
    #[account(address = ARCIUM_CLOCK_ACCOUNT_ADDRESS)]
    pub clock_account: Account<'info, ClockAccount>,
    pub system_program: Program<'info, System>,
    pub arcium_program: Program<'info, Arcium>,
}

#[callback_accounts("calculate_seller_dispute_win")]
#[derive(Accounts)]
pub struct CalculateSellerDisputeWinCallback<'info> {
    pub arcium_program: Program<'info, Arcium>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_CALC_SELLER_WIN))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    /// CHECK: Instructions sysvar account
    #[account(address = ::anchor_lang::solana_program::sysvar::instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

#[event]
pub struct OrderCreatedEvent {
    pub order_id: u64,
    pub buyer: Pubkey,
    pub amount: u64,
    pub arbiter: Pubkey,
}

#[event]
pub struct OrderAcceptedEvent {
    pub buyer: Pubkey,
    pub seller: Pubkey,
    pub amount: u64,
}

#[event]
pub struct OrderShippedEvent {
    pub buyer: Pubkey,
    pub seller: Pubkey,
    pub tracking_number: String,
}

#[event]
pub struct OrderDeliveredEvent {
    pub buyer: Pubkey,
    pub seller: Pubkey,
}

#[event]
pub struct OrderCompletedEvent {
    pub buyer: Pubkey,
    pub seller: Pubkey,
    pub amount: u64,
}

#[event]
pub struct OrderAutoCompletedEvent {
    pub buyer: Pubkey,
    pub seller: Pubkey,
}

#[event]
pub struct OrderRefundedEvent {
    pub buyer: Pubkey,
    pub amount: u64,
    pub reason: String,
}

#[event]
pub struct DisputeOpenedEvent {
    pub buyer: Pubkey,
    pub seller: Pubkey,
    pub reason: String,
}

#[event]
pub struct DisputeResolvedEvent {
    pub buyer: Pubkey,
    pub seller: Pubkey,
    pub winner: String,
}

#[event]
pub struct ReputationCalculatedPrivateEvent {
    pub encrypted_score: [u8; 32],
    pub nonce: [u8; 16],
}

#[event]
pub struct PrivateReputationUpdateNeeded {
    pub buyer: Pubkey,
    pub seller: Pubkey,
    pub msg: String,
}

#[event]
pub struct AmountVerifiedEvent {
    pub encrypted_amount: [u8; 32],
    pub nonce: [u8; 16],
    pub is_valid: bool,
}

#[event]
pub struct SellerStakeCalculatedEvent {
    pub encrypted_stake: [u8; 32],
    pub nonce: [u8; 16],
}

#[event]
pub struct PlatformFeeCalculatedEvent {
    pub encrypted_fee: [u8; 32],
    pub nonce: [u8; 16],
}

#[event]
pub struct RefundAmountCalculatedEvent {
    pub encrypted_refund: [u8; 32],
    pub nonce: [u8; 16],
}

#[event]
pub struct CompletionDistributionCalculatedEvent {
    pub encrypted_to_seller: [u8; 32],
    pub encrypted_to_platform: [u8; 32],
    pub encrypted_to_buyer: [u8; 32],
    pub nonce: [u8; 16],
}

#[event]
pub struct BuyerDisputeWinDistributionEvent {
    pub encrypted_to_seller: [u8; 32],
    pub encrypted_to_platform: [u8; 32],
    pub encrypted_to_buyer: [u8; 32],
    pub nonce: [u8; 16],
}

#[event]
pub struct SellerDisputeWinDistributionEvent {
    pub encrypted_to_seller: [u8; 32],
    pub encrypted_to_platform: [u8; 32],
    pub encrypted_to_buyer: [u8; 32],
    pub nonce: [u8; 16],
}

fn update_reputation_score(rep: &mut UserReputation) {
    if rep.total_orders == 0 {
        rep.reputation_score = 500;
        return;
    }

    let success_rate = (rep.successful_orders * 100) / rep.total_orders;
    let penalty = rep.disputes_lost * 50;
    let bonus = rep.disputes_won * 10;

    let raw_score = success_rate + bonus;
    let with_penalty = if raw_score >= penalty {
        raw_score - penalty
    } else {
        0
    };

    rep.reputation_score = if with_penalty > 1000 {
        1000
    } else {
        with_penalty
    };
}

#[error_code]
pub enum ErrorCode {
    #[msg("The computation was aborted")]
    AbortedComputation,
    #[msg("Cluster not set")]
    ClusterNotSet,
    #[msg("Invalid escrow state for this operation")]
    InvalidState,
    #[msg("Deadline has expired")]
    DeadlineExpired,
    #[msg("Deadline has not been reached yet")]
    DeadlineNotReached,
    #[msg("Unauthorized: wrong signer")]
    Unauthorized,
    #[msg("Invalid amount: must be greater than 0")]
    InvalidAmount,
    #[msg("Invalid tracking number")]
    InvalidTracking,
    #[msg("Invalid dispute reason")]
    InvalidReason,
    #[msg("No arbiters available in pool")]
    NoArbitersAvailable,
    #[msg("Shipping data too large (max 256 bytes)")]
    ShippingDataTooLarge,
    #[msg("Arbiter already exists in pool")]
    ArbiterAlreadyExists,
}
