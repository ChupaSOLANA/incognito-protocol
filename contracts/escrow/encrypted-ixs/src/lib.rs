use arcis_imports::*;

#[encrypted]
mod circuits {
    use arcis_imports::*;

    pub struct ShippingAddress {
        street: [u8; 64],
        city: [u8; 32],
        postal_code: [u8; 16],
        country: [u8; 32],
        phone: [u8; 20],
    }

    #[instruction]
    pub fn encrypt_shipping_address(
        input_ctxt: Enc<Shared, ShippingAddress>,
    ) -> Enc<Shared, ShippingAddress> {
        let input = input_ctxt.to_arcis();
        input_ctxt.owner.from_arcis(input)
    }

    pub struct ReputationInput {
        total_orders: u64,
        successful_orders: u64,
        disputes_won: u64,
        disputes_lost: u64,
    }

    #[instruction]
    pub fn calculate_reputation_score(
        input_ctxt: Enc<Shared, ReputationInput>,
    ) -> Enc<Shared, u64> {
        let input = input_ctxt.to_arcis();

        let score = if input.total_orders == 0 {
            500u64
        } else {
            let success_rate = (input.successful_orders * 100) / input.total_orders;
            let penalty = input.disputes_lost * 50;
            let bonus = input.disputes_won * 10;

            let raw_score = success_rate + bonus;
            let with_penalty = if raw_score >= penalty {
                raw_score - penalty
            } else {
                0u64
            };

            if with_penalty > 1000 {
                1000u64
            } else {
                with_penalty
            }
        };

        input_ctxt.owner.from_arcis(score)
    }

    pub struct VerifyAmountInput {
        amount: u64,
        min_amount: u64,
    }

    #[instruction]
    pub fn verify_escrow_amount(
        input_ctxt: Enc<Shared, VerifyAmountInput>,
    ) -> (Enc<Shared, u64>, bool) {
        let input = input_ctxt.to_arcis();
        let is_valid = input.amount >= input.min_amount;
        (input_ctxt.owner.from_arcis(input.amount), is_valid.reveal())
    }

    pub struct StakeCalculationInput {
        order_amount: u64,
        stake_percentage_bps: u64,
    }

    #[instruction]
    pub fn calculate_seller_stake(
        input_ctxt: Enc<Shared, StakeCalculationInput>,
    ) -> Enc<Shared, u64> {
        let input = input_ctxt.to_arcis();
        let stake = (input.order_amount * input.stake_percentage_bps) / 10000;
        input_ctxt.owner.from_arcis(stake)
    }

    pub struct FeeCalculationInput {
        order_amount: u64,
        fee_percentage_bps: u64,
    }

    #[instruction]
    pub fn calculate_platform_fee(
        input_ctxt: Enc<Shared, FeeCalculationInput>,
    ) -> Enc<Shared, u64> {
        let input = input_ctxt.to_arcis();
        let fee = (input.order_amount * input.fee_percentage_bps) / 10000;
        input_ctxt.owner.from_arcis(fee)
    }

    pub struct RefundCalculationInput {
        order_amount: u64,
        seller_stake: u64,
        include_penalty: bool,
    }

    #[instruction]
    pub fn calculate_refund_amount(
        input_ctxt: Enc<Shared, RefundCalculationInput>,
    ) -> Enc<Shared, u64> {
        let input = input_ctxt.to_arcis();
        let refund = if input.include_penalty {
            input.order_amount + input.seller_stake
        } else {
            input.order_amount
        };
        input_ctxt.owner.from_arcis(refund)
    }

    pub struct FundDistributionInput {
        order_amount: u64,
        seller_stake: u64,
        platform_fee: u64,
    }

    pub struct FundDistribution {
        to_seller: u64,
        to_platform: u64,
        to_buyer: u64,
    }

    #[instruction]
    pub fn calculate_completion_distribution(
        input_ctxt: Enc<Shared, FundDistributionInput>,
    ) -> Enc<Shared, FundDistribution> {
        let input = input_ctxt.to_arcis();

        let to_seller = input.order_amount - input.platform_fee + input.seller_stake;
        let to_platform = input.platform_fee;
        let to_buyer = 0u64;

        let distribution = FundDistribution {
            to_seller,
            to_platform,
            to_buyer,
        };

        input_ctxt.owner.from_arcis(distribution)
    }

    #[instruction]
    pub fn calculate_buyer_dispute_win(
        input_ctxt: Enc<Shared, FundDistributionInput>,
    ) -> Enc<Shared, FundDistribution> {
        let input = input_ctxt.to_arcis();

        let to_buyer = input.order_amount + input.seller_stake;
        let to_platform = input.platform_fee;
        let to_seller = 0u64;

        let distribution = FundDistribution {
            to_seller,
            to_platform,
            to_buyer,
        };

        input_ctxt.owner.from_arcis(distribution)
    }

    #[instruction]
    pub fn calculate_seller_dispute_win(
        input_ctxt: Enc<Shared, FundDistributionInput>,
    ) -> Enc<Shared, FundDistribution> {
        let input = input_ctxt.to_arcis();

        let to_seller = input.order_amount - input.platform_fee + input.seller_stake;
        let to_platform = input.platform_fee;
        let to_buyer = 0u64;

        let distribution = FundDistribution {
            to_seller,
            to_platform,
            to_buyer,
        };

        input_ctxt.owner.from_arcis(distribution)
    }

    pub struct BalanceCheckInput {
        current_balance: u64,
        required_amount: u64,
    }

    #[instruction]
    pub fn verify_sufficient_balance(
        input_ctxt: Enc<Shared, BalanceCheckInput>,
    ) -> (Enc<Shared, u64>, bool) {
        let input = input_ctxt.to_arcis();
        let has_funds = input.current_balance >= input.required_amount;
        (input_ctxt.owner.from_arcis(input.current_balance), has_funds.reveal())
    }

    pub struct TimeoutCheckInput {
        current_timestamp: u64,
        deadline_timestamp: u64,
    }

    #[instruction]
    pub fn check_timeout(
        input_ctxt: Enc<Shared, TimeoutCheckInput>,
    ) -> bool {
        let input = input_ctxt.to_arcis();
        let is_timeout = input.current_timestamp > input.deadline_timestamp;
        is_timeout.reveal()
    }
}
