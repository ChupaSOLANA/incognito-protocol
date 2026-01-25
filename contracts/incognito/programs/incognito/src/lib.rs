use anchor_lang::prelude::*;
use anchor_lang::solana_program::sysvar::rent::Rent;
use anchor_lang::solana_program::hash::hashv;
use arcium_anchor::prelude::*;

const COMP_DEF_OFFSET_ADD_TOGETHER: u32 = comp_def_offset("add_together");
const COMP_DEF_OFFSET_DEPOSIT_SHIELDED: u32 = comp_def_offset("deposit_shielded");
const COMP_DEF_OFFSET_WITHDRAW_SHIELDED: u32 = comp_def_offset("withdraw_shielded");
const COMP_DEF_OFFSET_DEPOSIT_NOTE: u32 = comp_def_offset("deposit_note");
const COMP_DEF_OFFSET_WITHDRAW_NOTE_CHECK: u32 = comp_def_offset("withdraw_note_check");

pub const SOL_BALANCE_SEED: &[u8] = b"sol_balance";
pub const SOL_VAULT_SEED: &[u8] = b"sol_vault";
pub const POOL_STATE_SEED: &[u8] = b"pool_state";
pub const NULLIFIER_SEED: &[u8] = b"nf";
pub const MAX_MERKLE_DEPTH: usize = 32;
pub const COMMITMENT_SEED: &[u8] = b"commitment";

declare_id!("4N49EyRoX9p9zoiv1weeeqpaJTGbEHizbzZVgrsrVQeC");

#[account]
pub struct SolBalance {
    pub ciphertext: [u8; 32],
    pub nonce: [u8; 16],
    pub x25519_pubkey: [u8; 32],
    pub bump: u8,
}

#[account]
pub struct SolVault {
    pub total_deposited: u64,
    pub bump: u8,
}

fn vault_space() -> usize {
    8 + 8 + 1
}

#[account]
pub struct PoolState {
    pub root: [u8; 32],
    pub depth: u8,
    pub leaf_count: u64,
    pub bump: u8,
}

#[account]
pub struct NullifierMarker {}

#[account]
pub struct CommitmentMarker {}

fn ps_space() -> usize {
    8 + 32 + 1 + 8 + 1
}

#[inline]
fn h2(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    hashv(&[a, b]).to_bytes()
}

#[inline]
fn h1(x: &[u8; 32]) -> [u8; 32] {
    hashv(&[x]).to_bytes()
}

#[inline]
fn leaf_from(commitment: &[u8; 32], nf_hash: &[u8; 32]) -> [u8; 32] {
    h2(commitment, nf_hash)
}

fn zero_at(level: usize) -> [u8; 32] {
    let mut z = [0u8; 32];
    for _ in 0..level {
        z = h2(&z, &z);
    }
    z
}

fn verify_merkle_path(
    leaf: [u8; 32],
    path: &[[u8; 32]],
    root: [u8; 32],
    index: u64,
    depth: u8,
) -> bool {
    if path.len() != depth as usize {
        return false;
    }

    let mut current = leaf;
    let mut idx = index;

    for sibling in path.iter() {
        current = if (idx & 1) == 0 {
            h2(&current, sibling)
        } else {
            h2(sibling, &current)
        };
        idx >>= 1;
    }

    current == root
}

fn verify_and_insert_leaf(
    pool_state: &mut PoolState,
    leaf: [u8; 32],
    merkle_path: Vec<[u8; 32]>,
) -> Result<()> {
    let ps = pool_state;

    let cap = 1u128
        .checked_shl(ps.depth as u32)
        .ok_or(ErrorCode::InvalidDepth)?;
    require!((ps.leaf_count as u128) < cap, ErrorCode::TreeFull);

    require!(merkle_path.len() == ps.depth as usize, ErrorCode::InvalidPath);

    let index = ps.leaf_count;

    let mut current = [0u8; 32];
    let mut idx = index;

    for sibling in merkle_path.iter() {
        current = if (idx & 1) == 0 {
            h2(&current, sibling)
        } else {
            h2(sibling, &current)
        };
        idx >>= 1;
    }

    require!(current == ps.root, ErrorCode::InvalidMerklePath);

    current = leaf;
    idx = index;

    for sibling in merkle_path.iter() {
        current = if (idx & 1) == 0 {
            h2(&current, sibling)
        } else {
            h2(sibling, &current)
        };
        idx >>= 1;
    }

    ps.root = current;
    ps.leaf_count = ps
        .leaf_count
        .checked_add(1)
        .ok_or(ErrorCode::Overflow)?;

    Ok(())
}

fn verify_and_insert_commitment(
    pool_state: &mut PoolState,
    commitment: [u8; 32],
    merkle_path: Vec<[u8; 32]>,
) -> Result<()> {
    verify_and_insert_leaf(pool_state, commitment, merkle_path)
}

fn verify_and_insert_commitment_with_nf_hash(
    pool_state: &mut PoolState,
    commitment: [u8; 32],
    nf_hash: [u8; 32],
    merkle_path: Vec<[u8; 32]>,
) -> Result<()> {
    let leaf = leaf_from(&commitment, &nf_hash);
    verify_and_insert_leaf(pool_state, leaf, merkle_path)
}

#[init_computation_definition_accounts("add_together", payer)]
#[derive(Accounts)]
pub struct InitAddTogetherCompDef<'info> {
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

#[init_computation_definition_accounts("deposit_shielded", payer)]
#[derive(Accounts)]
pub struct InitDepositShieldedCompDef<'info> {
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

#[init_computation_definition_accounts("withdraw_shielded", payer)]
#[derive(Accounts)]
pub struct InitWithdrawShieldedCompDef<'info> {
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

#[init_computation_definition_accounts("deposit_note", payer)]
#[derive(Accounts)]
pub struct InitDepositNoteCompDef<'info> {
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

#[init_computation_definition_accounts("withdraw_note_check", payer)]
#[derive(Accounts)]
pub struct InitWithdrawNoteCheckCompDef<'info> {
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

#[queue_computation_accounts("add_together", payer)]
#[derive(Accounts)]
#[instruction(computation_offset: u64)]
pub struct AddTogether<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(
        init_if_needed,
        space = 9,
        payer = payer,
        seeds = [&SIGN_PDA_SEED],
        bump,
        address = derive_sign_pda!()
    )]
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
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_ADD_TOGETHER))]
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

#[queue_computation_accounts("deposit_shielded", payer)]
#[derive(Accounts)]
#[instruction(computation_offset: u64)]
pub struct DepositShielded<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(
        init_if_needed,
        space = 9,
        payer = payer,
        seeds = [&SIGN_PDA_SEED],
        bump,
        address = derive_sign_pda!()
    )]
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
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_DEPOSIT_SHIELDED))]
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

#[queue_computation_accounts("withdraw_shielded", payer)]
#[derive(Accounts)]
#[instruction(computation_offset: u64)]
pub struct WithdrawShielded<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(
        init_if_needed,
        space = 9,
        payer = payer,
        seeds = [&SIGN_PDA_SEED],
        bump,
        address = derive_sign_pda!()
    )]
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
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_WITHDRAW_SHIELDED))]
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

#[queue_computation_accounts("deposit_note", payer)]
#[derive(Accounts)]
#[instruction(computation_offset: u64)]
pub struct DepositNote<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(
        init_if_needed,
        space = 9,
        payer = payer,
        seeds = [&SIGN_PDA_SEED],
        bump,
        address = derive_sign_pda!()
    )]
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
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_DEPOSIT_NOTE))]
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

#[queue_computation_accounts("withdraw_note_check", payer)]
#[derive(Accounts)]
#[instruction(computation_offset: u64)]
pub struct WithdrawNoteCheck<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(
        init_if_needed,
        space = 9,
        payer = payer,
        seeds = [&SIGN_PDA_SEED],
        bump,
        address = derive_sign_pda!()
    )]
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
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_WITHDRAW_NOTE_CHECK))]
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

#[callback_accounts("add_together")]
#[derive(Accounts)]
pub struct AddTogetherCallback<'info> {
    pub arcium_program: Program<'info, Arcium>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_ADD_TOGETHER))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    /// CHECK: Instructions sysvar account
    #[account(address = ::anchor_lang::solana_program::sysvar::instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

#[callback_accounts("deposit_shielded")]
#[derive(Accounts)]
pub struct DepositShieldedCallback<'info> {
    pub arcium_program: Program<'info, Arcium>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_DEPOSIT_SHIELDED))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    /// CHECK: Instructions sysvar account
    #[account(address = ::anchor_lang::solana_program::sysvar::instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

#[callback_accounts("withdraw_shielded")]
#[derive(Accounts)]
pub struct WithdrawShieldedCallback<'info> {
    pub arcium_program: Program<'info, Arcium>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_WITHDRAW_SHIELDED))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    /// CHECK: Instructions sysvar account
    #[account(address = ::anchor_lang::solana_program::sysvar::instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

#[callback_accounts("deposit_note")]
#[derive(Accounts)]
pub struct DepositNoteCallback<'info> {
    pub arcium_program: Program<'info, Arcium>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_DEPOSIT_NOTE))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    /// CHECK: Instructions sysvar account
    #[account(address = ::anchor_lang::solana_program::sysvar::instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

#[callback_accounts("withdraw_note_check")]
#[derive(Accounts)]
pub struct WithdrawNoteCheckCallback<'info> {
    pub arcium_program: Program<'info, Arcium>,
    #[account(address = derive_comp_def_pda!(COMP_DEF_OFFSET_WITHDRAW_NOTE_CHECK))]
    pub comp_def_account: Account<'info, ComputationDefinitionAccount>,
    /// CHECK: Instructions sysvar account
    #[account(address = ::anchor_lang::solana_program::sysvar::instructions::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
}

#[derive(Accounts)]
pub struct InitVault<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(
        init,
        payer = payer,
        space = vault_space(),
        seeds = [SOL_VAULT_SEED],
        bump
    )]
    pub sol_vault: Account<'info, SolVault>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct InitPool<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(init, payer = payer, space = ps_space(), seeds = [POOL_STATE_SEED], bump)]
    pub pool_state: Account<'info, PoolState>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct ClosePool<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,
    #[account(
        mut,
        seeds = [POOL_STATE_SEED],
        bump = pool_state.bump,
        close = authority
    )]
    pub pool_state: Account<'info, PoolState>,
}

#[derive(Accounts)]
#[instruction(amount: u64, commitment: [u8; 32], nf_hash: [u8; 32], merkle_path: Vec<[u8; 32]>)]
pub struct DepositToPool<'info> {
    #[account(mut)]
    pub depositor: Signer<'info>,

    #[account(
        mut,
        seeds = [SOL_VAULT_SEED],
        bump = sol_vault.bump
    )]
    pub sol_vault: Account<'info, SolVault>,

    #[account(mut, seeds = [POOL_STATE_SEED], bump = pool_state.bump)]
    pub pool_state: Account<'info, PoolState>,

    #[account(
        init,
        payer = depositor,
        space = 8,
        seeds = [COMMITMENT_SEED, commitment.as_ref()],
        bump
    )]
    pub commitment_marker: Account<'info, CommitmentMarker>,

    /// CHECK: Wrapper stealth address for fee payment (one-time address, no validation needed)
    #[account(mut)]
    pub wrapper_stealth_address: UncheckedAccount<'info>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(commitment: [u8; 32])]
pub struct AddClaimNote<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    #[account(mut, seeds = [POOL_STATE_SEED], bump = pool_state.bump)]
    pub pool_state: Account<'info, PoolState>,

    #[account(
        init,
        payer = payer,
        space = 8,
        seeds = [COMMITMENT_SEED, commitment.as_ref()],
        bump
    )]
    pub commitment_marker: Account<'info, CommitmentMarker>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(
    amount: u64,
    commitment: [u8; 32],
    merkle_path: Vec<[u8; 32]>,
    nullifier: [u8; 32],
    index: u64,
    recipient_pubkey: [u8; 32],
    change_commitment: Option<[u8; 32]>,
    change_nf_hash: Option<[u8; 32]>,
    change_merkle_path: Option<Vec<[u8; 32]>>
)]
pub struct WithdrawFromPool<'info> {
    #[account(mut)]
    pub recipient: Signer<'info>,

    #[account(
        mut,
        seeds = [SOL_VAULT_SEED],
        bump = sol_vault.bump
    )]
    pub sol_vault: Account<'info, SolVault>,

    #[account(mut, seeds = [POOL_STATE_SEED], bump = pool_state.bump)]
    pub pool_state: Account<'info, PoolState>,

    #[account(
        init,
        payer = recipient,
        space = 8,
        seeds = [NULLIFIER_SEED, nullifier.as_ref()],
        bump
    )]
    pub nullifier_marker: Account<'info, NullifierMarker>,

    /// CHECK: When creating change, must match expected PDA. When not creating change, ignored.
    #[account(mut)]
    pub change_commitment_marker: UncheckedAccount<'info>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(nullifier: [u8; 32])]
pub struct MarkSpent<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(seeds = [POOL_STATE_SEED], bump = pool_state.bump)]
    pub pool_state: Account<'info, PoolState>,
    #[account(
        init,
        payer = payer,
        space = 8,
        seeds = [NULLIFIER_SEED, nullifier.as_ref()],
        bump
    )]
    pub nullifier_marker: Account<'info, NullifierMarker>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(commitment: [u8; 32], merkle_path: Vec<[u8; 32]>, nullifier: [u8; 32], index: u64)]
pub struct VerifyProof<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    #[account(seeds = [POOL_STATE_SEED], bump = pool_state.bump)]
    pub pool_state: Account<'info, PoolState>,

    #[account(
        init,
        payer = payer,
        space = 8,
        seeds = [NULLIFIER_SEED, nullifier.as_ref()],
        bump
    )]
    pub nullifier_marker: Account<'info, NullifierMarker>,

    pub system_program: Program<'info, System>,
}

#[arcium_program]
pub mod incognito {
    use super::*;

    pub fn init_vault(ctx: Context<InitVault>) -> Result<()> {
        let vault = &mut ctx.accounts.sol_vault;
        vault.total_deposited = 0;
        vault.bump = ctx.bumps.sol_vault;

        emit!(VaultInitialized {});
        Ok(())
    }

    pub fn init_add_together_comp_def(ctx: Context<InitAddTogetherCompDef>) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn add_together(
        ctx: Context<AddTogether>,
        computation_offset: u64,
        ciphertext_0: [u8; 32],
        ciphertext_1: [u8; 32],
        pub_key: [u8; 32],
        nonce: u128,
    ) -> Result<()> {
        ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;
        let args = vec![
            Argument::ArcisPubkey(pub_key),
            Argument::PlaintextU128(nonce),
            Argument::EncryptedU8(ciphertext_0),
            Argument::EncryptedU8(ciphertext_1),
        ];
        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            None,
            vec![AddTogetherCallback::callback_ix(&[])],
        )?;
        Ok(())
    }

    #[arcium_callback(encrypted_ix = "add_together")]
    pub fn add_together_callback(
        ctx: Context<AddTogetherCallback>,
        output: ComputationOutputs<AddTogetherOutput>,
    ) -> Result<()> {
        let o = match output {
            ComputationOutputs::Success(AddTogetherOutput { field_0 }) => field_0,
            _ => return Err(ErrorCode::AbortedComputation.into()),
        };
        emit!(SumEvent {
            sum: o.ciphertexts[0],
            nonce: o.nonce.to_le_bytes()
        });
        Ok(())
    }

    pub fn init_deposit_shielded_comp_def(
        ctx: Context<InitDepositShieldedCompDef>,
    ) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn deposit_shielded(
        ctx: Context<DepositShielded>,
        computation_offset: u64,
        ciphertext_balance: [u8; 32],
        ciphertext_amount: [u8; 32],
        pub_key: [u8; 32],
        nonce: u128,
    ) -> Result<()> {
        ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;
        let args = vec![
            Argument::ArcisPubkey(pub_key),
            Argument::PlaintextU128(nonce),
            Argument::EncryptedU64(ciphertext_balance),
            Argument::EncryptedU64(ciphertext_amount),
        ];
        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            None,
            vec![DepositShieldedCallback::callback_ix(&[])],
        )?;
        Ok(())
    }

    #[arcium_callback(encrypted_ix = "deposit_shielded")]
    pub fn deposit_shielded_callback(
        ctx: Context<DepositShieldedCallback>,
        output: ComputationOutputs<DepositShieldedOutput>,
    ) -> Result<()> {
        let o = match output {
            ComputationOutputs::Success(DepositShieldedOutput { field_0 }) => field_0,
            _ => return Err(ErrorCode::AbortedComputation.into()),
        };
        emit!(DepositShieldedEvent {
            new_balance: o.ciphertexts[0],
            nonce: o.nonce.to_le_bytes()
        });
        Ok(())
    }

    pub fn init_withdraw_shielded_comp_def(
        ctx: Context<InitWithdrawShieldedCompDef>,
    ) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn withdraw_shielded(
        ctx: Context<WithdrawShielded>,
        computation_offset: u64,
        ciphertext_balance: [u8; 32],
        ciphertext_amount: [u8; 32],
        pub_key: [u8; 32],
        nonce: u128,
    ) -> Result<()> {
        ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;
        let args = vec![
            Argument::ArcisPubkey(pub_key),
            Argument::PlaintextU128(nonce),
            Argument::EncryptedU64(ciphertext_balance),
            Argument::EncryptedU64(ciphertext_amount),
        ];
        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            None,
            vec![WithdrawShieldedCallback::callback_ix(&[])],
        )?;
        Ok(())
    }

    #[arcium_callback(encrypted_ix = "withdraw_shielded")]
    pub fn withdraw_shielded_callback(
        ctx: Context<WithdrawShieldedCallback>,
        output: ComputationOutputs<WithdrawShieldedOutput>,
    ) -> Result<()> {
        let group = match output {
            ComputationOutputs::Success(WithdrawShieldedOutput { field_0 }) => field_0,
            _ => return Err(ErrorCode::AbortedComputation.into()),
        };
        let enc = group.field_0;
        let success = group.field_1;
        emit!(WithdrawShieldedEvent {
            new_balance: enc.ciphertexts[0],
            nonce: enc.nonce.to_le_bytes(),
            success
        });
        Ok(())
    }

    pub fn init_pool(ctx: Context<InitPool>, depth: u8) -> Result<()> {
        require!((depth as usize) <= MAX_MERKLE_DEPTH, ErrorCode::InvalidDepth);

        let ps = &mut ctx.accounts.pool_state;

        ps.root = zero_at(depth as usize);
        ps.depth = depth;
        ps.leaf_count = 0;
        ps.bump = ctx.bumps.pool_state;

        emit!(PoolInitialized {
            depth,
            root: ps.root
        });
        Ok(())
    }

    pub fn close_pool(_ctx: Context<ClosePool>) -> Result<()> {
        emit!(PoolClosed {});
        Ok(())
    }

    pub fn deposit_to_pool(
        ctx: Context<DepositToPool>,
        amount: u64,
        commitment: [u8; 32],
        nf_hash: [u8; 32],
        merkle_path: Vec<[u8; 32]>,
    ) -> Result<()> {
        require!(amount > 0, ErrorCode::InvalidAmount);

        const WRAPPER_FEE: u64 = 50_000_000;
        require!(amount > WRAPPER_FEE, ErrorCode::InsufficientDepositAmount);

        let vault_amount = amount.checked_sub(WRAPPER_FEE).ok_or(ErrorCode::Overflow)?;

        anchor_lang::solana_program::program::invoke(
            &anchor_lang::solana_program::system_instruction::transfer(
                &ctx.accounts.depositor.key(),
                &ctx.accounts.wrapper_stealth_address.key(),
                WRAPPER_FEE,
            ),
            &[
                ctx.accounts.depositor.to_account_info(),
                ctx.accounts.wrapper_stealth_address.to_account_info(),
                ctx.accounts.system_program.to_account_info(),
            ],
        )?;

        let vault_ix = anchor_lang::solana_program::system_instruction::transfer(
            &ctx.accounts.depositor.key(),
            &ctx.accounts.sol_vault.key(),
            vault_amount,
        );
        anchor_lang::solana_program::program::invoke(
            &vault_ix,
            &[
                ctx.accounts.depositor.to_account_info(),
                ctx.accounts.sol_vault.to_account_info(),
            ],
        )?;

        ctx.accounts.sol_vault.total_deposited = ctx
            .accounts
            .sol_vault
            .total_deposited
            .checked_add(vault_amount)
            .ok_or(ErrorCode::Overflow)?;

        let index = ctx.accounts.pool_state.leaf_count;
        verify_and_insert_commitment_with_nf_hash(
            &mut ctx.accounts.pool_state,
            commitment,
            nf_hash,
            merkle_path,
        )?;

        emit!(RealDepositEvent {
            depositor: ctx.accounts.depositor.key(),
            amount,
            commitment,
            nf_hash,
            index,
            root: ctx.accounts.pool_state.root,
        });

        Ok(())
    }

    pub fn add_claim_note(
        ctx: Context<AddClaimNote>,
        commitment: [u8; 32],
        nf_hash: [u8; 32],
        merkle_path: Vec<[u8; 32]>,
    ) -> Result<()> {
        // Add note to Merkle tree without requiring SOL deposit
        // Used for change notes and seller payment notes in escrow

        let index = ctx.accounts.pool_state.leaf_count;
        verify_and_insert_commitment_with_nf_hash(
            &mut ctx.accounts.pool_state,
            commitment,
            nf_hash,
            merkle_path,
        )?;

        emit!(ClaimNoteAdded {
            commitment,
            nf_hash,
            index,
            root: ctx.accounts.pool_state.root,
        });

        Ok(())
    }

    pub fn withdraw_from_pool(
        ctx: Context<WithdrawFromPool>,
        amount: u64,
        commitment: [u8; 32],
        merkle_path: Vec<[u8; 32]>,
        nullifier: [u8; 32],
        index: u64,
        recipient_pubkey: [u8; 32],
        change_commitment: Option<[u8; 32]>,
        change_nf_hash: Option<[u8; 32]>,
        change_merkle_path: Option<Vec<[u8; 32]>>,
    ) -> Result<()> {
        require!(amount > 0, ErrorCode::InvalidAmount);

        let ps = &mut ctx.accounts.pool_state;

        let recipient_key_bytes = ctx.accounts.recipient.key().to_bytes();
        require!(
            recipient_key_bytes == recipient_pubkey,
            ErrorCode::UnauthorizedRecipient
        );

        let nf_hash = h1(&nullifier);
        let bound_leaf = leaf_from(&commitment, &nf_hash);

        require!(
            verify_merkle_path(bound_leaf, &merkle_path, ps.root, index, ps.depth),
            ErrorCode::InvalidMerkleProof
        );

        if let (Some(change_c), Some(change_nfh), Some(change_path)) =
            (change_commitment, change_nf_hash, change_merkle_path) {

            let (expected_pda, bump) = Pubkey::find_program_address(
                &[COMMITMENT_SEED, change_c.as_ref()],
                ctx.program_id,
            );

            require!(
                ctx.accounts.change_commitment_marker.key() == expected_pda,
                ErrorCode::InvalidChangeMarker
            );

            if ctx.accounts.change_commitment_marker.data_is_empty() {
                let rent = Rent::get()?;
                let space = 8;
                let lamports_needed = rent.minimum_balance(space);

                anchor_lang::solana_program::program::invoke_signed(
                    &anchor_lang::solana_program::system_instruction::create_account(
                        &ctx.accounts.recipient.key(),
                        &expected_pda,
                        lamports_needed,
                        space as u64,
                        ctx.program_id,
                    ),
                    &[
                        ctx.accounts.recipient.to_account_info(),
                        ctx.accounts.change_commitment_marker.to_account_info(),
                    ],
                    &[&[COMMITMENT_SEED, change_c.as_ref(), &[bump]]],
                )?;
            }

            let change_index = ps.leaf_count;
            verify_and_insert_commitment_with_nf_hash(
                ps,
                change_c,
                change_nfh,
                change_path,
            )?;

            emit!(ChangeNoteCreated {
                original_nullifier: nullifier,
                change_commitment: change_c,
                change_index,
            });
        }

        let vault_balance = ctx.accounts.sol_vault.to_account_info().lamports();
        require!(vault_balance >= amount, ErrorCode::InsufficientVaultBalance);

        **ctx
            .accounts
            .sol_vault
            .to_account_info()
            .try_borrow_mut_lamports()? -= amount;
        **ctx
            .accounts
            .recipient
            .to_account_info()
            .try_borrow_mut_lamports()? += amount;

        ctx.accounts.sol_vault.total_deposited = ctx
            .accounts
            .sol_vault
            .total_deposited
            .checked_sub(amount)
            .ok_or(ErrorCode::Overflow)?;

        emit!(RealWithdrawEvent {
            recipient: ctx.accounts.recipient.key(),
            amount,
            nullifier,
            index,
        });

        Ok(())
    }

    pub fn mark_spent(ctx: Context<MarkSpent>, nullifier: [u8; 32]) -> Result<()> {
        let _nm = &ctx.accounts.nullifier_marker;
        let ps = &ctx.accounts.pool_state;

        emit!(NoteSpent {
            nullifier,
            root: ps.root
        });
        Ok(())
    }

    pub fn verify_proof(
        ctx: Context<VerifyProof>,
        commitment: [u8; 32],
        merkle_path: Vec<[u8; 32]>,
        nullifier: [u8; 32],
        index: u64,
    ) -> Result<()> {
        let ps = &ctx.accounts.pool_state;

        let nf_hash = h1(&nullifier);
        let bound_leaf = leaf_from(&commitment, &nf_hash);

        require!(
            verify_merkle_path(bound_leaf, &merkle_path, ps.root, index, ps.depth),
            ErrorCode::InvalidMerkleProof
        );

        emit!(ProofVerified {
            commitment,
            nullifier,
            index
        });
        Ok(())
    }

    pub fn init_deposit_note_comp_def(ctx: Context<InitDepositNoteCompDef>) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn deposit_note(
        ctx: Context<DepositNote>,
        computation_offset: u64,
        ciphertext_amount: [u8; 32],
        pub_key: [u8; 32],
        nonce: u128,
    ) -> Result<()> {
        ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;
        let args = vec![
            Argument::ArcisPubkey(pub_key),
            Argument::PlaintextU128(nonce),
            Argument::EncryptedU64(ciphertext_amount),
        ];
        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            None,
            vec![DepositNoteCallback::callback_ix(&[])],
        )?;
        Ok(())
    }

    #[arcium_callback(encrypted_ix = "deposit_note")]
    pub fn deposit_note_callback(
        ctx: Context<DepositNoteCallback>,
        output: ComputationOutputs<DepositNoteOutput>,
    ) -> Result<()> {
        let o = match output {
            ComputationOutputs::Success(DepositNoteOutput { field_0 }) => field_0,
            _ => return Err(ErrorCode::AbortedComputation.into()),
        };
        emit!(DepositNoteEvent {
            ct_amount: o.ciphertexts[0],
            nonce: o.nonce.to_le_bytes()
        });
        Ok(())
    }

    pub fn init_withdraw_note_check_comp_def(
        ctx: Context<InitWithdrawNoteCheckCompDef>,
    ) -> Result<()> {
        init_comp_def(ctx.accounts, true, 0, None, None)?;
        Ok(())
    }

    pub fn withdraw_note_check(
        ctx: Context<WithdrawNoteCheck>,
        computation_offset: u64,
        ciphertext_note_amount: [u8; 32],
        ciphertext_want: [u8; 32],
        pub_key: [u8; 32],
        nonce: u128,
    ) -> Result<()> {
        ctx.accounts.sign_pda_account.bump = ctx.bumps.sign_pda_account;
        let args = vec![
            Argument::ArcisPubkey(pub_key),
            Argument::PlaintextU128(nonce),
            Argument::EncryptedU64(ciphertext_note_amount),
            Argument::EncryptedU64(ciphertext_want),
        ];
        queue_computation(
            ctx.accounts,
            computation_offset,
            args,
            None,
            vec![WithdrawNoteCheckCallback::callback_ix(&[])],
        )?;
        Ok(())
    }

    #[arcium_callback(encrypted_ix = "withdraw_note_check")]
    pub fn withdraw_note_check_callback(
        ctx: Context<WithdrawNoteCheckCallback>,
        output: ComputationOutputs<WithdrawNoteCheckOutput>,
    ) -> Result<()> {
        let group = match output { 
            ComputationOutputs::Success(WithdrawNoteCheckOutput { field_0 }) => field_0,
            _ => return Err(ErrorCode::AbortedComputation.into()),
        };
        let enc = group.field_0;
        let ok = group.field_1;
        emit!(WithdrawNoteCheckEvent {
            note_amount_ct: enc.ciphertexts[0],
            nonce: enc.nonce.to_le_bytes(),
            ok
        });
        Ok(())
    }
}

#[event]
pub struct VaultInitialized {}

#[event]
pub struct RealDepositEvent {
    pub depositor: Pubkey,
    pub amount: u64,
    pub commitment: [u8; 32],
    pub nf_hash: [u8; 32],
    pub index: u64,
    pub root: [u8; 32],
}

#[event]
pub struct RealWithdrawEvent {
    pub recipient: Pubkey,
    pub amount: u64,
    pub nullifier: [u8; 32],
    pub index: u64,
}

#[event]
pub struct SumEvent {
    pub sum: [u8; 32],
    pub nonce: [u8; 16],
}

#[event]
pub struct DepositShieldedEvent {
    pub new_balance: [u8; 32],
    pub nonce: [u8; 16],
}

#[event]
pub struct WithdrawShieldedEvent {
    pub new_balance: [u8; 32],
    pub nonce: [u8; 16],
    pub success: bool,
}

#[event]
pub struct PoolInitialized {
    pub depth: u8,
    pub root: [u8; 32],
}

#[event]
pub struct PoolClosed {}

#[event]
pub struct NoteSpent {
    pub nullifier: [u8; 32],
    pub root: [u8; 32],
}

#[event]
pub struct ProofVerified {
    pub commitment: [u8; 32],
    pub nullifier: [u8; 32],
    pub index: u64,
}

#[event]
pub struct DepositNoteEvent {
    pub ct_amount: [u8; 32],
    pub nonce: [u8; 16],
}

#[event]
pub struct WithdrawNoteCheckEvent {
    pub note_amount_ct: [u8; 32],
    pub nonce: [u8; 16],
    pub ok: bool,
}

#[event]
pub struct ChangeNoteCreated {
    pub original_nullifier: [u8; 32],
    pub change_commitment: [u8; 32],
    pub change_index: u64,
}

#[event]
pub struct ClaimNoteAdded {
    pub commitment: [u8; 32],
    pub nf_hash: [u8; 32],
    pub index: u64,
    pub root: [u8; 32],
}

#[error_code]
pub enum ErrorCode {
    #[msg("The computation was aborted")]
    AbortedComputation,
    #[msg("Cluster not set")]
    ClusterNotSet,
    #[msg("Invalid depth")]
    InvalidDepth,
    #[msg("Merkle tree is full")]
    TreeFull,
    #[msg("Overflow")]
    Overflow,
    #[msg("Invalid amount")]
    InvalidAmount,
    #[msg("Invalid Merkle path length")]
    InvalidPath,
    #[msg("Invalid Merkle proof")]
    InvalidMerkleProof,
    #[msg("Nullifier already used (double-spend attempt)")]
    NullifierAlreadyUsed,
    #[msg("Insufficient vault balance")]
    InsufficientVaultBalance,
    #[msg("Invalid Merkle path - does not reconstruct current root")]
    InvalidMerklePath,
    #[msg("Unauthorized recipient - recipient does not match commitment")]
    UnauthorizedRecipient,
    #[msg("Commitment already exists")]
    CommitmentAlreadyExists,
    #[msg("Change commitment marker account required when creating change note")]
    ChangeMarkerRequired,
    #[msg("Invalid change commitment marker PDA")]
    InvalidChangeMarker,
    #[msg("Deposit amount must be greater than wrapper fee (0.05 SOL)")]
    InsufficientDepositAmount,
}
