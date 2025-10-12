use anchor_lang::prelude::*;
use anchor_lang::solana_program::hash::{hashv, Hash};

declare_id!("DyCNNyUfcYL8X3PNCDcSDXiYtx4ycheG7i13mckjB32j"); // même clé que dans Anchor.toml

#[program]
pub mod merkle_registry {
    use super::*;

    pub fn init_tree(ctx: Context<InitTree>, seed: [u8;32], root: [u8;32]) -> Result<()> {
        let tree = &mut ctx.accounts.tree;
        tree.authority = ctx.accounts.authority.key();
        tree.root = root;
        tree.seed = seed;
        tree.bump = ctx.bumps.tree; // 0.31: accès direct au champ
        Ok(())
    }

    pub fn set_authority(ctx: Context<SetAuthority>, new_authority: Pubkey) -> Result<()> {
        require!(ctx.accounts.authority.key() == ctx.accounts.tree.authority, MerkleError::Unauthorized);
        ctx.accounts.tree.authority = new_authority;
        Ok(())
    }

    pub fn update_root(ctx: Context<UpdateRoot>, new_root: [u8;32]) -> Result<()> {
        require!(ctx.accounts.authority.key() == ctx.accounts.tree.authority, MerkleError::Unauthorized);
        ctx.accounts.tree.root = new_root;
        emit!(RootUpdated { tree: ctx.accounts.tree.key(), root: new_root });
        Ok(())
    }

    pub fn verify_proof(ctx: Context<VerifyProof>, leaf: [u8;32], proof: Vec<[u8;32]>, index: u64) -> Result<()> {
        let mut h = leaf;
        let mut idx = index;
        for sib in proof.iter() {
            let (left, right) = if idx & 1 == 0 { (h, *sib) } else { (*sib, h) };
            h = hash_pair(left, right);
            idx >>= 1;
        }
        require!(h == ctx.accounts.tree.root, MerkleError::InvalidProof);
        Ok(())
    }
}

#[account]
pub struct MerkleTree {
    pub authority: Pubkey,
    pub root: [u8;32],
    pub seed: [u8;32],
    pub bump: u8,
}
impl MerkleTree { pub const LEN: usize = 8 + 32 + 32 + 32 + 1; }

#[derive(Accounts)]
#[instruction(seed: [u8;32])]
pub struct InitTree<'info> {
    #[account(init, payer = authority, space = MerkleTree::LEN, seeds = [b"merkle", seed.as_ref()], bump)]
    pub tree: Account<'info, MerkleTree>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct UpdateRoot<'info> {
    #[account(mut, seeds = [b"merkle", tree.seed.as_ref()], bump = tree.bump)]
    pub tree: Account<'info, MerkleTree>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct SetAuthority<'info> {
    #[account(mut, seeds = [b"merkle", tree.seed.as_ref()], bump = tree.bump)]
    pub tree: Account<'info, MerkleTree>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct VerifyProof<'info> {
    /// CHECK: read-only
    pub authority: UncheckedAccount<'info>,
    #[account(seeds = [b"merkle", tree.seed.as_ref()], bump = tree.bump)]
    pub tree: Account<'info, MerkleTree>,
}

#[event]
pub struct RootUpdated { pub tree: Pubkey, pub root: [u8;32], }

#[error_code]
pub enum MerkleError {
    #[msg("unauthorized")]
    Unauthorized,
    #[msg("invalid merkle proof")]
    InvalidProof,
}

fn hash_pair(left: [u8;32], right: [u8;32]) -> [u8;32] {
    let h: Hash = hashv(&[&left, &right]);
    h.to_bytes()
}
