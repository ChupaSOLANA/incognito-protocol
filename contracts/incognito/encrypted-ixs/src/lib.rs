// encrypted-ixs/src/lib.rs
use arcis_imports::*;

/// All MPC/Arcium-encrypted instructions live inside this module.
/// Labels MUST match on-chain & TS exactly:
/// add_together, deposit_shared, withdraw_shared, deposit_shielded, withdraw_shielded,
/// deposit_note, withdraw_note_check.
#[encrypted]
mod circuits {
    use arcis_imports::*;

    // ------------------- add_together -------------------

    /// Two small integers to add privately.
    pub struct InputValues {
        v1: u8,
        v2: u8,
    }

    /// Returns Enc<Shared, u16> where value = v1 + v2.
    #[instruction]
    pub fn add_together(input_ctxt: Enc<Shared, InputValues>) -> Enc<Shared, u16> {
        let input = input_ctxt.to_arcis();
        let sum = input.v1 as u16 + input.v2 as u16;
        input_ctxt.owner.from_arcis(sum)
    }

    // ---------------- shared balance primitives ---------

    /// Balance transition input (shared for demo/MVP).
    pub struct BalanceInput {
        balance: u64,
        amount: u64,
    }

    /// deposit_shared: new_balance = balance + amount
    #[instruction]
    pub fn deposit_shared(input_ctxt: Enc<Shared, BalanceInput>) -> Enc<Shared, u64> {
        let input = input_ctxt.to_arcis();
        let new_balance = input.balance + input.amount;
        input_ctxt.owner.from_arcis(new_balance)
    }

    /// withdraw_shared: (new_balance, success = balance >= amount)
    #[instruction]
    pub fn withdraw_shared(input_ctxt: Enc<Shared, BalanceInput>) -> (Enc<Shared, u64>, bool) {
        let input = input_ctxt.to_arcis();
        let can = input.balance >= input.amount;
        let new_balance = if can { input.balance - input.amount } else { input.balance };
        (input_ctxt.owner.from_arcis(new_balance), can.reveal())
    }

    // ---------------- shielded balance primitives -------

    /// deposit_shielded: same logic as shared (pure-MPC check), kept distinct for routing.
    #[instruction]
    pub fn deposit_shielded(input_ctxt: Enc<Shared, BalanceInput>) -> Enc<Shared, u64> {
        let input = input_ctxt.to_arcis();
        let new_balance = input.balance + input.amount;
        input_ctxt.owner.from_arcis(new_balance)
    }

    /// withdraw_shielded: (new_balance, success = balance >= amount)
    #[instruction]
    pub fn withdraw_shielded(input_ctxt: Enc<Shared, BalanceInput>) -> (Enc<Shared, u64>, bool) {
        let input = input_ctxt.to_arcis();
        let can = input.balance >= input.amount;
        let new_balance = if can { input.balance - input.amount } else { input.balance };
        (input_ctxt.owner.from_arcis(new_balance), can.reveal())
    }

    // ---------------- note circuits (Step 4) ------------

    /// deposit_note(input: Enc<Shared, { amount:u64 }>) -> Enc<Shared,u64>
    /// Returns ct_amount with Arcis-managed nonce (carried in callback output).
    pub struct DepositNoteInput {
        amount: u64,
    }

    #[instruction]
    pub fn deposit_note(input_ctxt: Enc<Shared, DepositNoteInput>) -> Enc<Shared, u64> {
        let input = input_ctxt.to_arcis();
        input_ctxt.owner.from_arcis(input.amount)
    }

    /// withdraw_note_check(input: Enc<Shared, { note_amount:u64, want:u64 }>)
    /// -> (Enc<Shared,u64>, bool)
    /// Returns the note_amount re-encrypted (never revealed) and ok flag = (note_amount >= want).
    pub struct WithdrawNoteCheckInput {
        note_amount: u64,
        want: u64,
    }

    #[instruction]
    pub fn withdraw_note_check(
        input_ctxt: Enc<Shared, WithdrawNoteCheckInput>,
    ) -> (Enc<Shared, u64>, bool) {
        let input = input_ctxt.to_arcis();
        let ok = input.note_amount >= input.want;
        (input_ctxt.owner.from_arcis(input.note_amount), ok.reveal())
    }
}
