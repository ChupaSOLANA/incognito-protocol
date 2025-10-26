use arcis_imports::*;

#[encrypted]
mod circuits {
    use arcis_imports::*;

    pub struct InputValues {
        v1: u8,
        v2: u8,
    }

    #[instruction]
    pub fn add_together(input_ctxt: Enc<Shared, InputValues>) -> Enc<Shared, u16> {
        let input = input_ctxt.to_arcis();
        let sum = input.v1 as u16 + input.v2 as u16;
        input_ctxt.owner.from_arcis(sum)
    }

    pub struct BalanceInput {
        balance: u64,
        amount: u64,
    }

    #[instruction]
    pub fn deposit_shared(input_ctxt: Enc<Shared, BalanceInput>) -> Enc<Shared, u64> {
        let input = input_ctxt.to_arcis();
        let new_balance = input.balance + input.amount;
        input_ctxt.owner.from_arcis(new_balance)
    }

    #[instruction]
    pub fn withdraw_shared(input_ctxt: Enc<Shared, BalanceInput>) -> (Enc<Shared, u64>, bool) {
        let input = input_ctxt.to_arcis();
        let can = input.balance >= input.amount;
        let new_balance = if can { input.balance - input.amount } else { input.balance };
        (input_ctxt.owner.from_arcis(new_balance), can.reveal())
    }

    #[instruction]
    pub fn deposit_shielded(input_ctxt: Enc<Shared, BalanceInput>) -> Enc<Shared, u64> {
        let input = input_ctxt.to_arcis();
        let new_balance = input.balance + input.amount;
        input_ctxt.owner.from_arcis(new_balance)
    }

    #[instruction]
    pub fn withdraw_shielded(input_ctxt: Enc<Shared, BalanceInput>) -> (Enc<Shared, u64>, bool) {
        let input = input_ctxt.to_arcis();
        let can = input.balance >= input.amount;
        let new_balance = if can { input.balance - input.amount } else { input.balance };
        (input_ctxt.owner.from_arcis(new_balance), can.reveal())
    }

    pub struct DepositNoteInput {
        amount: u64,
    }

    #[instruction]
    pub fn deposit_note(input_ctxt: Enc<Shared, DepositNoteInput>) -> Enc<Shared, u64> {
        let input = input_ctxt.to_arcis();
        input_ctxt.owner.from_arcis(input.amount)
    }

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
