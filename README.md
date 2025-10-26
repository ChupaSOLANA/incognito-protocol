# Incognito Protocol

A privacy-focused decentralized marketplace built on Solana with confidential transfers, privacy pool notes, encrypted messaging, and stealth addresses.

## What Is This?

Incognito Protocol is a privacy marketplace where users can buy and sell goods while keeping transaction amounts, balances, and identities private. It combines Solana's Token-2022 confidential transfers with off-chain privacy techniques.

## Features

- **Confidential Transfers**: Encrypted token balances and amounts
- **Privacy Pool Notes**: Unlinkable deposits and withdrawals using Merkle trees
- **Encrypted Shipping**: End-to-end encrypted communications
- **Dual Payment System**: Pay with confidential tokens or privacy notes
- **Escrow with Dispute Resolution**: Automated escrow with reputation tracking
- **Flexible Finalization**: Release funds immediately or wait 7 days

## Quick Start

### Prerequisites

Install the following:

- Rust 1.70+
- Solana CLI 1.18+
- Anchor 0.31.1
- Node.js 16+
- Python 3.9+

```bash
pip install fastapi uvicorn pydantic solders solana pynacl cryptography streamlit
```

### Setup & Run

**1. Start Arcium Localnet**

```bash
cd contracts/incognito
arcium localnet
```

Keep this running in a separate terminal.

**2. Initialize Privacy Pool**

```bash
cd contracts/incognito
python3 ../../clients/cli/setup_confidential_env.py
```

**3. Start API Server**

From project root:

```bash
uvicorn services.api.app:app --host 0.0.0.0 --port 8001 --reload
```

API available at `http://localhost:8001`

**4. Start Dashboard**

From project root:

```bash
streamlit run dashboard/app/dashboard.py
```

Dashboard opens at `http://localhost:8501`

## Using the Marketplace

### Deposit Funds

**Option 1: Confidential SOL (cSOL)**

```bash
spl-token deposit-confidential-tokens <MINT> <AMOUNT> --owner <KEYPAIR>
spl-token apply-pending-balance <MINT> --owner <KEYPAIR>
```

**Option 2: Privacy Pool Notes**

```bash
python3 -m clients.cli.incognito_marketplace deposit --amount 10 --user keys/userA.json
```

### Buy/Sell Flow

1. Seller creates listing via Dashboard
2. Buyer purchases item (funds locked in escrow)
3. Seller ships and provides encrypted tracking info
4. Buyer confirms delivery
5. Buyer releases funds immediately OR waits 7 days for auto-release

## Project Structure

```
incognito-protocol-1/
├── contracts/
│   ├── incognito/          # Privacy pool contract
│   └── escrow/             # Escrow contract
├── services/
│   ├── api/                # FastAPI backend
│   └── crypto_core/        # Privacy primitives
├── dashboard/              # Streamlit UI
├── clients/                # CLI tools
└── data/                   # Runtime storage
```

## API Endpoints

**Marketplace:**
- `POST /listing` - Create listing
- `GET /listings` - Browse listings
- `POST /marketplace/buy` - Purchase item

**Escrow:**
- `POST /escrow/create` - Create order
- `POST /escrow/accept` - Seller accepts
- `POST /escrow/ship` - Seller ships
- `POST /escrow/confirm_delivery` - Buyer confirms
- `POST /escrow/buyer_release_early` - Immediate release
- `POST /escrow/finalize` - Auto-finalize after 7 days

Full docs: `http://localhost:8001/docs`

## Privacy Guarantees

- Transaction amounts: Encrypted using ElGamal
- Account balances: Confidential and unqueryable
- Participant identity: Obscured via stealth addresses
- Shipping data: End-to-end encrypted

## Security Notice

This is a development prototype. Before production:

- Audit smart contracts
- Implement rate limiting
- Use secure key management
- Deploy to devnet/mainnet
- Add monitoring systems

## License

MIT License
