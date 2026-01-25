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
- **Modern Web Interface**: React-based UI with real-time updates

## Setup & Run

**1. Start Arcium Localnet**

```bash
cd contracts/incognito
arcium localnet
```

Keep this running in a separate terminal.

**2. Setup Confidential Environment**

```bash
cd contracts/incognito
python3 ../../clients/cli/setup_confidential_env.py
```

**3. Start API Server**

From project root:

```bash
uvicorn services.api.app:app --host 0.0.0.0 --port 8001 --reload
```

**4. Start Web Interface** (Recommended)

From project root:

```bash
cd web-interface
npm install  # First time only
npm run dev
```

Web interface opens at `http://localhost:8080`

**5. Alternative: Streamlit Dashboard** (Legacy)

From project root:

```bash
streamlit run dashboard/app/dashboard.py
```

Dashboard opens at `http://localhost:8501`

## Requirements

- **Solana CLI**: For blockchain interactions
- **Python 3.8+**: For API server and CLI tools
- **Node.js 18+**: For web interface (recommended: use nvm with `.nvmrc`)
- **Arcium CLI**: For confidential computing features

