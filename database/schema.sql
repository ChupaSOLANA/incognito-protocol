-- PostgreSQL schema for Incognito Protocol marketplace
-- With field-level encryption for sensitive data

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- USER NOTES TABLE
-- Stores privacy pool notes with CLIENT-SIDE encrypted data (END-TO-END)
-- ============================================================================
--
-- SECURITY MODEL:
-- - encrypted_blob is encrypted CLIENT-SIDE with user's Solana keypair (NaCl)
-- - API server is BLIND - cannot decrypt the blob
-- - Only the user with their private key can decrypt
-- - Database is just a fast, encrypted cache (zero trust required)
-- ============================================================================

CREATE TABLE user_notes (
    id BIGSERIAL PRIMARY KEY,

    -- Privacy: Store hash of pubkey, not plaintext
    owner_pubkey_hash VARCHAR(64) NOT NULL,

    -- Public data (on-chain Merkle tree)
    commitment VARCHAR(64) UNIQUE NOT NULL,

    -- Client-side encrypted blob (API CANNOT DECRYPT!)
    -- Format: {"ciphertext": "base64...", "nonce": "base64..."}
    -- Contains (encrypted): secret, nullifier, amount_sol, leaf_index
    -- Encrypted with user's Solana keypair using NaCl public key encryption
    encrypted_blob JSONB NOT NULL,

    -- State
    spent BOOLEAN DEFAULT FALSE,

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    spent_at TIMESTAMP,

    -- Indexes for performance
    CONSTRAINT valid_spent_at CHECK (
        (spent = FALSE AND spent_at IS NULL) OR
        (spent = TRUE AND spent_at IS NOT NULL)
    )
);

-- Indexes
CREATE INDEX idx_notes_owner ON user_notes(owner_pubkey_hash);
CREATE INDEX idx_notes_commitment ON user_notes(commitment);
CREATE INDEX idx_notes_unspent ON user_notes(owner_pubkey_hash, spent) WHERE spent = FALSE;
CREATE INDEX idx_notes_created ON user_notes(created_at DESC);

-- ============================================================================
-- LISTINGS TABLE
-- Marketplace product listings
-- ============================================================================

CREATE TABLE listings (
    id BIGSERIAL PRIMARY KEY,
    listing_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),

    -- Seller info (hashed for privacy)
    seller_pubkey_hash VARCHAR(64) NOT NULL,

    -- Product details
    title VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),

    -- Pricing
    price_lamports BIGINT NOT NULL CHECK (price_lamports > 0),

    -- IPFS content
    ipfs_cid VARCHAR(100),  -- Product images/files

    -- Status
    active BOOLEAN DEFAULT TRUE,

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Full-text search
    tsv_search TSVECTOR
);

-- Indexes
CREATE INDEX idx_listings_seller ON listings(seller_pubkey_hash);
CREATE INDEX idx_listings_active ON listings(active) WHERE active = TRUE;
CREATE INDEX idx_listings_category ON listings(category) WHERE active = TRUE;
CREATE INDEX idx_listings_price ON listings(price_lamports) WHERE active = TRUE;
CREATE INDEX idx_listings_search ON listings USING GIN(tsv_search);

-- Trigger for full-text search
CREATE OR REPLACE FUNCTION listings_search_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv_search :=
        setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.description, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.category, '')), 'C');
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER tsvector_update BEFORE INSERT OR UPDATE
ON listings FOR EACH ROW EXECUTE FUNCTION listings_search_trigger();

-- ============================================================================
-- ESCROWS TABLE
-- Escrow state machine for secure transactions
-- ============================================================================

CREATE TYPE escrow_state AS ENUM (
    'created',
    'accepted',
    'shipped',
    'delivered',
    'completed',
    'disputed',
    'refunded'
);

CREATE TABLE escrows (
    id BIGSERIAL PRIMARY KEY,
    escrow_pubkey VARCHAR(44) UNIQUE NOT NULL,  -- On-chain escrow account

    -- Parties (hashed for privacy)
    buyer_pubkey_hash VARCHAR(64) NOT NULL,
    seller_pubkey_hash VARCHAR(64) NOT NULL,

    -- Linked listing
    listing_id UUID REFERENCES listings(listing_id),

    -- Amount
    amount_lamports BIGINT NOT NULL CHECK (amount_lamports > 0),

    -- Payment tracking
    payment_method VARCHAR(20) DEFAULT 'note',  -- Always 'note' now
    note_commitment VARCHAR(64),  -- Which note was spent

    -- State machine
    state escrow_state NOT NULL DEFAULT 'created',

    -- Encrypted shipping address (only buyer/seller can decrypt)
    -- Encrypted with XChaCha20-Poly1305 (from original implementation)
    shipping_address_encrypted JSONB,

    -- Arbiter for disputes
    arbiter_pubkey_hash VARCHAR(64),

    -- Timestamps for state transitions
    created_at TIMESTAMP DEFAULT NOW(),
    accepted_at TIMESTAMP,
    shipped_at TIMESTAMP,
    delivered_at TIMESTAMP,
    completed_at TIMESTAMP,
    disputed_at TIMESTAMP,
    refunded_at TIMESTAMP,

    -- Metadata
    dispute_reason TEXT,
    resolution_notes TEXT,

    -- State transition constraints
    CONSTRAINT valid_state_transitions CHECK (
        (state = 'created' AND accepted_at IS NULL) OR
        (state = 'accepted' AND accepted_at IS NOT NULL) OR
        (state = 'shipped' AND shipped_at IS NOT NULL) OR
        (state = 'delivered' AND delivered_at IS NOT NULL) OR
        (state = 'completed' AND completed_at IS NOT NULL) OR
        (state = 'disputed' AND disputed_at IS NOT NULL) OR
        (state = 'refunded' AND refunded_at IS NOT NULL)
    )
);

-- Indexes
CREATE INDEX idx_escrows_buyer ON escrows(buyer_pubkey_hash);
CREATE INDEX idx_escrows_seller ON escrows(seller_pubkey_hash);
CREATE INDEX idx_escrows_state ON escrows(state);
CREATE INDEX idx_escrows_listing ON escrows(listing_id);
CREATE INDEX idx_escrows_created ON escrows(created_at DESC);

-- ============================================================================
-- MESSAGES TABLE
-- Encrypted buyer-seller communications
-- ============================================================================

CREATE TABLE messages (
    id BIGSERIAL PRIMARY KEY,
    message_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),

    -- Linked escrow
    escrow_pubkey VARCHAR(44) REFERENCES escrows(escrow_pubkey),

    -- Sender/recipient (hashed)
    sender_pubkey_hash VARCHAR(64) NOT NULL,
    recipient_pubkey_hash VARCHAR(64) NOT NULL,

    -- Encrypted message content (XChaCha20-Poly1305)
    content_encrypted JSONB NOT NULL,

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    read_at TIMESTAMP
);

-- Indexes
CREATE INDEX idx_messages_escrow ON messages(escrow_pubkey);
CREATE INDEX idx_messages_recipient ON messages(recipient_pubkey_hash, read_at);
CREATE INDEX idx_messages_created ON messages(created_at DESC);

-- ============================================================================
-- NULLIFIER_REGISTRY TABLE
-- Prevent double-spending of privacy notes
-- ============================================================================

CREATE TABLE nullifier_registry (
    id BIGSERIAL PRIMARY KEY,

    -- Hash of nullifier (not plaintext for privacy)
    nullifier_hash VARCHAR(64) UNIQUE NOT NULL,

    -- Which note was spent
    commitment VARCHAR(64) REFERENCES user_notes(commitment),

    -- When it was spent
    spent_at TIMESTAMP DEFAULT NOW(),

    -- Transaction info
    tx_signature VARCHAR(88)
);

-- Index for fast double-spend checks
CREATE INDEX idx_nullifiers_hash ON nullifier_registry(nullifier_hash);

-- ============================================================================
-- MERKLE_TREES TABLE
-- Track on-chain Merkle tree state
-- ============================================================================

CREATE TABLE merkle_trees (
    id BIGSERIAL PRIMARY KEY,

    -- On-chain tree account
    tree_pubkey VARCHAR(44) UNIQUE NOT NULL,

    -- Current state
    root VARCHAR(64) NOT NULL,
    next_index INTEGER NOT NULL DEFAULT 0,

    -- Metadata
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- STEALTH_ADDRESSES TABLE
-- One-time use addresses for privacy
-- ============================================================================

CREATE TABLE stealth_addresses (
    id BIGSERIAL PRIMARY KEY,

    -- Stealth address
    stealth_pubkey VARCHAR(44) UNIQUE NOT NULL,

    -- Owner (hashed)
    owner_pubkey_hash VARCHAR(64) NOT NULL,

    -- Encrypted private key (only owner can decrypt)
    private_key_encrypted JSONB NOT NULL,

    -- Usage tracking
    used BOOLEAN DEFAULT FALSE,
    used_at TIMESTAMP,

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_stealth_owner ON stealth_addresses(owner_pubkey_hash);
CREATE INDEX idx_stealth_unused ON stealth_addresses(owner_pubkey_hash, used) WHERE used = FALSE;

-- ============================================================================
-- AUDIT_LOG TABLE
-- Security and compliance tracking
-- ============================================================================

CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,

    -- What happened
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB,

    -- Who did it (hashed)
    actor_pubkey_hash VARCHAR(64),

    -- IP address (for security monitoring)
    ip_address INET,

    -- Timestamp
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for searching recent events
CREATE INDEX idx_audit_created ON audit_log(created_at DESC);
CREATE INDEX idx_audit_actor ON audit_log(actor_pubkey_hash, created_at DESC);
CREATE INDEX idx_audit_type ON audit_log(event_type, created_at DESC);

-- ============================================================================
-- VIEWS
-- Convenient read-only queries
-- ============================================================================

-- Active listings with seller info
CREATE VIEW active_listings AS
SELECT
    l.listing_id,
    l.seller_pubkey_hash,
    l.title,
    l.description,
    l.category,
    l.price_lamports,
    l.ipfs_cid,
    l.created_at,
    COUNT(e.id) as escrow_count
FROM listings l
LEFT JOIN escrows e ON l.listing_id = e.listing_id
WHERE l.active = TRUE
GROUP BY l.id;

-- Escrow summary with state
CREATE VIEW escrow_summary AS
SELECT
    e.escrow_pubkey,
    e.buyer_pubkey_hash,
    e.seller_pubkey_hash,
    e.amount_lamports,
    e.state,
    l.title as listing_title,
    l.price_lamports as listing_price,
    e.created_at,
    e.completed_at,
    EXTRACT(EPOCH FROM (COALESCE(e.completed_at, NOW()) - e.created_at)) as duration_seconds
FROM escrows e
LEFT JOIN listings l ON e.listing_id = l.listing_id;

-- User note balances
CREATE VIEW user_balances AS
SELECT
    owner_pubkey_hash,
    COUNT(*) as total_notes,
    COUNT(*) FILTER (WHERE spent = FALSE) as unspent_notes,
    SUM(amount_lamports) FILTER (WHERE spent = FALSE) as balance_lamports
FROM user_notes
GROUP BY owner_pubkey_hash;

-- ============================================================================
-- FUNCTIONS
-- Database logic
-- ============================================================================

-- Mark note as spent (atomic operation)
CREATE OR REPLACE FUNCTION spend_note(
    p_commitment VARCHAR(64),
    p_nullifier_hash VARCHAR(64),
    p_tx_signature VARCHAR(88)
) RETURNS BOOLEAN AS $$
DECLARE
    v_note_id BIGINT;
BEGIN
    -- Check if already spent
    IF EXISTS (SELECT 1 FROM nullifier_registry WHERE nullifier_hash = p_nullifier_hash) THEN
        RAISE EXCEPTION 'Note already spent (double-spend attempt)';
    END IF;

    -- Mark note as spent
    UPDATE user_notes
    SET spent = TRUE, spent_at = NOW()
    WHERE commitment = p_commitment AND spent = FALSE
    RETURNING id INTO v_note_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Note not found or already spent';
    END IF;

    -- Register nullifier
    INSERT INTO nullifier_registry (nullifier_hash, commitment, tx_signature)
    VALUES (p_nullifier_hash, p_commitment, p_tx_signature);

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SECURITY
-- Row-level security policies (optional, for multi-tenant scenarios)
-- ============================================================================

-- Enable RLS on sensitive tables
-- ALTER TABLE user_notes ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE stealth_addresses ENABLE ROW LEVEL SECURITY;

-- Example policy: Users can only see their own notes
-- CREATE POLICY user_notes_policy ON user_notes
--     USING (owner_pubkey_hash = current_setting('app.current_user_hash')::VARCHAR);

-- ============================================================================
-- MAINTENANCE
-- ============================================================================

-- Function to clean up old audit logs (run monthly)
CREATE OR REPLACE FUNCTION cleanup_old_audit_logs(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM audit_log
    WHERE created_at < NOW() - (days_to_keep || ' days')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert default Merkle tree
INSERT INTO merkle_trees (tree_pubkey, root, next_index)
VALUES ('PrivacyPoolTreePubkey1111111111111111111', '0000000000000000000000000000000000000000000000000000000000000000', 0)
ON CONFLICT (tree_pubkey) DO NOTHING;

-- ============================================================================
-- COMMENTS
-- Documentation for maintainability
-- ============================================================================

COMMENT ON TABLE user_notes IS 'Privacy pool notes with encrypted secrets/nullifiers';
COMMENT ON COLUMN user_notes.secret_encrypted IS 'AES-256-GCM encrypted note secret (JSONB with ciphertext/nonce)';
COMMENT ON COLUMN user_notes.nullifier_encrypted IS 'AES-256-GCM encrypted note nullifier (prevents double-spending)';
COMMENT ON COLUMN user_notes.owner_pubkey_hash IS 'SHA256 hash of owner pubkey (for privacy)';

COMMENT ON TABLE listings IS 'Marketplace product listings';
COMMENT ON TABLE escrows IS 'Escrow state machine for secure buyer-seller transactions';
COMMENT ON TABLE messages IS 'Encrypted communications between buyers and sellers';
COMMENT ON TABLE nullifier_registry IS 'Prevents double-spending of privacy notes';
COMMENT ON TABLE stealth_addresses IS 'One-time use addresses with encrypted private keys';
COMMENT ON TABLE audit_log IS 'Security and compliance event tracking';

COMMENT ON FUNCTION spend_note IS 'Atomically mark note as spent and register nullifier';
