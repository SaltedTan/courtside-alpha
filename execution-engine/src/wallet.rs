// ============================================================================
//  wallet.rs — Testnet EIP-712 order signing for Polymarket CLOB
//
//  Uses a deterministic secp256k1 key derived from a fixed seed.
//  NO REAL FUNDS — this key is published in source code and is for demo only.
//
//  Flow per trade:
//    1.  Build a Polymarket Order struct (amount, tokenId, side, expiry, …)
//    2.  Compute EIP-712 struct_hash → domain_separator → final digest
//    3.  Sign with secp256k1 (RFC 6979 deterministic) → r ‖ s ‖ v (65 bytes)
//    4.  Return hex-encoded order_hash + signed_tx for DB storage & dashboard
// ============================================================================

use anyhow::{Context, Result};
use num_bigint::BigUint;
use secp256k1::{Message, PublicKey, SecretKey, SECP256K1};
use sha3::{Digest, Keccak256};

// ── Polymarket contract constants ─────────────────────────────────────────────

/// Polymarket CTF Exchange on Polygon mainnet (Chain ID 137).
/// We sign against the real domain separator so the signatures are
/// structurally valid — but the test key holds zero real USDC.
pub const CHAIN_ID: u64 = 137;
const EXCHANGE_HEX: &str = "4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E";

/// Deterministic seed — regenerating the engine always produces the same key.
/// IMPORTANT: This wallet holds NO real funds. Do not send assets here.
const TEST_SEED: &str = "courtside-alpha-shadow-trader-testnet-2026-no-real-funds";

// ── EIP-712 type strings ──────────────────────────────────────────────────────

const DOMAIN_TYPEHASH_STR: &str =
    "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)";

/// Matches Polymarket's CTF Exchange Order ABI exactly.
const ORDER_TYPEHASH_STR: &str =
    "Order(uint256 salt,address maker,address signer,address taker,uint256 tokenId,\
     uint256 makerAmount,uint256 takerAmount,uint256 expiration,uint256 nonce,\
     uint256 feeRateBps,uint8 side,uint8 signatureType)";

// ── Public types ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum Side {
    Buy  = 0,
    Sell = 1,
}

/// Output of a successful order sign operation.
pub struct SignedOrder {
    /// "0x…" — keccak256 EIP-712 digest (the hash that was signed).
    pub order_hash: String,
    /// "0x…" — 65-byte compact signature: r(32) ‖ s(32) ‖ v(1).
    pub signed_tx:  String,
    /// USDC amount in base units (6 decimals), e.g. $50 → 50_000_000.
    pub maker_amount: u64,
    /// Outcome tokens expected in base units.
    pub taker_amount: u64,
}

// ── Wallet ────────────────────────────────────────────────────────────────────

pub struct TestWallet {
    secret:      SecretKey,
    /// Ethereum address as "0x…" hex string.
    pub address: String,
    addr_raw:    [u8; 20],
}

impl TestWallet {
    /// Derive a deterministic test wallet from the built-in seed phrase.
    pub fn new() -> Result<Self> {
        let seed_hash = keccak256(TEST_SEED.as_bytes());
        let secret    = SecretKey::from_slice(&seed_hash)
            .context("failed to build secp256k1 key from seed hash")?;

        // Ethereum address = keccak256(uncompressed_pubkey[1..])[12..]
        let pubkey       = PublicKey::from_secret_key(SECP256K1, &secret);
        let uncompressed = pubkey.serialize_uncompressed(); // [0x04 | x(32) | y(32)]
        let addr_hash    = keccak256(&uncompressed[1..]);
        let mut addr_raw = [0u8; 20];
        addr_raw.copy_from_slice(&addr_hash[12..]);
        let address = format!("0x{}", hex::encode(addr_raw));

        Ok(Self { secret, address, addr_raw })
    }

    /// Construct and sign a Polymarket CLOB order for a given outcome token.
    ///
    /// * `token_id_str` — Polymarket token ID as a decimal string (uint256).
    /// * `market_prob`  — current market price / implied probability [0, 1].
    /// * `stake_usdc`   — USDC to spend (e.g. 50.0).
    /// * `side`         — Buy or Sell.
    pub fn sign_order(
        &self,
        token_id_str: &str,
        market_prob:  f64,
        stake_usdc:   f64,
        side:         Side,
    ) -> Result<SignedOrder> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Salt: unique per order — deterministic from timestamp + token
        let salt = keccak256(format!("{now}{token_id_str}").as_bytes());

        // USDC and outcome token amounts in base units (6 decimal places)
        let maker_amount = (stake_usdc * 1_000_000.0) as u64;
        let taker_amount = if market_prob > 0.001 {
            ((stake_usdc / market_prob) * 1_000_000.0) as u64
        } else {
            maker_amount
        };

        let expiration     = now + 3_600; // 1-hour TTL
        let nonce          = 0u64;
        let fee_rate_bps   = 0u64;
        let side_byte      = side as u8;
        let sig_type       = 0u8; // 0 = EOA signing

        let taker_addr   = [0u8; 20]; // zero address → any taker (market order)
        let token_id_enc = parse_uint256(token_id_str);

        // ── EIP-712 struct hash ──────────────────────────────────────────────
        let struct_hash = {
            let type_hash = keccak256(ORDER_TYPEHASH_STR.as_bytes());
            let mut buf   = Vec::with_capacity(13 * 32);
            buf.extend_from_slice(&type_hash);
            buf.extend_from_slice(&salt);                          // bytes32 salt
            buf.extend_from_slice(&pad_addr(&self.addr_raw));     // address maker
            buf.extend_from_slice(&pad_addr(&self.addr_raw));     // address signer
            buf.extend_from_slice(&pad_addr(&taker_addr));        // address taker
            buf.extend_from_slice(&token_id_enc);                 // uint256 tokenId
            buf.extend_from_slice(&pad_u64(maker_amount));        // uint256 makerAmount
            buf.extend_from_slice(&pad_u64(taker_amount));        // uint256 takerAmount
            buf.extend_from_slice(&pad_u64(expiration));          // uint256 expiration
            buf.extend_from_slice(&pad_u64(nonce));               // uint256 nonce
            buf.extend_from_slice(&pad_u64(fee_rate_bps));        // uint256 feeRateBps
            buf.extend_from_slice(&pad_u8(side_byte));            // uint8  side
            buf.extend_from_slice(&pad_u8(sig_type));             // uint8  signatureType
            keccak256(&buf)
        };

        // ── Domain separator ─────────────────────────────────────────────────
        let domain_sep = build_domain_separator();

        // ── EIP-712 final digest: keccak256("\x19\x01" ‖ domain ‖ struct) ───
        let mut pre = Vec::with_capacity(2 + 32 + 32);
        pre.push(0x19);
        pre.push(0x01);
        pre.extend_from_slice(&domain_sep);
        pre.extend_from_slice(&struct_hash);
        let order_hash = keccak256(&pre);

        // ── secp256k1 sign (RFC 6979 deterministic, recoverable) ─────────────
        let msg     = Message::from_digest_slice(&order_hash)
            .context("order hash must be exactly 32 bytes")?;
        let rec_sig = SECP256K1.sign_ecdsa_recoverable(&msg, &self.secret);
        let (rec_id, sig_bytes) = rec_sig.serialize_compact(); // ([u8; 64], RecoveryId)
        let v = rec_id.to_i32() as u8 + 27; // Ethereum convention: 27 or 28

        let mut full_sig = [0u8; 65];
        full_sig[..64].copy_from_slice(&sig_bytes);
        full_sig[64] = v;

        Ok(SignedOrder {
            order_hash:   format!("0x{}", hex::encode(order_hash)),
            signed_tx:    format!("0x{}", hex::encode(full_sig)),
            maker_amount,
            taker_amount,
        })
    }
}

// ── EIP-712 domain separator ──────────────────────────────────────────────────

fn build_domain_separator() -> [u8; 32] {
    let type_hash    = keccak256(DOMAIN_TYPEHASH_STR.as_bytes());
    let name_hash    = keccak256(b"Polymarket CTF Exchange");
    let version_hash = keccak256(b"1");

    // Decode the exchange address hex → 20 bytes
    let addr_bytes = hex::decode(EXCHANGE_HEX).unwrap_or_default();
    let mut addr20 = [0u8; 20];
    let src        = addr_bytes.as_slice();
    let n          = src.len().min(20);
    addr20[20 - n..].copy_from_slice(&src[src.len() - n..]);

    let mut buf = Vec::with_capacity(5 * 32);
    buf.extend_from_slice(&type_hash);
    buf.extend_from_slice(&name_hash);
    buf.extend_from_slice(&version_hash);
    buf.extend_from_slice(&pad_u64(CHAIN_ID));     // chainId as uint256
    buf.extend_from_slice(&pad_addr(&addr20));
    keccak256(&buf)
}

// ── Encoding helpers ──────────────────────────────────────────────────────────

fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(data);
    h.finalize().into()
}

/// ABI-encode a 20-byte Ethereum address as a right-aligned 32-byte word.
fn pad_addr(addr: &[u8; 20]) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[12..].copy_from_slice(addr);
    b
}

/// ABI-encode a u64 as a big-endian 32-byte word (uint256).
fn pad_u64(v: u64) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[24..].copy_from_slice(&v.to_be_bytes());
    b
}

/// ABI-encode a u8 as a right-padded 32-byte word (uint8).
fn pad_u8(v: u8) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[31] = v;
    b
}

/// Parse a decimal string (potentially >u64) as big-endian uint256 bytes.
/// Polymarket token IDs are ~77-digit decimal integers.
fn parse_uint256(s: &str) -> [u8; 32] {
    let big   = BigUint::parse_bytes(s.as_bytes(), 10).unwrap_or_default();
    let bytes = big.to_bytes_be();
    let mut out = [0u8; 32];
    let len     = bytes.len().min(32);
    out[32 - len..].copy_from_slice(&bytes[bytes.len() - len..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_wallet_address() {
        let w1 = TestWallet::new().unwrap();
        let w2 = TestWallet::new().unwrap();
        assert_eq!(w1.address, w2.address);
        assert_eq!(w1.address, "0xe0d46b2474bfbac1ee2b467ffd2027fcdb9fc00a");
    }

    #[test]
    fn test_sign_order() {
        let wallet = TestWallet::new().unwrap();
        let token_id = "55157410042453472019910363290640071350812423584852928509425167093226759792078";
        let market_prob = 0.5;
        let stake = 50.0;
        let side = Side::Buy;

        let signed = wallet.sign_order(token_id, market_prob, stake, side).unwrap();
        assert!(signed.order_hash.starts_with("0x"));
        assert_eq!(signed.order_hash.len(), 66); // 0x + 64 hex chars
        assert!(signed.signed_tx.starts_with("0x"));
        assert_eq!(signed.signed_tx.len(), 132); // 0x + 130 hex chars (65 bytes)
        assert_eq!(signed.maker_amount, 50_000_000);
        assert_eq!(signed.taker_amount, 100_000_000);
    }

    /// Cryptographic round-trip: sign a digest, recover the public key,
    /// re-derive the Ethereum address, and confirm it matches the wallet.
    #[test]
    fn test_signature_recovery() {
        use secp256k1::{Message, PublicKey, SECP256K1};
        use secp256k1::ecdsa::{RecoverableSignature, RecoveryId};

        let wallet = TestWallet::new().unwrap();

        // Sign an arbitrary 32-byte digest
        let digest = keccak256(b"recovery test");
        let msg    = Message::from_digest_slice(&digest).unwrap();
        let rec_sig = SECP256K1.sign_ecdsa_recoverable(&msg, &wallet.secret);
        let (rec_id, sig_bytes) = rec_sig.serialize_compact();

        // Recover the public key from (sig, digest)
        let rec_id2   = RecoveryId::from_i32(rec_id.to_i32()).unwrap();
        let rec_sig2  = RecoverableSignature::from_compact(&sig_bytes, rec_id2).unwrap();
        let recovered: PublicKey = SECP256K1.recover_ecdsa(&msg, &rec_sig2).unwrap();

        // Derive Ethereum address from the recovered public key
        let uncompressed = recovered.serialize_uncompressed();
        let addr_hash    = keccak256(&uncompressed[1..]);
        let recovered_addr = format!("0x{}", hex::encode(&addr_hash[12..]));

        assert_eq!(
            recovered_addr, wallet.address,
            "Recovered address must match the wallet's signing address"
        );
    }
}
