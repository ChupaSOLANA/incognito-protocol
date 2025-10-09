# clients/cli/listings.py
from __future__ import annotations

import os
import json
import time
import hashlib
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

# Merkle tree util maison (déjà présent dans le repo)
from services.crypto_core.merkle import MerkleTree

STATE_PATH = Path(os.getenv("LISTINGS_STATE", "state/listings_state.json")).resolve()
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ------------ State I/O ------------
def _load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text("utf-8"))
        except Exception:
            pass
    return {"records": [], "leaves": [], "root": None, "updated_at": int(time.time())}

def _save_state(st: Dict[str, Any]) -> None:
    st["updated_at"] = int(time.time())
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_PATH)

# ------------ Helpers ------------
CANON_KEYS = ("id","seller_pub","title","description","unit_price_sol","quantity","active","images")

def _canon_listing(rec: Dict[str, Any]) -> Dict[str, Any]:
    # sous-ensemble stable et tri de clés pour hash
    out = {k: rec.get(k, None) for k in CANON_KEYS}
    # normalisations légères
    out["quantity"] = int(out.get("quantity") or 0)
    out["active"] = bool(out.get("active", True)) and out["quantity"] > 0
    # images doit rester une liste simple
    imgs = out.get("images") or []
    out["images"] = list(map(str, imgs)) if isinstance(imgs, (list, tuple)) else []
    # string price
    up = out.get("unit_price_sol")
    out["unit_price_sol"] = str(up) if up is not None else "0"
    return out

def _commitment_of(rec: Dict[str, Any]) -> str:
    canon = _canon_listing(rec)
    blob = json.dumps(canon, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def _rebuild_merkle(st: Dict[str, Any]) -> None:
    st["leaves"] = [r["commitment"] for r in st["records"]]
    if st["leaves"]:
        mt = MerkleTree(st["leaves"])
        if not mt.layers and getattr(mt, "leaf_bytes", None):
            mt.build_tree()
        st["root"] = mt.root().hex()
    else:
        st["root"] = None

def _index_by_id(st: Dict[str, Any], listing_id_hex: str) -> int:
    for i, r in enumerate(st["records"]):
        if r.get("id") == listing_id_hex:
            return i
    return -1

def _assert_owner(rec: Dict[str, Any], owner_pubkey: str) -> None:
    if rec.get("seller_pub") != owner_pubkey:
        raise PermissionError("owner mismatch")

# ------------ API (utilisées par services/api/cli_adapter.py) ------------

def create_listing(
    owner_pubkey: str,
    title: str,
    description: Optional[str] = None,
    price_sol: str = "0",
    quantity: int = 1,
    image_uris: Optional[List[str]] = None,
    # tolère l’alias 'name' envoyé par l’adapter
    name: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    st = _load_state()
    listing_id = "0x" + secrets.token_hex(16)  # id random (on pourrait aussi hasher)
    rec = {
        "id": listing_id,
        "seller_pub": owner_pubkey,
        "title": title or name or "Untitled",
        "description": description,
        "unit_price_sol": str(price_sol),
        "price": str(price_sol),            # compat /marketplace/buy qui lit 'price'
        "quantity": int(quantity),
        "active": True,
        "images": list(image_uris or []),
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }
    rec["active"] = rec["quantity"] > 0
    rec["commitment"] = _commitment_of(rec)
    rec["leaf_index"] = len(st["leaves"])
    st["records"].append(rec)
    _rebuild_merkle(st)
    _save_state(st)
    return rec

# alias attendu par l’adapter
listing_create = create_listing

def list_active_listings() -> List[Dict[str, Any]]:
    st = _load_state()
    return [r for r in st["records"] if r.get("active", True) and int(r.get("quantity", 0)) > 0]

# alias
list_listings = list_active_listings
all_listings  = list_active_listings

def list_my_listings(owner_pubkey: str) -> List[Dict[str, Any]]:
    st = _load_state()
    return [r for r in st["records"] if r.get("seller_pub") == owner_pubkey]

# alias
list_by_owner = list_my_listings
get_listings_by_owner = list_my_listings

def get_listing(listing_id_hex: str) -> Optional[Dict[str, Any]]:
    st = _load_state()
    idx = _index_by_id(st, listing_id_hex)
    return st["records"][idx] if idx >= 0 else None

def update_listing_price(owner_pubkey: str, listing_id_hex: str, new_price_sol: str) -> Dict[str, Any]:
    st = _load_state()
    idx = _index_by_id(st, listing_id_hex)
    if idx < 0:
        raise ValueError("not found")
    rec = dict(st["records"][idx])
    _assert_owner(rec, owner_pubkey)
    rec["unit_price_sol"] = str(new_price_sol)
    rec["price"] = str(new_price_sol)  # compat
    rec["updated_at"] = int(time.time())
    rec["commitment"] = _commitment_of(rec)
    st["records"][idx] = rec
    st["leaves"][idx] = rec["commitment"]
    _rebuild_merkle(st)
    _save_state(st)
    return rec

# alias
listing_update_price = update_listing_price

def update_listing_quantity(
    owner_pubkey: str,
    listing_id_hex: str,
    quantity_new: Optional[int] = None,
    quantity_delta: Optional[int] = None,
) -> Dict[str, Any]:
    st = _load_state()
    idx = _index_by_id(st, listing_id_hex)
    if idx < 0:
        raise ValueError("not found")
    rec = dict(st["records"][idx])
    _assert_owner(rec, owner_pubkey)

    q = int(rec.get("quantity", 0))
    if quantity_new is not None:
        q = int(quantity_new)
    if quantity_delta is not None:
        q = q + int(quantity_delta)

    if q < 0:
        q = 0
    rec["quantity"] = q
    rec["active"] = q > 0
    rec["updated_at"] = int(time.time())
    rec["commitment"] = _commitment_of(rec)
    st["records"][idx] = rec
    st["leaves"][idx] = rec["commitment"]
    _rebuild_merkle(st)
    _save_state(st)
    return rec

# alias
listing_update_quantity = update_listing_quantity
set_listing_quantity    = update_listing_quantity

def update_listing_meta(
    owner_pubkey: str,
    listing_id_hex: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    st = _load_state()
    idx = _index_by_id(st, listing_id_hex)
    if idx < 0:
        raise ValueError("not found")
    rec = dict(st["records"][idx])
    _assert_owner(rec, owner_pubkey)

    if title is not None:
        rec["title"] = title
    if description is not None:
        rec["description"] = description

    rec["updated_at"] = int(time.time())
    rec["commitment"] = _commitment_of(rec)
    st["records"][idx] = rec
    st["leaves"][idx] = rec["commitment"]
    _rebuild_merkle(st)
    _save_state(st)
    return rec

# alias
listing_update_meta = update_listing_meta
update_listing       = update_listing_meta  # fallback générique

def update_listing_images(owner_pubkey: str, listing_id_hex: str, image_uris: List[str]) -> Dict[str, Any]:
    st = _load_state()
    idx = _index_by_id(st, listing_id_hex)
    if idx < 0:
        raise ValueError("not found")
    rec = dict(st["records"][idx])
    _assert_owner(rec, owner_pubkey)

    rec["images"] = list(image_uris or [])
    rec["updated_at"] = int(time.time())
    rec["commitment"] = _commitment_of(rec)
    st["records"][idx] = rec
    st["leaves"][idx] = rec["commitment"]
    _rebuild_merkle(st)
    _save_state(st)
    return rec

# alias
listing_update_images = update_listing_images

def deactivate_listing(listing_id_hex: str) -> None:
    """Utilisé par le flow /buy; pas besoin d'owner ici."""
    st = _load_state()
    idx = _index_by_id(st, listing_id_hex)
    if idx < 0:
        return
    rec = dict(st["records"][idx])
    rec["quantity"] = 0
    rec["active"] = False
    rec["updated_at"] = int(time.time())
    rec["commitment"] = _commitment_of(rec)
    st["records"][idx] = rec
    st["leaves"][idx] = rec["commitment"]
    _rebuild_merkle(st)
    _save_state(st)

def remove_listing(owner_pubkey: str, listing_id_hex: str) -> int:
    """Soft-delete: active=False + quantity=0 (conservé pour l'historique/Merkle)."""
    st = _load_state()
    idx = _index_by_id(st, listing_id_hex)
    if idx < 0:
        return 0
    rec = dict(st["records"][idx])
    _assert_owner(rec, owner_pubkey)

    rec["quantity"] = 0
    rec["active"] = False
    rec["updated_at"] = int(time.time())
    rec["commitment"] = _commitment_of(rec)
    st["records"][idx] = rec
    st["leaves"][idx] = rec["commitment"]
    _rebuild_merkle(st)
    _save_state(st)
    return 1

# alias
delete_listing  = remove_listing
listing_delete  = remove_listing
mark_sold       = deactivate_listing
set_inactive    = deactivate_listing
