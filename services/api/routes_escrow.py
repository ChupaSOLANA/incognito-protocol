# services/api/routes_escrow.py
from __future__ import annotations

from fastapi import APIRouter, Query
from typing import Optional, Literal

from services.api.schemas_api import EscrowListRes

# ---- Importe la même source d'état que /marketplace/buy utilise ----
# Adapte ces imports selon où vivent tes helpers (_escrow_state)
try:
    # recommandé: un petit module escrow_store centralise l'état
    from services.api.escrow_store import escrow_state as _escrow_state
except Exception:
    # fallback si _escrow_state est défini dans le même module que marketplace/buy
    from services.api.routes_marketplace import _escrow_state  # type: ignore

router = APIRouter()

@router.get("/escrow/list", response_model=EscrowListRes)
def escrow_list(
    party_pub: str = Query(..., description="Buyer or seller pubkey (base58)"),
    role: Literal["buyer", "seller"] = Query(...),
    status: Optional[str] = Query(None),
):
    """
    Liste les escrows où `party_pub` est impliqué (côté buyer ou seller).
    On lit exactement le même store que /marketplace/buy a rempli.
    """
    st = _escrow_state()  # IMPORTANT: même backing que buy
    items = list(st.get("escrows", []))

    if role == "buyer":
        items = [e for e in items if e.get("buyer_pub") == party_pub]
    else:
        items = [e for e in items if e.get("seller_pub") == party_pub]

    if status:
        items = [e for e in items if e.get("status") == status]

    # Façonne la vue publique (on enlève les champs internes comme payment_mode, note_hex, etc.)
    def _public(e: dict) -> dict:
        return {
            "id": e["id"],
            "buyer_pub": e["buyer_pub"],
            "seller_pub": e["seller_pub"],
            "amount_sol": str(e["amount_sol"]),
            "status": e["status"],
            "details_ct": e.get("details_ct"),
            "listing_id": e.get("listing_id"),
            "quantity": e.get("quantity"),
            "commitment": e["commitment"],
            "leaf_index": e.get("leaf_index"),
            "created_at": e["created_at"],
            "updated_at": e["updated_at"],
        }

    return EscrowListRes(items=[_public(e) for e in items])
