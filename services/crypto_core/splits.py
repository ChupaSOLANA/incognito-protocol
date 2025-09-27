# crypto_core/splits.py
from __future__ import annotations
from decimal import Decimal, ROUND_DOWN
import secrets
from typing import List, Dict, Tuple

def random_split_amounts(total: Decimal, n: int) -> List[Decimal]:
    """Return n positive Decimal parts that sum to total."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    remain = total
    parts: List[Decimal] = []
    for i in range(n - 1):
        pct = Decimal(str((secrets.randbelow(51) + 20) / 100))  # 0.20..0.70
        part = (remain * pct).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
        if part <= Decimal("0"):
            part = Decimal("0.000000001")
        if remain - part <= Decimal("0"):
            part = remain / Decimal(n - i)
        parts.append(part)
        remain -= part
    parts.append(remain.quantize(Decimal("0.000000001"), rounding=ROUND_DOWN))
    s = sum(parts, Decimal("0"))
    drift = (total - s).quantize(Decimal("0.000000001"))
    if drift != 0:
        parts[-1] = (parts[-1] + drift).quantize(Decimal("0.000000001"))
    return parts

def greedy_coin_select(notes: List[Dict], target: Decimal) -> Tuple[List[Dict], Decimal]:
    cand = sorted(notes, key=lambda n: Decimal(str(n["amount"])))
    total = Decimal("0")
    chosen = []
    for n in cand:
        chosen.append(n)
        total += Decimal(str(n["amount"]))
        if total + Decimal("0.000000001") >= target:
            return chosen, total
    return [], Decimal("0")
