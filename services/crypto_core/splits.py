# crypto_core/splits.py
from __future__ import annotations
from decimal import Decimal, ROUND_DOWN
import secrets
from typing import List, Dict, Tuple
import random

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

def split_bounded(total: Decimal, n: int, low: float = 0.5, high: float = 1.5) -> list[Decimal]:
    """
    Split `total` into `n` random parts, each between [low*avg, high*avg],
    where avg = total/n. Parts sum exactly to total (1e-9 precision).
    """
    if n <= 0:
        return []
    if n == 1:
        return [total.quantize(Decimal("0.000000001"))]

    avg = total / Decimal(n)
    min_amt = avg * Decimal(str(low))
    max_amt = avg * Decimal(str(high))

    # Step 1: sample n random floats in [low, high]
    raw = [random.uniform(low, high) for _ in range(n)]
    s = sum(raw)

    # Step 2: normalize to sum = total
    scaled = [total * Decimal(r / s) for r in raw]

    # Step 3: clamp + quantize
    q = Decimal("0.000000001")
    parts = [max(min_amt, min(max_amt, x)).quantize(q, rounding=ROUND_DOWN) for x in scaled]

    # Step 4: adjust last element to fix rounding error
    diff = total - sum(parts)
    parts[-1] = (parts[-1] + diff).quantize(q, rounding=ROUND_DOWN)

    # Step 5: shuffle to hide order
    random.shuffle(parts)
    return parts