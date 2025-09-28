# dashboard/app/dashboard.py
import os, sys, json, time, subprocess, requests, pathlib
from decimal import Decimal
import streamlit as st

# ---------------- Config ----------------
st.set_page_config(page_title="Incognito – Demo", page_icon="🕶️", layout="wide")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

# Threshold for showing stealth entries (hide dust)
MIN_STEALTH_SOL = 0.01  # filter out addresses below this balance

# --- Make repo root importable so clients/cli/... works when running from dashboard/app/ ---
REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])  # ../../ from this file
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from CLI (MINT + wrapper helpers)
_total_available_for_recipient = None
_load_wrapper_state = None
try:
    from clients.cli import incognito_marketplace as mp  # your CLI module
    MINT = mp.MINT
    _total_available_for_recipient = mp.total_available_for_recipient  # (state, pubkey) -> Decimal
    _load_wrapper_state = mp._load_wrapper_state
except Exception as e:
    MINT = os.getenv("MINT", "6ScGfdRoKuk4gjHVbFjBMwxLdgqxx5gHwKLaZTTj3Zrw")
    st.warning(f"[dashboard] Could not import clients.cli.incognito_marketplace: {e}")

# --------------- Utils ------------------
def short(pk: str, n=6) -> str:
    return pk if not pk or len(pk) <= 2*n else f"{pk[:n]}…{pk[-n:]}"

def list_user_keyfiles(keys_dir: str = "keys"):
    if not os.path.isdir(keys_dir):
        return []
    return sorted(
        os.path.join(keys_dir, f)
        for f in os.listdir(keys_dir)
        if f.endswith(".json") and f.lower().startswith("user")
    )

def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate()
        return proc.returncode, (out or "").strip(), (err or "").strip()
    except Exception as e:
        return 1, "", str(e)

def get_pubkey_from_keypair(path: str) -> str | None:
    rc, out, err = run_cmd(["solana-keygen", "pubkey", path])
    if rc == 0 and out:
        return out.strip()
    st.error(f"solana-keygen error: {err or out}")
    return None

def get_sol_balance(pubkey: str) -> Decimal | None:
    rc, out, err = run_cmd(["solana", "balance", pubkey])
    if rc == 0 and out:
        try:
            return Decimal(out.split()[0])
        except Exception:
            return None
    return None

def api_get(path: str, **params):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=20)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"error": str(e)}

def api_post(path: str, payload: dict):
    headers = {"content-type": "application/json"}
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, headers=headers, timeout=90)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        return 0, {"error": str(e)}

def fmt_amt(x) -> str:
    try:
        return str(Decimal(str(x)).quantize(Decimal("0.000000001")))
    except Exception:
        return str(x)

def ensure_state():
    st.session_state.setdefault("active_keyfile", None)
    st.session_state.setdefault("active_pub", None)
    st.session_state.setdefault("blur_amounts", True)
    st.session_state.setdefault("auto_refresh", False)
    st.session_state.setdefault("auto_refresh_interval", 10)
ensure_state()

def safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def flash(msg: str, kind: str = "info", seconds: float = 1.5):
    ph = st.empty()
    if   kind == "success": ph.success(msg)
    elif kind == "warning": ph.warning(msg)
    elif kind == "error":   ph.error(msg)
    else:                   ph.info(msg)
    time.sleep(seconds)
    ph.empty()

# ---- Known wallets (cache) ----
@st.cache_data(ttl=60)
def load_known_wallets(keys_dir: str = "keys") -> list[dict]:
    items = []
    for path in list_user_keyfiles(keys_dir):
        rc, out, err = run_cmd(["solana-keygen", "pubkey", path])
        if rc == 0 and out.strip():
            items.append({"keyfile": path, "pubkey": out.strip()})
    return items

def other_known_wallet_pks(active_keyfile: str | None) -> list[dict]:
    wallets = load_known_wallets()
    out = []
    for w in wallets:
        if active_keyfile and w["keyfile"] == active_keyfile:
            continue
        label = f"{os.path.basename(w['keyfile'])} – {short(w['pubkey'])}"
        out.append({"label": label, "pubkey": w["pubkey"]})
    return out

def recipient_selector(ui_key: str, active_keyfile: str, default_pub: str) -> str:
    others = other_known_wallet_pks(active_keyfile)
    if not others:
        return st.text_input("Recipient public key (base58)", value=default_pub, key=f"{ui_key}_custom_only").strip()
    mode = st.radio("Recipient", options=["Known wallet", "Custom"], horizontal=True, key=f"{ui_key}_mode")
    if mode == "Known wallet":
        options = [o["label"] for o in others]
        sel = st.selectbox("Choose wallet", options=options, key=f"{ui_key}_known_sel")
        idx = options.index(sel)
        return others[idx]["pubkey"]
    else:
        return st.text_input("Recipient public key (base58)", value=default_pub, key=f"{ui_key}_custom").strip()

# ---- Available wrapper notes (primary via CLI; fallback via JSON) ----
def _fallback_available_from_state(pub: str) -> Decimal | None:
    try:
        path = pathlib.Path(REPO_ROOT) / "merkle_state.json"
        if not path.exists():
            return None
        stt = json.loads(path.read_text())
        total = Decimal("0")
        for n in stt.get("notes", []):
            if not n.get("spent", False) and n.get("recipient_pub") == pub:
                try:
                    total += Decimal(str(n.get("amount", "0")))
                except Exception:
                    pass
        return total
    except Exception:
        return None

def available_wrapper_for(pub: str) -> Decimal | None:
    if _total_available_for_recipient and _load_wrapper_state:
        try:
            stt = _load_wrapper_state()
            return Decimal(str(_total_available_for_recipient(stt, pub)))
        except Exception:
            pass
    return _fallback_available_from_state(pub)

# --------------- Sidebar: user + options ---------------
with st.sidebar:
    st.header("User")
    user_files = list_user_keyfiles("keys")
    if not user_files:
        st.error("No user keypairs found in ./keys (expect userA.json, userB.json, …)")
        st.stop()

    default_idx = 0
    if st.session_state["active_keyfile"] in user_files:
        default_idx = user_files.index(st.session_state["active_keyfile"])
    sel = st.selectbox("Keypair (./keys)", options=user_files, index=default_idx)

    if st.button("🔓 Load user", use_container_width=True):
        pub = get_pubkey_from_keypair(sel)
        if pub:
            st.session_state["active_keyfile"] = sel
            st.session_state["active_pub"] = pub
            st.toast(f"Loaded {os.path.basename(sel)} → {short(pub)}", icon="✅")
            safe_rerun()

    st.divider()
    st.subheader("Privacy")
    st.session_state["blur_amounts"] = st.toggle(
        "Blur amounts by default",
        value=st.session_state["blur_amounts"],
        help="Toggle masking of balances and totals.",
    )

    st.divider()
    st.subheader("Refresh")
    st.session_state["auto_refresh"] = st.toggle(
        "Enable auto-refresh",
        value=st.session_state["auto_refresh"],
        help="Periodically re-run to update balances and metrics.",
    )
    st.session_state["auto_refresh_interval"] = st.number_input(
        "Interval (seconds)", min_value=3, max_value=120,
        value=int(st.session_state["auto_refresh_interval"]), step=1,
    )
    if st.button("Refresh now", use_container_width=True):
        safe_rerun()

# Guard
if not st.session_state["active_pub"]:
    st.title("Incognito – Demo")
    st.info("Pick a user in the left sidebar, then click **Load user**.")
    st.stop()

active_key = st.session_state["active_keyfile"]
active_pub = st.session_state["active_pub"]

# --------------- Top metric (SOL only) ---------------
st.markdown(f"### Active user: `{os.path.basename(active_key)}` · **{active_pub}**")
bal = get_sol_balance(active_pub)
text = "•••" if st.session_state["blur_amounts"] and bal is not None else (fmt_amt(bal) if bal is not None else "n/a")
st.metric("SOL balance", text)
st.divider()

# --------------- Shared stealth data (used by Stealth & Sweep) ---------------
stealth_data = None
c_stealth, d_stealth = api_get(
    f"/stealth/{active_pub}",
    include_balances="true",
    min_sol=MIN_STEALTH_SOL,    # <<< filter out dust from the API side
)
if c_stealth == 200 and isinstance(d_stealth, dict):
    stealth_data = d_stealth

# --------------- Tabs (order like CLI) ---------------
tab_deposit, tab_withdraw, tab_handoff, tab_convert, tab_stealth, tab_sweep, tab_overview, tab_events = st.tabs(
    ["Deposit", "Withdraw", "Handoff", "Convert", "Stealth", "Sweep", "Overview", "Events"]
)

# --- Deposit ---
with tab_deposit:
    st.subheader("Shielded Deposit → Note to recipient + Pool/Stealth splits")

    recip_pub = recipient_selector("deposit_recipient", active_key, active_pub)

    # Fixed denominations only
    DENOMS = ["10", "25", "50", "100"]
    denom_label = st.selectbox("Select amount (SOL)", options=DENOMS, index=0)
    amt = denom_label  # string

    if st.button("Deposit", type="primary"):
        payload = {"depositor_keyfile": active_key, "recipient_pub": recip_pub.strip(), "amount_sol": amt}
        flash("Submitting deposit…")
        c, res = api_post("/deposit", payload)
        if c == 200:
            flash("Deposit sent ✅", "success")
            st.json(res)
            safe_rerun()
        else:
            flash("Deposit failed ❌", "error")
            st.error(res)

# --- Withdraw ---
with tab_withdraw:
    st.subheader("Shielded Withdraw (mint cSOL → confidential transfer)")

    # Show available from wrapper notes (via helpers or fallback)
    avail = available_wrapper_for(active_pub)
    if avail is not None:
        shown = "•••" if st.session_state["blur_amounts"] else fmt_amt(avail)
        st.info(f"Available from wrapper notes: **{shown} SOL**")
    else:
        st.info("Available from wrapper notes: n/a")

    all_amt = st.checkbox("Withdraw ALL available", value=True)
    amt_input = None
    if not all_amt:
        amt_input = st.text_input("Amount (SOL)", value="3", key="withdraw_amt")

    if st.button("Withdraw", type="primary"):
        payload = {"recipient_keyfile": active_key}
        if not all_amt and amt_input:
            payload["amount_sol"] = amt_input.strip()
        flash("Submitting withdraw…")
        c, res = api_post("/withdraw", payload)
        if c == 200:
            flash("Withdraw sent ✅", "success")
            st.json(res)
            safe_rerun()
        else:
            flash("Withdraw failed ❌", "error")
            st.error(res)

# --- Handoff ---
with tab_handoff:
    st.subheader("Off-chain Blind Handoff (notes A → B)")
    recip_pub = recipient_selector("handoff_recipient", active_key, active_pub)
    amt_h = st.text_input("Amount (SOL)", value="5", key="handoff_amt")
    nout = st.number_input("Number of blinded outputs", value=2, min_value=1, step=1)
    if st.button("Handoff", type="primary"):
        payload = {"sender_keyfile": active_key, "recipient_pub": recip_pub.strip(), "amount_sol": amt_h.strip(), "n_outputs": int(nout)}
        flash("Submitting handoff…")
        c, res = api_post("/handoff", payload)
        if c == 200:
            flash("Handoff done ✅", "success")
            st.json(res)
            safe_rerun()
        else:
            flash("Handoff failed ❌", "error")
            st.error(res)

# --- Convert ---
with tab_convert:
    st.subheader("Convert cSOL → SOL (to self stealth)")
    amt_c = st.text_input("Amount (cSOL)", value="4", key="convert_amt")
    nout_c = st.number_input("Number of stealth outputs", value=3, min_value=1, step=1, key="convert_n")
    if st.button("Convert", type="primary"):
        payload = {"sender_keyfile": active_key, "amount_sol": amt_c.strip(), "n_outputs": int(nout_c)}
        flash("Submitting convert…")
        c, res = api_post("/convert", payload)
        if c == 200:
            flash("Convert sent ✅", "success")
            st.json(res)
            safe_rerun()
        else:
            flash("Convert failed ❌", "error")
            st.error(res)

# --- Stealth ---
with tab_stealth:
    st.subheader(f"Stealth addresses (≥ {MIN_STEALTH_SOL} SOL)")
    if stealth_data:
        items = stealth_data.get("items", [])
        total = stealth_data.get("total_sol", "0")

        # Render as table (address + balance)
        st.markdown("#### Addresses")
        rows = []
        for it in items:
            pk = it.get("stealth_pubkey")
            bal = it.get("balance_sol") or "0"
            shown_bal = "•••" if st.session_state["blur_amounts"] else fmt_amt(bal)
            rows.append({"Stealth address": short(pk, 8), "Balance (SOL)": shown_bal})
        st.table(rows)

        # Total
        shown_total = "•••" if st.session_state["blur_amounts"] else total
        st.success(f"**Total across listed:** {shown_total} SOL")
    else:
        st.error(d_stealth)

# --- Sweep ---
with tab_sweep:
    st.subheader("Sweep funds from stealth → destination pubkey")

    # Show total sweepable using the same filtered stealth_data
    if stealth_data:
        total = stealth_data.get("total_sol", "0")
        shown_total = "•••" if st.session_state["blur_amounts"] else total
        st.info(f"Sweepable total (≥ {MIN_STEALTH_SOL} SOL per address): **{shown_total} SOL**")
    else:
        st.warning("Sweepable total not available.")

    others = other_known_wallet_pks(active_key)
    dest_pub = st.text_input(
        "Destination pubkey",
        value=(others[0]["pubkey"] if others else active_pub),
        help="You can paste any pubkey; default is first other known user or yourself.",
    )
    all_amt_sw = st.checkbox("Sweep ALL available (excl. buffer)", value=True, key="sweep_all")
    amt_sw = None
    if not all_amt_sw:
        amt_sw = st.text_input("Amount (SOL)", value="1.5", key="sweep_amt")
    if st.button("Sweep", type="primary"):
        payload = {"owner_pub": active_pub, "secret_keyfile": active_key, "dest_pub": dest_pub.strip()}
        if not all_amt_sw and amt_sw:
            payload["amount_sol"] = amt_sw.strip()
        flash("Submitting sweep…")
        c, res = api_post("/sweep", payload)
        if c == 200:
            flash("Sweep sent ✅", "success")
            st.json(res)
            safe_rerun()
        else:
            flash("Sweep failed ❌", "error")
            st.error(res)

# --- Overview ---
with tab_overview:
    st.subheader("Merkle status")
    code_status, status = api_get("/merkle/status")
    st.json(status if code_status == 200 else {"error": status})

# --- Events ---
with tab_events:
    st.subheader("Metrics")
    code_m, metrics = api_get("/metrics")
    st.json(metrics if code_m == 200 else {"error": metrics})

# --------------- Auto-refresh loop ---------------
if st.session_state["auto_refresh"]:
    interval = int(st.session_state["auto_refresh_interval"])
    placeholder = st.empty()
    with placeholder.container():
        st.caption(f"Auto-refresh in {interval}s…")
    time.sleep(interval)
    safe_rerun()
