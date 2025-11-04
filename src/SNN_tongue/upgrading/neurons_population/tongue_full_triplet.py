# SNN with STDP + eligibility trace to simulate an artificial tongue # that continuously learns to recognize multiple tastes (always-on). 
# that continuously learns to recognize multiple tastes.

import brian2 as b
b.prefs.codegen.target = 'numpy'
import matplotlib.pyplot as plt
import sys
import random
import numpy as np
from collections import Counter, deque
from numpy import trapezoid
import time
import math
import shutil

# GDI toggle for activation/deactivation
USE_GDI = True # True => use GDI; False => use rate normalization

# global base rate per every class -> not 500 in total to split among all the classes but 500 for everyone
BASE_RATE_PER_CLASS = 500

# Imbalance/test controls (keep baseline clean = True)
# Rates management and sanity check on it
TASTE_NAMES = ["SWEET", "BITTER", "SALTY", "SOUR", "UMAMI", "FATTY", "SPICY", "UNKNOWN"]
NUM_TASTES  = len(TASTE_NAMES)
UNKNOWN_ID  = TASTE_NAMES.index("UNKNOWN")
NUM_KNOWN   = UNKNOWN_ID
NORMALIZE_TEST_RATES = False # baseline=True; per test sbilanciati metti False
BASE_RATE_KNOWN   = np.full(NUM_KNOWN, 260.0, dtype=float)
BASE_RATE_UNKNOWN = 1.0  # meglio di 0.0 per evitare divisioni per zero
BASE_RATE_PER_TASTE = np.concatenate([BASE_RATE_KNOWN, [BASE_RATE_UNKNOWN]])
#BASE_RATE_PER_TASTE = None

def check_shapes():
    assert len(BASE_RATE_PER_TASTE) == NUM_TASTES, (
        f"BASE_RATE_PER_TASTE len={len(BASE_RATE_PER_TASTE)} "
        f"ma NUM_TASTES={NUM_TASTES}. Allinea le dimensioni."
    )

check_shapes()

# Individual profiles (4thresholds/rewards) -> every new trial => new different individual
def sample_individual(seed=None):
    rng = np.random.default_rng(seed)
    # species baseline for each taste (SWEET,BITTER,SALTY,SOUR,UMAMI,FATTY,SPICY)
    base_hi = np.array([0.45, 0.18, 0.35, 0.30, 0.50, 0.40, 0.40])
    base_lo = np.array([0.10, 0.02, 0.05, 0.05, 0.10, 0.05, 0.05])
    thr0_hi = np.clip(base_hi + rng.normal(0, 0.05, 7), 0.05, 0.95)
    thr0_lo = np.clip(base_lo + rng.normal(0, 0.03, 7), 0.00, 0.70)
    k_hab_hi  = rng.uniform(0.0015, 0.0035, 7)
    k_sens_hi = rng.uniform(0.0005, 0.0015, 7)
    k_hab_lo  = rng.uniform(0.0010, 0.0020, 7)
    k_sens_lo = rng.uniform(0.0003, 0.0010, 7)
    thr0_spice_ind = float(np.clip(0.35 + rng.normal(0, 0.08), 0.10, 0.80)) # SPICY individual threshold
    k_hab_spice_ind  = float(rng.uniform(0.0008, 0.0025)) # SPICY individual habituation withdrawal
    k_sens_spice_ind = float(rng.uniform(0.0005, 0.0018)) # SPICY individual sensitization increase
    return dict(
        thr0_hi=thr0_hi, thr0_lo=thr0_lo,
        k_hab_hi=k_hab_hi, k_sens_hi=k_sens_hi,
        k_hab_lo=k_hab_lo, k_sens_lo=k_sens_lo,
        # SPICY individual parameters
        thr0_spice_hub=thr0_spice_ind,
        k_hab_spice_hub=k_hab_spice_ind,
        k_sens_spice_hub=k_sens_spice_ind,
         # tratti individuali avversione SPICY
        spicy_aversion_trait = float(np.clip(rng.beta(2.0, 2.0), 0.0, 1.0)),  # 0..1
        k_hun_spice = float(rng.uniform(-0.20, -0.10)),  # fame riduce p (negativa)
        k_h2o_spice = float(rng.uniform(+0.15, +0.30))  # sete aumenta p (positiva)
    )

# state bias mapping for every taste (coeff dimensionless)
c_hun_hi = np.array([+0.06, 0.00, +0.01, 0.00, +0.03, +0.04, 0.00])
c_hun_lo = np.array([-0.05, 0.00, 0.00, 0.00, -0.02, -0.03, 0.00])
c_sat_hi = np.array([-0.07, 0.00, 0.00, 0.00, -0.03, -0.05, 0.00])
c_sat_lo = np.array([+0.05, 0.00, 0.00, 0.00, +0.02, +0.03, 0.00])
c_h2o_hi = np.array([0.00, 0.00, -0.08, 0.00, 0.00, 0.00, -0.02])
c_h2o_lo = np.array([0.00, 0.00, +0.02, 0.00, 0.00, 0.00, 0.00])

# state bias mapping (dimensionless)
def apply_internal_state_bias(profile, mod, tn):
    H = float(mod.HUN[0]); S = float(mod.SAT[0]); W = float(mod.H2O[0])
    thr0_hi = np.clip(profile['thr0_hi'] + c_hun_hi*H + c_sat_hi*S + c_h2o_hi*W, 0.05, 0.95)
    thr0_lo = np.clip(profile['thr0_lo'] + c_hun_lo*H + c_sat_lo*S + c_h2o_lo*W, 0.00, 0.70)
    # all neurons except for UNKNOWN
    for tsa in range(unknown_id):
        sl = taste_slice(tsa)
        tn.thr0_hi[sl] = thr0_hi[tsa]
        tn.thr0_lo[sl] = thr0_lo[tsa]
    #tn.thr0_hi[:unknown_id] = thr0_hi
    #tn.thr0_lo[:unknown_id] = thr0_lo

# SPICY state bias mapping (different from other tastes) (dimensionless)
# fame aumenta tolleranza, sete/sudore la diminuisce
# esempio: fame ti rende più tollerante (baseline un filo più bassa),
# sete/sudore ti rende meno tollerante (baseline più alta)
def apply_spicy_state_bias(profile, mod, tn,
                           k_hun=+0.05, k_h2o=-0.06):
    H = float(mod.HUN[0])
    W = float(mod.H2O[0])
    base = float(profile['thr0_spice_hub'])
    # esempio: fame ti rende più tollerante (baseline un filo più bassa),
    # sete/sudore ti rende meno tollerante (baseline più alta)
    tn.thr0_spice[taste_slice(spicy_id)] = float(np.clip(base + k_h2o*W + k_hun*(-H), 0.05, 0.95))

# time functions for timing training and test phase to calculate ETA
def fmt_mmss(seconds: float) -> str:
    m, s = divmod(int(max(0.0, seconds) + 0.5), 60)
    return f"{m:02d}:{s:02d}"

# seed for the random reproducibility
random.seed(0)
np.random.seed(0)
b.seed(0)

# single-line progress bar helpers
_last_pbar_len = 0

# deleting residual characters and mantaining the same line for the terminal bar
def pbar_update(msg, stream=sys.stdout):
    cols = shutil.get_terminal_size(fallback=(100, 30)).columns
    if cols and len(msg) >= cols:
        msg = msg[:cols-1] # on the same line -> not a new line
    stream.write('\r\x1b[2K' + msg)
    stream.flush()

# close the bar and new line
def pbar_done(stream=sys.stdout):
    global _last_pbar_len
    stream.write('\n')
    stream.flush()
    _last_pbar_len = 0

# Rates vector helper with normalization including UNKNOWN only when it is needed
def set_stimulus_vect_norm(rate_vec, total_rate=None, include_unknown=False):
    r = np.asarray(rate_vec, float).copy()
    unk = r[unknown_id] if include_unknown else 0.0
    r[unknown_id] = 0.0
    if total_rate is not None and r.sum() > 0:
        r *= float(total_rate) / r.sum()
    r[unknown_id] = unk
    pg.rates = r * b.Hz

# Rates vector helper without normalization
def set_stimulus_vector(rate_vec, include_unknown=False):
    r = np.asarray(rate_vec, dtype=float).copy()
    if not include_unknown:
        r[unknown_id] = 0.0
    pg.rates = r * b.Hz

# Rates vector helper for test stimulus with zero UNKNOWN rate always and OVERSAMPLING for imbalanced data
# only for test phase
def set_test_stimulus(rate_vec):
    r = np.asarray(rate_vec, dtype=float).copy()
    r[unknown_id] = 0.0  # UNKNOWN sempre 0 al test
    # without imbalanced data
    if NORMALIZE_TEST_RATES:
        tot = r[:unknown_id].sum()  # somma solo noti
        if tot > 0:
            k = max(1, (r[:unknown_id] > 0).sum())
            r[:unknown_id] *= (BASE_RATE_PER_CLASS * k) / tot
    else:
        # profilo per-gusto: se fornito, sovrascrive i canali attivi
        if BASE_RATE_PER_TASTE is not None:
            base = np.asarray(BASE_RATE_PER_TASTE, dtype=float)
            assert base.shape[0] == num_tastes, "BASE_RATE_PER_TASTE deve avere len=num_tastes"
            base = np.maximum(base, 1e-6) # avoiding per-zero division
            r[:unknown_id] = base[:unknown_id] * (r[:unknown_id] > 0)

    pg.rates = r * b.Hz

# EMA decoder helpers
def ema_update(m1, m2, x, lam):
    # scaling single updating
    m1_new = (1.0 - lam) * m1 + lam * x
    m2_new = (1.0 - lam) * m2 + lam * (x * x)
    return m1_new, m2_new

def ema_sd(m1, m2):
    var = max(1e-9, m2 - m1*m1)
    return np.sqrt(var)

# UNSUPERVISED LEARNING helpers => NNLS lightweight solver
#    A: (C, K) prototipi, tipicamente K=C se usi una colonna per classe
#    b: (C,) vettore score osservato
#    Ritorna alpha (K,), alpha>=0, sum(alpha)<=l1_cap
# 0.
def _project_to_simplex(va, z=1.0):
    # Proietta v >=0 con somma esattamente z (o <=z se già sotto)
    va = np.asarray(va, float)
    va = np.nan_to_num(va, nan=0.0, posinf=1e9, neginf=0.0)
    vs = np.maximum(va, 0.0)
    sa = vs.sum()
    if sa <= z:
        return vs
    u = np.sort(vs)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - z))[0][-1]
    tau = (cssv[rho] - z) / float(rho + 1)
    return np.maximum(vs - tau, 0.0)
# 1.
def nnls_projected_grad(A, b, iters=200, lr=None, l1_cap=1.0, tol=1e-6):
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    K = A.shape[1]
    alpha = np.zeros(K, float)
    At = A.T
    # step suggerito: 1/L con L = ||A||_2^2 (stima spettrale cheap con norma Frobenius)
    if lr is None:
        L = np.linalg.norm(A, ord='fro')**2
        lr = 1.0/max(L, 1e-9)
    prev = alpha.copy()
    for _ in range(iters):
        grad = At @ (A @ alpha - b)
        alpha = _project_to_simplex(alpha - lr*grad, z=l1_cap)
        if np.max(np.abs(alpha - prev)) < tol:
            break
        prev = alpha.copy()
    return alpha
# 2.
def recon_error(A, b, alpha):
    r = b - A @ alpha
    return float(np.linalg.norm(r, 2))  # L2
# 3. UNSUP: conf score and micro-probe for logging and verification of unsupervised learning success
def clamp01(x): 
    return float(np.clip(x, 0.0, 1.0)) # clipping
# 4. confidence probability => higher means good confidence: correct unsupervised learning, smaller means bad confidence
def conf_unsup_score(err, top_alpha, good_coverage):
    # conf_unsup = (1 - err) * top_alpha * good_coverage
    return clamp01((1.0 - float(err)) * float(top_alpha) * (1.0 if good_coverage else 0.0))
# 5.
def probe_boost_candidates(cands, base_rates, K=3, dur_ms=120, boost_frac=0.12):
    """
    Micro-trial: re-inietta lo stimolo K volte aumentando i canali candidati di +1015%.
    Ritorna (ok, pmr_series, gap_series, z_series_dict)
    - ok=True se PMR/gap e z dei candidati crescono "abbastanza" in maniera consistente.
    Nota: non tocca STDP (� già frozen nel TEST).
    """
    dur = dur_ms * b.ms
    pmr_list, gap_list = [], []
    z_series = {c: [] for c in cands}

    for _ in range(K):
        boosted = np.array(base_rates, dtype=float)
        for c in cands:
            boosted[c] *= (1.0 + boost_frac)

        # run micro-trial: conteggio differenziale
        prev = spike_mon.count[:].copy()
        if USE_GDI:
            set_stimulus_vector(boosted, include_unknown=False)
        else:
            set_stimulus_vect_norm(boosted, total_rate=None, include_unknown=False)
        net.run(dur)
        diff = (spike_mon.count[:] - prev).astype(float)

        # metriche
        s = diff.copy(); s[unknown_id] = -1e9
        top = float(s.max()); order = np.argsort(s)[::-1]
        second = float(s[order[1]]) if len(order) > 1 else 0.0
        E = float(np.sum(s[:unknown_id])) if np.isfinite(s[:unknown_id]).any() else 0.0
        pmr = (top / (E + 1e-9)) if E > 0 else 0.0
        gap = (top - second) / (top + 1e-9) if top > 0 else 0.0
        pmr_list.append(pmr)
        gap_list.append(gap)

        # z-score relativo alle aspettative positive (scala test)
        pos_expect_test = np.maximum(ema_pos_m1 * float(test_duration / training_duration), 1.0)
        z = s[:unknown_id] / np.maximum(pos_expect_test, 1.0)
        for c in cands:
            z_series[c].append(float(z[c]))

    # criterio: crescita sistematica - (largo ma utile):
    # - PMR e gap non devono peggiorare in media
    # - per ogni candidato almeno 2/3 degli step mostrano crescita z rispetto al primo
    pmr_ok = (np.mean(np.diff(pmr_list)) >= -0.01)      # non decresce
    gap_ok = (np.mean(np.diff(gap_list)) >= -0.01)
    z_ok_all = True
    for c in cands:
        base = z_series[c][0]
        ups = sum(1 for zc in z_series[c][1:] if zc >= base + 1e-3)
        need = max(1, math.ceil((K-1)*2/3))
        z_ok_all = z_ok_all and (ups >= need) # at least 2 on 3

    return (pmr_ok and gap_ok and z_ok_all), pmr_list, gap_list, z_series

# nnls management to classify mix correctly with 4-mix and 5-mix
def decode_by_nnls(
    scores, P_pos, P_cop, is_mixture_like,
    z_scores,                          # array z per-gusto (C,)
    frac_thr=0.10,                     # soglia frazionaria per i pesi -> between 0.10 and 0.15
    z_min_guard=0.01,                  # guardia minima su z
    allow_k4=True,                     # flag per quadruple
    allow_k5=True,                     # flag per quintuple
    gap=None, gap_thr=None,            # opzionale (se già calcolati)
    pmr=None, pmr_thr=None,            # opzionale
    abs_floor=None,                    # None, float o array per-gusto
    nnls_iters=250, nnls_lr=None, l1_cap=1.0
):
    # blend fisso
    alpha = 0.60 if is_mixture_like else 0.20
    P = alpha * P_cop + (1.0 - alpha) * P_pos

    b = scores.astype(float).copy()
    b_sum = float(b.sum())
    if b_sum > 0:
        b_n = b / b_sum
        P_n = P / np.maximum(P.sum(axis=0, keepdims=True), 1e-9)  # colonne ~1
    else:
        b_n, P_n = b, P

    # NNLS
    ws = nnls_projected_grad(P_n, b_n, iters=nnls_iters, lr=nnls_lr, l1_cap=l1_cap)
    err   = recon_error(P_n, b_n, ws)
    cover = float((P_n @ ws).sum())

    # stima cardinalità (HHI su b_n)
    p = b_n / (b_n.sum() + 1e-9)
    HHI = float((p**2).sum())
    k_est = int(np.clip(round(1.0 / max(HHI, 1e-9)), 1, scores.size))

    # soglie frazionarie: 0.10 base, 0.08 se target quintuple
    ws_sum = float(ws.sum())
    thr_rel_base = frac_thr * ws_sum
    thr_rel_k5   = min(frac_thr, 0.08) * ws_sum

    # conta coefficienti “significativi”
    k_abs = int(np.sum(ws >= max(1e-12, thr_rel_base)))
    k_abs5 = int(np.sum(ws >= max(1e-12, thr_rel_k5)))

    # criteria acceptation
    basic_ok = (err <= 0.40 and cover >= 0.60 and k_abs in (1,2,3,4,5))

    k4_ok = False
    if allow_k4 and (k_abs == 4 or (k_est >= 4 and int(np.sum(ws >= thr_rel_base)) >= 4)):
        # più pulito e distribuzione compatibile
        clean_ok   = (err <= 0.32 and cover >= 0.70)
        shape_ok   = (k_est >= 4)
        pmr_gap_ok = True
        if (gap is not None and gap_thr is not None) or (pmr is not None and pmr_thr is not None):
            pmr_gap_ok = ((gap is not None and gap_thr is not None and gap >= 0.95*gap_thr) or
                          (pmr is not None and pmr_thr is not None and pmr >= 1.05*pmr_thr))
        k4_ok = clean_ok and shape_ok and pmr_gap_ok

    k5_ok = False
    if allow_k5 and (k_abs5 == 5 or (k_est >= 5 and int(np.sum(ws >= thr_rel_k5)) >= 5)):
        # condizioni ancora più severe
        clean_ok   = (err <= 0.28 and cover >= 0.75)
        shape_ok   = (k_est >= 5)  # lo spettro deve "dire" 5 per classificare correttamente 5-mix
        pmr_gap_ok = True
        if (gap is not None and gap_thr is not None) or (pmr is not None and pmr_thr is not None):
            pmr_gap_ok = ((gap is not None and gap_thr is not None and gap >= 1.00*gap_thr) or
                          (pmr is not None and pmr_thr is not None and pmr >= 1.10*pmr_thr))

        # z-guard: z >= 0 per tutti; z >= z_min solo per i 3 più pesanti
        sel5 = (ws >= max(1e-12, thr_rel_k5))
        if np.count_nonzero(sel5) >= 5:
            idxs = np.where(sel5)[0]
            # ordina per peso decrescente e prendi i top-3
            top3 = idxs[np.argsort(ws[idxs])[::-1][:3]]
            z_all_nonneg = bool(np.all(z_scores[idxs] >= 0.0))
            z_top3_ok    = bool(np.all(z_scores[top3] >= z_min_guard))
        else:
            z_all_nonneg = False
            z_top3_ok    = False

        # eventual fallback absolute floor
        if isinstance(abs_floor, float):
            floor_ok = bool(np.all(scores[ws >= thr_rel_k5] >= abs_floor))
        elif isinstance(abs_floor, np.ndarray):
            sel = (ws >= thr_rel_k5)
            floor_ok = bool(np.all(scores[sel] >= abs_floor[sel]))
        else:
            floor_ok = True

        k5_ok = clean_ok and shape_ok and pmr_gap_ok and z_all_nonneg and z_top3_ok and floor_ok

    if not (basic_ok or k4_ok or k5_ok):
        # salviamo info di debug con la soglia effettiva che abbiamo usato
        return None, dict(err=float(err), cover=cover,
                          k_abs=int(k_abs5 if k5_ok else k_abs),
                          k_est=k_est,
                          thr_rel=float(thr_rel_k5 if k5_ok else thr_rel_base),
                          ws=ws)
    
    # selezione candidati
    thr_rel_eff = float(thr_rel_k5 if k5_ok else thr_rel_base)
    cand = [ids for ids, wi in enumerate(ws) if (wi >= thr_rel_eff) and (z_scores[ids] >= (0.0 if k5_ok else z_min_guard))]
    
    # una guardia addizionale se 4 o 5 candidati (es. min score assoluto)
    if (len(cand) >= 4) and isinstance(abs_floor, float):
        cand = [ids for ids in cand if scores[ids] >= abs_floor]
    elif (len(cand) >= 4) and isinstance(abs_floor, np.ndarray):
        cand = [ids for ids in cand if scores[ids] >= abs_floor[ids]]

    # se per rumore cand > 5, tieni i migliori 5 per peso ws -> si ordina per score di ognuno
    if len(cand) > 5:
        cand = sorted(cand, key=lambda ids: ws[ids], reverse=True)[:5]

    return cand, dict(err=float(err), cover=cover,
                      k_abs=int(len(cand)), k_est=k_est,
                      thr_rel=float(thr_rel_eff), ws=ws)

# normalizzazione L1 per popolazioni di neuroni e non più solo singoli
def col_norm_pop(target=None, mode="l1", temperature=1.0, diag_bias_gamma=1.20,
                 floor=0.0, allow_upscale=True, scale_max=1.15):
    # Normalizza i pesi in ingresso per ogni neurone post-sinaptico (j),
    # con un lieve bias alla diagonale (pre=i corrispondente alla sua classe).
    Wi = np.asarray(S.i[:], int)
    Wj = np.asarray(S.j[:], int)
    W  = S.w[:]
    C_known = unknown_id

    # target L1 per colonna: se non dato, usa media iniziale * fanin_known
    fanin = C_known
    if target is None:
        init_mean = float(np.mean(W)) if W.size else 0.5
        target = float(np.clip(init_mean*fanin, 0.5*fanin*0.2, 1.5*fanin*0.8))

    for jx in range(TOTAL_OUT):
        mask_col = (Wj == jx)
        if not np.any(mask_col): 
            continue
        wcol = W[mask_col].copy()
        icol = Wi[mask_col]

        # bias diagonale: se il neurone j appartiene alla pop di classe q → boosta i==q
        q = (jx // NEURONS_PER_TASTE)  # classe della pop di j
        wcol[(icol == q)] *= diag_bias_gamma

        if floor > 0:
            wcol = np.maximum(wcol, floor)

        if mode == "l1":
            sz = np.sum(wcol)
            if sz > 0:
                scale = target / sz
                if not allow_upscale:
                    scale = min(1.0, scale)
                scale = np.clip(scale, 1.0/scale_max, scale_max)
                wcol *= scale
        else:  # softmax
            xd = wcol / max(1e-9, temperature)
            xd = np.exp(xd - np.max(xd))
            xd /= np.sum(xd) + 1e-9
            wcol = xd * target

        # rimuovi il bias prima di scrivere
        wcol[(icol == q)] /= diag_bias_gamma
        W[mask_col] = wcol

    S.w[:] = W

# Restituisce lo slice di output per il gusto t (0..num_tastes-1) cosi evito di inserire UNKNOWN
def taste_slice(tx):
    start = tx * NEURONS_PER_TASTE
    return slice(start, start + NEURONS_PER_TASTE)

# True se ja ricade nella popolazione UNKNOWN
def is_unknown_output_index(ja):
    sl = taste_slice(unknown_id)
    return (ja >= sl.start and ja < sl.stop)

# manager to decide whether say UNKNOWN or not
def set_unknown_gate(pmr, gap, H, PMR_thr, gap_thr, H_thr):
    """
    Accende l'UNKNOWN se l'input è "diffuso":
    - PMR basso   (<= PMR_thr)
    - gap basso   (<= gap_thr)
    - entropia alta (>= H_thr)
    """
    triggers = ((pmr <= PMR_thr) and (gap <= gap_thr) and (H >= H_thr))
    S_unk.gain_unk = 1.0 if triggers else 0.0

# collapse all the population to its relevant taste to reward it
def population_scores_from_counts(counts):
    # counts: lunghezza TOTAL_OUT (per-neurone)
    if NEURONS_PER_TASTE == 1:
        return counts.astype(float)
    scores = np.zeros(num_tastes, dtype=float)
    for tx in range(num_tastes):
        sl = taste_slice(tx)
        # scegli aggregazione: mean/sum/max
        #scores[tx] = float(np.mean(counts[sl]))
        scores[tx] = float(np.sum(counts[sl]))
    return scores

# population neurons logging for each taste
def log_population_stats(counts, step=None, label=""):
    # pesi medi per gusto (solo S pg->taste_neurons)
    scores = population_scores_from_counts(np.asarray(counts, dtype=float))
    ws = np.asarray(S.w[:], float)
    js = np.asarray(S.j[:], int)
    mean_w = []
    for tx in range(num_tastes):
        sl = taste_slice(tx)
        mask = np.isin(js, np.arange(sl.start, sl.stop))
        mw = float(np.nanmean(ws[mask])) if np.any(mask) else float('nan')
        mean_w.append(mw)
    h = f"[Trial {step}] " if step is not None else ""
    print(f"{h}{label} | scores={np.round(scores, 2).tolist()} | mean_w={[None if not np.isfinite(xd) else round(xd,4) for xd in mean_w]}")

# print population stats
def pop_stats(p, q=None):
    idx = diag_indices_for_taste(p, q)  # p->pop(q) o p->pop(p) se q=None
    if idx.size == 0:
        return None
    wd = np.asarray(S.w[idx], float)
    return dict(N=int(idx.size),
                mean=float(np.mean(wd)),
                std=float(np.std(wd)),
                min=float(np.min(wd)),
                max=float(np.max(wd)))

# With populations have to reward all synapses i=t → j ∈ slice(t):
def diag_indices_for_taste(p, q=None):
    i_all = np.asarray(S.i[:], int)
    j_all = np.asarray(S.j[:], int)
    if q is None:
        # diagonal p→pop(q): i=p, j ∈ slice(p)
        sl = taste_slice(p)
        return np.where((i_all == p) & (j_all >= sl.start) & (j_all < sl.stop))[0]
    else:
        # off-diagonal p→pop(q): i=p, j ∈ slice(q)
        slq = taste_slice(q)
        return np.where((i_all == p) & (j_all >= slq.start) & (j_all < slq.stop))[0]

# Somma a scores un contributo 'diff_extra' che può essere (C,), (K,C), (C,K) o (K*C,).
# Comprimi su K (media) se necessario
# Applica solo alle C classi note (0..unknown_id-1), NON al posto UNKNOWN
# Currently 5 neurons x 8 tastes = 40 outcomes => need to collapse every class population in its respective class to avoid mismatches
def add_extra_scores(scores, diff_extra, unknown_id, reduce="sum"):
    """
    Collassa i contributi per-neurone (diff_extra) in contributi per-classe
    e li somma a scores (solo classi note: [0..unknown_id-1]).

    scores:      shape (C_all,)  -> include lo slot UNKNOWN in coda
    diff_extra:  shape (K*C_all,) o (K*C_known,) o già (C_all,)
    unknown_id:  indice della classe UNKNOWN ( = C_known )

    reduce: "sum" (default) o "mean" per come aggregare sui K neuroni.
    """
    vs = np.asarray(diff_extra, dtype=float).ravel()
    C_all   = scores.size          # es. 8 (7 gusti + UNKNOWN)
    C_known = unknown_id           # es. 7

    def _reduce(blocks):
        return blocks.sum(axis=1) if reduce == "sum" else blocks.mean(axis=1)

    per_class = None

    # 1) Caso naturale: include anche UNKNOWN -> v.size = K * C_all
    if vs.size % C_all == 0 and vs.size >= C_all:
        Ka = vs.size // C_all
        per_class = _reduce(vs.reshape(C_all, Ka))

    # 2) Caso: solo classi note -> v.size = K * C_known
    elif vs.size % C_known == 0 and vs.size >= C_known:
        Ka = vs.size // C_known
        tmp = _reduce(vs.reshape(C_known, Ka))
        # pad per avere (C_all,) e lasciare 0 su UNKNOWN
        per_class = np.zeros(C_all, dtype=float)
        per_class[:C_known] = tmp

    # 3) Caso: è già per-classe (o fallback)
    else:
        per_class = np.zeros(C_all, dtype=float)
        nw = min(vs.size, C_all)
        per_class[:nw] = vs[:nw]

    # somma solo sulle classi note
    scores[:unknown_id] += per_class[:unknown_id]
    return scores

# 1. Initialize the simulation
b.start_scope()
b.defaultclock.dt = 0.1 * b.ms  # high temporal precision
print("\n- ARTIFICIAL TONGUE's SNN with Triplet STDP and STP (Tsodyks-Markram), conductance-base LIF neurons, ELIGIBILITY TRACE, INTRINSIC HOMEOSTASIS and LATERAL INHIBITION: WTA (Winner-Take-All) -")

# 2. Define tastes
num_tastes = 8  # SWEET, BITTER, SALTY, SOUR, UMAMI, FATTY, SPICY, UNKNOWN
# Redundant POPULATION neurons mode (set >1 to enable)
NEURONS_PER_TASTE = 5  # 1 = only one neuron for each taste; >1 = different neuron populations for each taste
TOTAL_OUT = num_tastes * NEURONS_PER_TASTE

taste_labels = ["SWEET", "BITTER", "SALTY", "SOUR", "UMAMI",
                 "FATTY", "SPICY", "UNKNOWN"]
taste_reactions = {
    0: "Ouh... yummy!",
    1: "So bitter!",
    2: "Need water... now!",
    3: "Mehhh!",
    4: "So delicious!",
    5: "Oh, I'm a big fat boy!",
    6: "I'm a blazing dragon!",
    7: "WTF!"
}
# map from index to label
taste_map = {idx: lbl for idx, lbl in enumerate(taste_labels)}

# Separate normal and special taste
normal_tastes = {idx: lbl for idx, lbl in taste_map.items() if lbl != "UNKNOWN"} # without UNKNOWN
special_taste = {idx: lbl for idx, lbl in taste_map.items() if lbl == "UNKNOWN"} # only UNKNOWN

# Print all tastes
print("\nAll tastes available:")
for idx, lbl in normal_tastes.items():
    print(f"{idx}: {lbl}")

print("\nSpecial taste:")
for idx, lbl in special_taste.items():
    print(f"{idx}: {lbl}")
unknown_id = num_tastes-1

# 3. Simulation global parameters
# LIF (conductance-based) parameters:
C                    = 200 * b.pF       # membrane potential
gL                   =  10 * b.nS       # leak conductance
EL                   = -70 * b.mV       # leak potential
Vth                  = -52 * b.mV       # potential threshold
Vreset               = -60 * b.mV       # potential reset
# Synaptic reversal & time constants
Ee                   = 0*b.mV           # excitement synapses potential
Ei                   = -80*b.mV         # inhibition synapses potential
taue                 = 10*b.ms          # time decay for excitement
taui                 = 10*b.ms          # time decay for inhibition
# Size of conductance kick per spike (scaling)
g_step_exc           = 3.5 * b.nS       # excitation from inputs
g_step_bg            = 0.2 * b.nS       # tiny background excitation noise
g_step_inh_local     = 1.5 * b.nS       # lateral inhibition strength

# Global Divisive Inhibition (GDI) parameters -> we need to balance very well this improvement
# because it will break all the mixes otherwise
tau_gdi              = 40 * b.ms        # global pool temporal window
k_e2g_ff             = 0.05             # feed-forward contribute (Poisson input spikes) -> GDI
k_e2g_fb             = 0.010            # feedback contribute (output neuron spikes) -> GDI
gamma_gdi_0          = 0.30             # scaled reward for (dimensionless)
gdi_reset_each_trial = True             # managing carry-over thorugh trials

# Neuromodulators (DA Dopamine: reward, 5-HT Serotonine: aversion/fear, NE Noradrenaline: arousal/attention)
# Dopamine (DA) - reward gain
tau_DA               = 300 * b.ms       # fast decay: short gratification
tau_HT               = 2 * b.second     # slow decay: prudence, residual fear
da_gain              = 1.0              # how much DA expand the positive reinforcement
ht_gain              = 1.0              # how much 5-HT expand the punishment or how much it stops LTP
# Serotonine (5-HT) - aversive state on the entire circuit
k_theta_HT           = 2.0              # mV bias threshold per 5-HT unit
k_inh_HT             = 0.5              # WTA scaling per 5-HT unit
# aversive stochastic events
da_pulse_reward      = 1.0              # burst DA if classification is correct -> reward
ht_pulse_aversion    = 1.0              # burst 5-HT when imminent event is aversive
ht_pulse_fp          = 0.5              # burst 5-HT extra if there are a lot of strong FP
p_aversion_base      = 0.02             # probabilità base di evento avversivo quando SPICY è presente => mettere 0.02 per spegnere o quasi
p_aversion_slope     = 0.05             # quanto la probabilità cresce con l'intensità relativa => mettere 0.05 per spegnere o quasi
p_aversion_cap       = 0.60             # massimo assoluto (clamp)
da_penalty_avers     = 0.5              # frazione con cui attenuare il reward DA (0.5 = dimezza)
thr_spice_kick       = 0.04             # quanto alzare la soglia di tolleranza se l'evento accade (adattamento rapido)
# Noradrenaline (NE) -> arousal/attention
tau_NE               = 500 * b.ms       # fast decay
k_ex_NE              = 0.5              # reward scaling
k_inh_NE             = 0.2              # shrinks WTA for SNA
k_noise_NE           = 0.5              # low environment noise
ne_gain_r            = 0.3              # scaling reinforcement r entity
ne_pulse_amb         = 0.8              # burst on FP
# Histamine (HI) -> novelty/exploration (slower than NE)
tau_HI               = 1500 * b.ms      # slow decay
k_ex_HI              = 0.30             # gain on synapses
k_inh_HI             = -0.15            # HI decrease WTA
k_theta_HI           = -1.0             # mV threshold bias per HI unit
k_noise_HI           = 0.30             # more HI -> more environment noise
hi_gain_r            = 0.15             # reinforcement scaling with HI
hi_pulse_nov         = 0.8              # burst on ambiguity
hi_pulse_miss        = 1.0              # burst on error
# Acetylcholine (ACh) -> contextual plasticity / different attention on train and test
tau_ACH              = 700 * b.ms       # decay for ACh
ach_train_level      = 0.8              # high ACh during training
ach_test_level       = 0.40             # low ACh during test
k_ex_ACH             = 0.25             # exictement gain with ACh
ach_plasticity_gain  = 0.40             # ACh -> reward effect on -> w
k_noise_ACH          = 0.25             # ACh noise environment reduction
# GABA -> global inhibition/stability
tau_GABA             = 800 * b.ms       # decay for GABA
k_inh_GABA           = 0.60             # WTA scaling with GABA
gaba_pulse_stabilize = 0.8              # burst when activity is too much
gaba_active_neurons  = 4                # if > k neurons are activated in the same time � stability
gaba_total_spikes    = 120              # if total spikes per trial overcome this threshold � stability

# Dopamine delay dynamics + tonic tail (phasic vs tonic)
tau_DA_phasic        = 300 * b.ms
tau_DA_tonic         = 2 * b.second
dopamine_latency     = 150 * b.ms       # little delay before weight update -> more biologically plausible
k_tonic_DA           = 0.35             # how much the tonic tail contributes during plasticity
da_tonic_tail        = 0.25             # when the reward is gained => how big is the tonic tail quote
# state -> tonic bias (hungry/thirsty increase DA_tonic baseline)
k_hun_tonic          = 0.20
k_h2o_tonic          = 0.20

# Intrinsic homeostasis adapative threshold parameters
target_rate          = 50 * b.Hz        # reference firing per neuron (tu 40-80 Hz rule)
tau_rate             = 200 * b.ms       # extimate window for rating -> spikes LPF
tau_theta            = 1 * b.second     # threshold adaptive speed
theta_init           = 0.0 *b.mV        # starting theta threshold for homeostasis
rho_target = target_rate * tau_rate     # dimensionless (Hz*s)

# Decoder threshold parameters
k_sigma              = 0.9              # � if it is too weak
q_neg                = 0.99             # negative quantile

# Multi-label RL + EMA decoder
ema_lambda            = 0.10            # 0 < � d 1
tp_gate_ratio         = 0.28            # threshold to reward winner classes
fp_gate_warmup_steps  = 120             # delay punitions to loser classes if EMA didn't stabilize them yet
decoder_adapt_on_test = False           # updating decoder EMA in test phase
ema_factor            = 0.40            # EMA factor to punish more easy samples
use_rel_gate_in_test  = False           # using relative gates for mixtures and not only absolute gates
rel_gate_ratio_test   = 0.06            # second > 45 % rel_gate
mixture_thr_relax     = 0.30            # e 50% of threshold per-class
z_rel_min             = 0.005           # z margin threshold to let enter taste in relative gate
z_min_base            = 0.05            # prima 0.20
z_min_mix             = 0.0             # prima 0.10
# dynamic absolute thresholds for spikes counting  
rel_cap_abs           = 10.0            # absolute value for spikes
dyn_abs_min_frac      = 0.14            # helper for weak co-tastes -> it needs at least 30% of positive expected
# boosting parameters to push more weak examples
norm_rel_ratio_test   = 0.020           # winners with z_i >= 15% normalized top
min_norm_abs_spikes   = 2               # at least one real spike
eps_ema               = 1e-3            # epsilon for EMA decoder
mix_abs_pos_frac      = 0.06            # positive expected fraction
# metaplasticity -> STDP reward adapted to the historical of how often the reward is inside the hedonic window
meta_min              = 0.3             # lower range STDP scaling
meta_max              = 1.65            # higher range STDP scaling
meta_lambda           = 0.05            # EMA velocity
gwin_ema        = np.zeros(unknown_id)  # historical for every class

# Off-diag hyperparameters
beta                  = 0.03             # learning rate for negative reward
beta_offdiag          = 0.5 * beta       # off-diag parameter
use_offdiag_dopamine  = True             # quick toggle to activate/deactivate reward for off-diagonals
beta_offdiag_map      = np.ones(unknown_id)
beta_offdiag_map[6]   = 1.80             # SPICY: punisci quando “ruba” fuori classe
beta_offdiag_map[1]   = 1.25             # BITTER (rimetti la lieve stretta)
beta_offdiag_map[2]   = 1.25             # SALTY (lieve)
beta_offdiag_map[3]   = 1.60             # SOUR
beta_offdiag_map[5]   = 1.20             # FATTY (lieve)

# Normalization per-column (synaptic scaling in input)
use_col_norm          = True             # on the normalization
col_norm_mode         = "l1"             # "l1" (sum=target) or "softmax" -> synaptic scaling that mantains input scale per post neuron to avoid unfair competition
col_norm_every        = 1                # execute norm every N trial
col_norm_temp         = 1.0              # temperature softmax (if mode="softmax")
col_norm_target       = None             # if None, calculating the target at the beginning of the trial
diag_bias_gamma       = 1.20             # >1.0 = light bias to the diagonal weight before normalization
col_floor             = 0.02             # floor (0 or light epsilon) before norm
col_allow_upscale     = True             # light up-scaling
col_upscale_slack     = 0.95             # if L1 < 90% target � boost
col_scale_max         = 1.10             # max factor per step

# SPICY dynamic tolerance / aversion dynamics
spicy_id              = 6                # spicy taste is the sixth one
fatty_id              = 5                # fatty taste is the fifth one
thr0_spice_var        = 0.36             # baseline aversive threshold -> driven unit
tau_thr_spice         = 30 * b.second    # adapting threshold -> slow
tau_sd_spice          = 80 * b.ms        # spicy intensity integration
tau_a_spice           = 200 * b.ms       # dynamic aversion
k_spike_spice         = 0.010            # spike contribution pre SPICY->drive
k_a_spice             = 1.0              # aversion reward
#k_hab_spice          = 0.0015           # upgrade threshold � with adapting to aversion
eta_da_spice          = 2.0              # multiplier DA for adapting
#k_sens_spice         = 0.001            # sensitization if above threshold but without reward -> just an adapting on the previous threshold: now higher
reinforce_dur         = 150 * b.ms       # short window to push DA gate on SPICY

# Hedonic window for all the tastes (SWEET, SOUR ecc...) -> one taste is rewarding ONLY if his spikes fire during this period
tau_drive_win         = 50 * b.ms        # intensity/taste integration
tau_av_win            = 200 * b.ms       # aversion/sub-threshold integration
tau_thr_win           = 30 * b.second    # thresholds adapting
eta_da_win            = 2.0              # rewarding on the habit of the Hedonic window
k_spike_drive         = 0.015            # driving kick on each input spike

# Hedonic gating for DA state-dependente (fallback included) -> if a taste is recognized inside the hedonic window => full DA, otherwise if it is ricognized but it's not in the window => less DA
use_hedonic_da       = True
hed_fallback         = 0.40             # minimum reinforcement if prediction is not inside the hedonic window
hed_gate_k           = 2.0              # gating convergence: �k = more aggressive
hed_min              = 0.10             # lower fallback clamp
hed_max              = 0.95             # higher fallback clamp
k_hun_fb             = 0.35             # hungry -> � fallback for SWEET/UMAMI/FATTY
k_sat_fb             = 0.25             # satiety -> � fallback for SWEET/UMAMI/FATTY
k_h2o_fb             = 0.25             # thirsty -> � fallback for SALTY/SPICY
k_bitter_sat         = 0.10             # bitter -> � light fallback with satiety
# energy needs requirements mapping
hunger_idxs          = [0, 4, 5]        # SWEET, UMAMI, FATTY
water_idxs           = [2, 6]           # SALTY, SPICY

# Spike-Timing Dependent Plasticity STDP and environment parameters
tau                  = 30 * b.ms        # STDP time constant
Te                   = 50 * b.ms        # eligibility trace decay time constant
A_plus               = 0.01             # dimensionless
A_minus              = -0.0075          # dimensionless
alpha                = 0.1              # learning rate for positive reward
noise_mu             = 5                # noise mu constant
noise_sigma          = 0.8              # noise sigma constant
training_duration    = 1000 * b.ms      # stimulus duration
test_duration        = 1000 * b.ms      # test verification duration
pause_duration       = 100 * b.ms       # pause for eligibility decay
n_repeats            = 10               # repetitions per taste
progress_bar_len     = 30               # characters
weight_monitors      = []               # list for weights to monitor
threshold_ratio      = 0.40             # threshold for winner spiking neurons
min_spikes_for_known = 5                # minimum number of spikes for neuron, otherwise UNKNOWN
top2_margin_ratio    = 1.05             # top/second >= 1.4 -> safe
weight_decay         = 1e-4             # weight decay for trial
verbose_rewards      = False            # dopamine reward logs
test_emotion_mode    = "off"            # to test with active neuromodulators

# Short-Term Plasticity STP (STF/STD) (Tsodyks-Markram) parameters
use_stp              = True             # toggle to set STP ON/OFF
# default parameters for STP applied to pg -> taste neurons (avoiding collapse)
stp_u0               = 0.08             # baseline u (utilization)
stp_uinc             = 0.05             # increasing per-spike (facilitation)
stp_tau_rec          = 260 * b.ms       # recovery (STD)
stp_tau_facil        = 800 * b.ms       # facilitation decay (STF)
stp_r_ref            = 120.0            # reference Hz to calibrate gain for STP
stp_warmup_trials    = 150              # important initial warmup to avoid DA suppression in the beginning

# Dynamic OVERSAMPLING -> TRAIN ONLY
USE_DYNAMIC_OVERSAMPLING = False # switch to TRUE to apply it
CLASS_BOOST = np.ones(unknown_id, dtype=float)
# More gain to weak classes
for ca in [0, 2, 4, 5, 6]:  # SWEET,SALTY,SOUR,UMAMI,FATTY
    CLASS_BOOST[ca]  = 1.40
BOOST_LAM            = 0.10             # EMA speed
BOOST_GAIN           = 0.75             # quanto “spinge” il need
BOOST_CAP            = (0.80, 1.30)     # clamp per gusto (min,max)
BOOST_APPLY_GUARD_STEPS = fp_gate_warmup_steps  # non applicare boost nei primissimi step (EMA ancora grezze)

# Biological per-class attention bias -> TRAIN ONLY (no oversampling)
USE_ATTENTIONAL_BIAS = True
ATTN_BIAS_GAIN_MV    = 0.8              # mV di bias max circa per unità "need_norm-1"
ATTN_BIAS_CAP_MV     = 2.0              # cap di sicurezza per gusto

# Bio-plausible early-stopping (DA gating + best snapshot)
ema_perf             = 0.0
perf_alpha           = 0.03             # smoothing dolce della performance
DA_GATE_JACC         = 0.50             # sotto questa Jaccard per trial: niente DA (no consolidamento)
VAL_EVERY            = 0                # 0 = solo proxy intra-trial; (>0 per mini-validation periodica)
# Early stopping setup
best_score           = -float("inf")
best_step            = -1
best_state           = None
patience             = 0
# Bio-plausible consolidation & early-stop params
USE_SLOW_CONSOLIDATION = True
PATIENCE_LIMIT       = 30              # switching it between 15 and 45
ETA_CONSOL           = 0.05            # slow capture rate (0..1) toward current fast weights
DA_THR_CONSOL        = 0.35            # require enough DA to consolidate into w_slow
BETA_MIX_TEST        = 0.10            # how much fast to keep when mixing slow→test (0..1)
USE_SOFT_PULL        = True            # softly drift toward best-state when stagnating
RHO_PULL             = 0.25            # 0..1, per-trial pull strength
PLASTICITY_DECAY     = 0.90            # multiplicative decay on S.stdp_on when stagnating
MAX_PLASTICITY_DECAYS= 5
W_EMA                = 12
# storico corto per stimare il rumore dell'EMA
ema_hist             = deque(maxlen=32)
ema_perf_sd          = 0.0             # aggiornata ad ogni trial
decays_done          = 0
# Reduce-on-plateau globals
PLAST_GLOBAL         = 1.0
PLAST_GLOBAL_FLOOR   = 0.15
REDUCE_FACTOR        = 0.80
COOLDOWN             = 8
PLATEAU_WINDOW       = 5               # ogni 5 trial senza migliorie posso decidere un decay per il plateau
cooldown_left        = 0

# Connectivity switch: "diagonal" | "dense"
connectivity_mode    = "dense"  # "dense" -> fully-connected | "diagonal" -> one to one

# Plasticity pull (in Tensorflow's ReduceLROnPlateau style)
def set_plasticity_scale(scale: float):
    s = float(np.clip(scale, 0.0, 1.0))
    # 1) gate sinaptico esplicito
    if 'stdp_on' in S.variables:
        S.stdp_on[:] = s
    # 2) scala globale del rinforzo
    global PLAST_GLOBAL
    PLAST_GLOBAL = s
    # 3) raffredda le tracce di eligibility già accumulate
    if 'elig' in S.variables:
        S.elig[:] *= (0.5 + 0.5*s)  # compressione dolce

def soft_pull_toward_best(best_sd, rho: float = 0.25):
    """
    Drift morbido dei pesi correnti verso il best snapshot (bio-plausibile “consolidation pull”).
    """
    if best_sd is None:
        return
    # soft blend solo sui pesi sinaptici veloci + theta
    if best_sd.get('w') is not None:
        S.w[:] = (1.0 - rho) * S.w[:] + rho * best_sd['w']
    if hasattr(taste_neurons, 'theta'):
        taste_neurons.theta[:] = (1.0 - 0.5*rho) * taste_neurons.theta[:] + (0.5*rho) * best_sd['theta']

# helper to define states fallback for hedonic window during DA reinforcement
def state_hed_fallback_vec(mod, base=hed_fallback):
    H = float(mod.HUN[0])
    S = float(mod.SAT[0])
    W = float(mod.H2O[0])
    fb = np.full(unknown_id, base, dtype=float)

    # hungry/satiety
    fb[hunger_idxs] += k_hun_fb*H - k_sat_fb*S
    # thirsty
    fb[water_idxs]  += k_h2o_fb*W
    # bitter: less hungry/satiety
    fb[1] -= k_bitter_sat * S

    return np.clip(fb, hed_min, hed_max)

# SPICY aversion helper
# probabilità di evento avversivo in base all'intensità relativa dello SPICY
def spicy_aversion_triggered(tn, mod, spicy_id,
                             p_base=0.10, slope=0.25, cap=0.60,
                             trait=0.5,
                             k_hun=-0.15, k_h2o=+0.20,
                             rng=None):
    """
    Ritorna (happened, p) dove p dipende da:
      - intensità relativa (excess)
      - tratto individuale 'trait' in [0,1]  -> scala p_base e cap
      - stato (HUN/H2O) via k_hun/k_h2o
    """
    rng = np.random.default_rng() if rng is None else rng

    sl_spice = taste_slice(spicy_id)
    drive = float(np.mean(tn.spice_drive[sl_spice]))
    cur_thr = float(np.mean(tn.thr_spice[sl_spice]))
    base_thr = float(np.mean(tn.thr0_spice[sl_spice]))
    thr = cur_thr if cur_thr > 1e-6 else base_thr
    excess = max(0.0, (drive - thr) / (thr + 1e-9))

    # modulazione di stato
    H = float(mod.HUN[0]); W = float(mod.H2O[0])
    p_bias = (1.0 + k_hun*H + k_h2o*W)
    p_bias = np.clip(p_bias, 0.6, 1.6)

    # modulazione per-individuo (trait)
    trait_scale = (0.8 + 0.4*float(trait))   # 0.8..1.2
    p_b = p_base * trait_scale
    c_b = cap     * trait_scale

    p = np.clip((p_b + slope*excess) * p_bias, 0.0, c_b)
    return (rng.random() < p), p

# helpers to define toggles per ablation
def set_ach(on=True):
    global ach_train_level, ach_plasticity_gain, ach_test_level, k_ex_ACH, k_noise_ACH
    if on:
        ach_train_level, ach_plasticity_gain, ach_test_level = 0.8, 0.40, 0.10
        k_ex_ACH, k_noise_ACH = 0.25, 0.25
    else:
        ach_train_level, ach_plasticity_gain, ach_test_level = 0.0, 0.0, 0.0
        k_ex_ACH, k_noise_ACH = 0.0, 0.0

def set_gaba(on=True):
    global k_inh_GABA, gaba_pulse_stabilize, gaba_active_neurons, gaba_total_spikes
    if on:
        k_inh_GABA, gaba_pulse_stabilize = 0.60, 0.8
        gaba_active_neurons, gaba_total_spikes = 4, 120
    else:
        k_inh_GABA, gaba_pulse_stabilize = 0.0, 0.0
        gaba_active_neurons, gaba_total_spikes = 9999, 10**9 # trigger deactivation

# helper to index test tastes without noise in the stimuli list
def make_mix(ids, amp=250):
    vix= np.zeros(num_tastes)
    for idx in ids: vix[idx] = amp
    label = " + ".join([f"'{taste_map[idx]}'" for idx in ids])
    return vix, ids, f"TASTE: {label}"

# helper to index training tastes in the stimuli list
def noisy_mix(ids, amp=250, mu=noise_mu, sigma=noise_sigma):
    vix = np.clip(np.random.normal(mu, sigma, num_tastes), 0, None)
    for idx in ids:
        vix[idx] = amp
    label = " + ".join([f"'{taste_map[idx]}'" for idx in ids])
    return vix, ids, f"TASTE: {label} (train)"

# test generators OOD/NULL (expected = UNKNOWN)
def make_null(low=5, high=20):
    vix = np.random.randint(low, high+1, size=num_tastes).astype(float)
    vix[unknown_id] = 0.0
    return vix, [unknown_id], "NULL (only low background)"
    
# AUGMENTATION MIRATA PER MIX E COPPIE
'''Riassunto “a cosa serve cosa”:
jitter_active: simula variazioni realistiche di intensità dei canali attivi → meno overfitting a un’unica ampiezza.
channel_dropout: simula un sensore indebolito → robustezza ai co-gusti “ballerini”.
global_gain: simula individui/condizioni con tono generale più alto/basso → generalizzazione migliore.
augment_mix: pipeline unica che applica i tre step sopra ai mix.
make_asymmetric_pair: crea esempi con dominante + co-gusto (30–70%) → la rete impara a non “schiacciare” co-gusti veri ma deboli.
Sostituzioni noisy_mix → augment_mix: più variabilità utile nei dati (non solo più quantità).
Coppie asimmetriche ad alto SNR: “medicine” mirate per le coppie che confondono.
make_near_ood + ood_calibration più lunga: riduce FP su casi “quasi noti ma diffusi”, preservando il rifiuto UNKNOWN.
Curriculum soft: ordina gli stimoli per difficoltà, per evitare che i mix pesanti rompano precocemente le diagonali.
'''
"""Jitter sulle ampiezze dei soli canali ATTIVI (±frac).
Serve a rompere la regolarità dei mix e a migliorare la generalizzazione."""
def jitter_active(va, frac=0.20, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    ws = va.copy()
    act = np.where(ws > 0)[0] # indices of active channels consideration
    if act.size: # if there is at least one active channel
        ws[act] = np.maximum(0.0, ws[act] * (1.0 + rng.uniform(-frac, frac, size=act.size))) # jitter only active channels
    return ws

"""Coppia asimmetrica: un gusto dominante (amp_hi) + co-gusto al 30–70%.
Utile per sbloccare i co-gusti deboli e ridurre conflitti off-diagonali."""
# Nota: non esagerare con la frequenza di questi esempi, altrimenti la rete
# impara a ignorare i co-gusti veri ma deboli.
def make_asymmetric_pair(ids, jds, amp_hi=280, co_frac=(0.3, 0.7), rng=None):
    rng = np.random.default_rng() if rng is None else rng
    hi, lo = (ids, jds) if rng.random() < 0.5 else (jds, ids) # random order
    frac = rng.uniform(co_frac[0], co_frac[1])
    va = np.zeros(num_tastes)
    va[hi] = amp_hi # dominant frequency
    va[lo] = max(80, int(amp_hi * frac)) # at least 80 Hz for co-taste
    return va, [ids, jds], f"TASTE: '{taste_map[ids]}' + '{taste_map[jds]}' (asym)"

# Tripla asimmetrica: un gusto dominante (amp_hi) + 2 co-gusti al 25–55%.
# Utile per sbloccare i co-gusti deboli e ridurre conflitti off-diagonali.
# Nota: non esagerare con la frequenza di questi esempi, altrimenti la rete
# impara a ignorare i co-gusti veri ma deboli.
def make_asymmetric_triple(ia,ja,ka, amp_hi=300, co_lo=0.25, co_hi=0.55, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    dom = rng.choice([ia,ja,ka]) # dominant taste in the triple
    co  = [xd for xd in (ia,ja,ka) if xd!=dom]  # co-tastes in the triple
    va = np.zeros(num_tastes) # initialize rates vector
    va[dom] = amp_hi # dominant frequency 
    for c in co:
        va[c] = max(70, int(amp_hi * rng.uniform(co_lo, co_hi)))
    va = jitter_active(va, frac=rng.uniform(0.10,0.22), rng=rng)
    va = channel_dropout(va, p=rng.uniform(0.18,0.25), rng=rng)
    va = global_gain(va, lo=0.85, hi=1.18, rng=rng)
    return va, [ia,ja,ka], f"TASTE: '{taste_map[ia]}' + '{taste_map[ja]}' + '{taste_map[ka]}' (asym)"

# Dropout su canali attivi (p).
# Simula un sensore che a volte si inceppa o si indebolisce.
# Nota: non esagerare con la frequenza di questo step, altrimenti la rete
# impara a ignorare i co-gusti veri ma deboli.
def channel_dropout(va, p=0.15, min_keep=1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    ws = va.copy()
    act = [k for k in range(unknown_id) if ws[k] > 0]
    if len(act) <= min_keep:
        return ws
    kept = 0
    for k in act:
        if rng.random() < p and (len([xd for xd in act if ws[xd] > 0]) - kept) > min_keep:
            ws[k] *= rng.uniform(0.1, 0.5)  # lo smorzi, non lo spegni del tutto
        else:
            kept += 1
    return ws

# Gain globale 0.85–1.20× (variazione ‘individuo’/intensità ambiente).
def global_gain(va, lo=0.85, hi=1.20, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    ga = rng.uniform(lo, hi) # global gain factor => uniform scaling seed
    ws = va.copy()
    ws[:unknown_id] *= ga # gain only on normal tastes not including UNKNOWN
    return ws

# Pipeline: noisy_mix -> jitter -> dropout canale debole -> global gain.
# Incrementa varietà senza cambiare la semantica del mix.
def augment_mix(ids, amp=250, mu=noise_mu, sigma=noise_sigma, rng=None):
    rng = np.random.default_rng() if rng is None else rng # create own RNG seed if not provided
    vix = np.clip(np.random.normal(mu, sigma, num_tastes), 0, None)
    for idx in ids:
        vix[idx] = amp
    vix = jitter_active(vix, frac=0.08, rng=rng) # jitter active channels only 
    vix = channel_dropout(vix, p=0.08, rng=rng) # channel dropout on active channels only
    vix = global_gain(vix, lo=0.85, hi=1.20, rng=rng) # global gain scaling on normal tastes only
    label = " + ".join([f"'{taste_map[idx]}'" for idx in ids]) + " (aug)" # label with augment info
    return vix, ids, f"TASTE: {label}"

def make_ood_diffuse(low=20, high=80):
    # no dominant class
    vix = np.random.uniform(low, high, size=num_tastes)
    vix[unknown_id] = 0.0
    return vix, [unknown_id], "OOD (diffuse low-mid rates)"

def make_ood_many(low=60, high=130, k=5):
    # many average-low canals
    vix = np.zeros(num_tastes)
    picks = np.random.choice(np.arange(num_tastes-1), size=k, replace=False)
    vix[picks] = np.random.uniform(low, high, size=k)
    return vix, [unknown_id], f"OOD (many-{k} mid amps)"

def make_near_ood(lo=10, hi=40, bump=25):
    """Stimoli quasi-NULL con 1–2 canali appena sopra il fondo.
    Servono a ridurre FP su pattern ‘diffusi ma energetici’."""
    vix = np.random.uniform(lo, hi, size=num_tastes)
    vix[unknown_id] = 0.0
    k = np.random.choice(np.arange(num_tastes-1), size=np.random.randint(1,3), replace=False)
    vix[k] += bump
    return vix, [unknown_id], "NEAR-OOD (weak bumps)"

# functions set to save actual weight state of neurons and restore them after trial
# 1.
def snapshot_state():
    # helper per prelevare vettori se esistono
    def _get_arr(G, name):
        try:
            if name in G.variables:   # Brian2 Variables dict check
                return G.variables[name].get_value().copy()
        except Exception:
            pass
        return None

    return dict(
        # Sinapsi (STDP)
        w=S.w[:].copy(),
        # Tracce triplet (i tuoi nomi)
        x=_get_arr(S, 'x'),
        xbar=_get_arr(S, 'xbar'),
        y=_get_arr(S, 'y'),
        ybar=_get_arr(S, 'ybar'),
        # Tracce pair-based (fallback, se mai presenti)
        Apre=_get_arr(S, 'Apre'),
        Apost=_get_arr(S, 'Apost'),
        elig=_get_arr(S, 'elig'),

        # Neuroni di uscita
        theta=taste_neurons.theta[:].copy(),
        thr_hi=taste_neurons.thr_hi[:].copy(),
        thr_lo=taste_neurons.thr_lo[:].copy(),
        ge=taste_neurons.ge[:].copy(),
        gi=taste_neurons.gi[:].copy(),
        s=taste_neurons.s[:].copy(),
        wfast=taste_neurons.wfast[:].copy(),

        # STP
        x_stp=_get_arr(S, 'x_stp'),
        u=_get_arr(S, 'u'),

        # Neuromodulatori
        mod=np.array([
            float(mod.DA_f[0]), float(mod.DA_t[0]), float(mod.HT[0]),
            float(mod.NE[0]),   float(mod.HI[0]),   float(mod.ACH[0]),
            float(mod.GABA[0])
        ], dtype=float),

        # GDI
        gdi=float(gdi_pool.x[0]),
        gamma_gdi = S.variables['gamma_gdi'].get_value().item()
            if 'gamma_gdi' in S.variables else None)

# 2.
def restore_state(sd):
    # Sinapsi (STDP)
    if sd.get('w') is not None:
        S.w[:] = sd['w']

    # Preferisci le tue tracce triplet, altrimenti prova Apre/Apost (pair)
    if sd.get('x') is not None and 'x' in S.variables: S.x[:] = sd['x']
    if sd.get('xbar') is not None and 'xbar' in S.variables: S.xbar[:] = sd['xbar']
    if sd.get('y') is not None and 'y' in S.variables: S.y[:] = sd['y']
    if sd.get('ybar') is not None and 'ybar' in S.variables: S.ybar[:] = sd['ybar']

    if sd.get('Apre') is not None and 'Apre' in S.variables: S.Apre[:] = sd['Apre']
    if sd.get('Apost') is not None and 'Apost' in S.variables: S.Apost[:] = sd['Apost']

    if sd.get('elig') is not None and 'elig' in S.variables: S.elig[:] = sd['elig']

    # Output neurons
    taste_neurons.theta[:]  = sd['theta']
    taste_neurons.thr_hi[:] = sd['thr_hi']
    taste_neurons.thr_lo[:] = sd['thr_lo']
    taste_neurons.ge[:]     = sd['ge']
    taste_neurons.gi[:]     = sd['gi']
    taste_neurons.s[:]      = sd['s']
    taste_neurons.wfast[:]  = sd['wfast']

    # STP
    if sd.get('x_stp') is not None and 'x_stp' in S.variables:
        S.x_stp[:] = sd['x_stp']
    if sd.get('u') is not None and 'u' in S.variables:
        S.u[:] = sd['u']

    # Neuromodulators
    mod.DA_f[:] = sd['mod'][0]; mod.DA_t[:] = sd['mod'][1]
    mod.HT[:]   = sd['mod'][2]; mod.NE[:]   = sd['mod'][3]
    mod.HI[:]   = sd['mod'][4]; mod.ACH[:]  = sd['mod'][5]
    mod.GABA[:] = sd['mod'][6]

    # GDI
    gdi_pool.x[:] = sd['gdi']
    if sd.get('gamma_gdi') is not None and 'gamma_gdi' in S.variables:
       S.variables['gamma_gdi'].set_value(sd['gamma_gdi'])

# 1. Variante senza pesi
def restore_state_without(sd):
    """
    Ripristina lo stato della rete escludendo i pesi sinaptici S.w[:].
    Utile quando hai già deciso i pesi (es. mix soft di best_w e w_slow_best)
    ma vuoi ricaricare soglie, tracce, modulazione, GDI, ecc. dal checkpoint e
    non eliminare la bio-plausibilità dell'early stopping biologico implementato.
    """
    # Tracce STDP / eligibility
    if sd.get('x') is not None and 'x' in S.variables: S.x[:] = sd['x']
    if sd.get('xbar') is not None and 'xbar' in S.variables: S.xbar[:] = sd['xbar']
    if sd.get('y') is not None and 'y' in S.variables: S.y[:] = sd['y']
    if sd.get('ybar') is not None and 'ybar' in S.variables: S.ybar[:] = sd['ybar']

    if sd.get('Apre') is not None and 'Apre' in S.variables: S.Apre[:] = sd['Apre']
    if sd.get('Apost') is not None and 'Apost' in S.variables: S.Apost[:] = sd['Apost']

    if sd.get('elig') is not None and 'elig' in S.variables: S.elig[:] = sd['elig']

    # Neuroni di uscita
    if sd.get('theta')   is not None: taste_neurons.theta[:]  = sd['theta']
    if sd.get('thr_hi')  is not None: taste_neurons.thr_hi[:] = sd['thr_hi']
    if sd.get('thr_lo')  is not None: taste_neurons.thr_lo[:] = sd['thr_lo']
    if sd.get('ge')      is not None: taste_neurons.ge[:]     = sd['ge']
    if sd.get('gi')      is not None: taste_neurons.gi[:]     = sd['gi']
    if sd.get('s')       is not None: taste_neurons.s[:]      = sd['s']
    if sd.get('wfast')   is not None: taste_neurons.wfast[:]  = sd['wfast']

    # STP (se presenti)
    if sd.get('x_stp') is not None and 'x_stp' in S.variables:
        S.x_stp[:] = sd['x_stp']
    if sd.get('u') is not None and 'u' in S.variables:
        S.u[:] = sd['u']

    # Neuromodulatori
    if sd.get('mod') is not None and len(sd['mod']) >= 7:
        mod.DA_f[:] = sd['mod'][0]; mod.DA_t[:] = sd['mod'][1]
        mod.HT[:]   = sd['mod'][2]; mod.NE[:]   = sd['mod'][3]
        mod.HI[:]   = sd['mod'][4]; mod.ACH[:]  = sd['mod'][5]
        mod.GABA[:] = sd['mod'][6]

    # GDI
    if sd.get('gdi') is not None:
        gdi_pool.x[:] = sd['gdi']
    if sd.get('gamma_gdi') is not None and 'gamma_gdi' in S.variables:
        S.variables['gamma_gdi'].set_value(sd['gamma_gdi'])

# utilities to read and assign shared scalar variables
# 1.
def read_shared_scalar(S, name: str):
    if name not in S.variables:
        return None
    val = S.get_states([name], units=False, format='dict')[name]
    return float(np.asarray(val))  # gestisce 0-d array / numpy scalar
# 2.
def write_shared_scalar(S, name: str, value: float):
    setattr(S, name, value)  # se dimensionless; altrimenti value * unit

# OOD/NULL calibration: increase threshold on OOD queues
def ood_calibration(n_null=16, n_ood=32, dur=200*b.ms, gap=0*b.ms, thr_vec=None):
    saved_stdp = float(S.stdp_on[0]) # saved current plasticity state
    S.stdp_on[:] = 0.0
    saved_noise = pg_noise.rates # the same with noise
    pg_noise.rates = 0 * b.Hz
    saved_gamma = read_shared_scalar(S, 'gamma_gdi')

    # list of lists to collect all negative spikes for each class
    tmp_spikes = [[] for _ in range(num_tastes-1)] 
    pmr_list, h_list, gap_list = [], [], [] # lists to collect class metrics

    def _one_trial(vix):
        # no UNKNOWN
        set_stimulus_vector(vix, include_unknown=False)
        prev = spike_mon.count[:].copy() # saving previous trial spikes
        net.run(dur) # starting simulation
        diff = (spike_mon.count[:] - prev).astype(float) # misuring current spikes with substract previous to current ones
        diff_pop = population_scores_from_counts(diff) # population management
        # storing per-class
        for idx in range(num_tastes-1):
            tmp_spikes[idx].append(int(diff_pop[idx]))

        # feature open-set calculation
        s = diff_pop.copy()
        s[unknown_id] = -1e9
        order = np.argsort(s)[::-1] # desc sorting (last spike for first)
        top, second = float(s[order[0]]), float(s[order[1]]) if len(order) > 1 else 0.0
        E = float(np.sum(s[:unknown_id]))
        pmr = top / (E + 1e-9)
        gap_rel = (top - second) / (top + 1e-9) # numerical epsilon needs to be installed to avoid division per 0
        p = s[:unknown_id] / (E + 1e-9) # p is the spikes / their energy
        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum() # normalized distribution
        h = float(-(p*np.log(p)).sum()) # entropy
        pmr_list.append(pmr)
        h_list.append(h)
        gap_list.append(gap_rel)

    # NULL
    for _ in range(n_null):
        vix, _, _ = make_null()
        _one_trial(vix)
        if gap > 0*b.ms: net.run(gap)

    # OOD diffuse
    for _ in range(n_ood//2):
        vix, _, _ = make_ood_diffuse()
        _one_trial(vix)
        if gap > 0*b.ms: net.run(gap)

    # OOD many-k
    for _ in range(n_ood//2):
        vix, _, _ = make_ood_many(k=np.random.randint(4,6))
        _one_trial(vix)
        if gap > 0*b.ms: net.run(gap)

    # updating per-class thresholds against FP increasing negative quantile
    if thr_vec is not None:
        for idx in range(num_tastes-1):
            if tmp_spikes[idx]:
                neg_q = 0.9995 if idx not in (3,6) else 0.997
                thr_vec[idx] = max(
                    float(thr_vec[idx]),
                    float(np.quantile(tmp_spikes[idx], neg_q))
                )

    # open-set data-driven thresholds
    # (=> if during the test PMR/H/gap are inside the "negative typical part", refusing)
    PMR_thr_auto = float(np.quantile(pmr_list, 0.80))  # soglia più bassa -> meno trigger
    H_thr_auto   = float(np.quantile(h_list,  0.90))  # entropia deve essere davvero alta
    gap_thr_auto = float(np.quantile(gap_list, 0.20))  # gap molto basso per trigger
    # failsafe
    PMR_thr_auto *= 0.7
    H_thr_auto    = max(H_thr_auto, 1.5)  # alza il requisito di entropia
    gap_thr_auto *= 0.8

    # restore states after OOD
    if saved_gamma is not None:
        write_shared_scalar(S, 'gamma_gdi', saved_gamma)
    pg_noise.rates = saved_noise
    S.stdp_on[:] = saved_stdp

    # 0.997 per-class quantile on negatives OOD/NULL
    ood_q = np.array([
        (np.quantile(tmp_spikes[idx], 0.999) if len(tmp_spikes[idx]) else 0.0)
        for idx in range(num_tastes-1)
    ], dtype=float)

    return PMR_thr_auto, H_thr_auto, gap_thr_auto, ood_q

# more often a class is inside hedonic window (high gwin_ema) -> less push (conservative)
def meta_scale(idx):
    return np.clip(meta_min + (meta_max - meta_min) * (1.0 - float(gwin_ema[idx])), meta_min, meta_max)

# Individual initialization before TRAINING loop
INDIV_ID = 42  # seed for representing different people
profile = sample_individual(seed=INDIV_ID)
# dinamiche SPICY personali (se vuoi per-individuo)
k_hab_spice  = profile['k_hab_spice_hub']
k_sens_spice = profile['k_sens_spice_hub']
# 4. LIF conductance-based OUTPUT taste neurons with intrinsic homeostatis and dynamic SPICY aversion
taste_neurons = b.NeuronGroup(
    TOTAL_OUT, # by this way, i can use both methods: population mode and single neuron
    model='''
        dv/dt = (gL*(EL - v) + ge*(Ee - v) + gi*(Ei - v))/C : volt (unless refractory)
        dge/dt = -ge/taue : siemens
        dgi/dt = -gi/taui : siemens
        ds/dt = -s/tau_rate : 1
        dtheta/dt = homeo_on * (s - rho_target)/tau_theta * mV : volt
        dwfast/dt = -wfast/70/ms : volt
        theta_bias : volt
        homeo_on : 1

        # ---- GDI (pool linked + centering/saturation) ----
        gdi : 1 (linked)                 # global pool value
        gdi_center : 1                   # pool baseline
        gdi_half   : 1                   # half-saturation
        pos_gdi = 0.5*((gdi - gdi_center) + abs(gdi - gdi_center)) : 1
        gdi_eff = pos_gdi / (1.0 + pos_gdi/gdi_half) : 1

        # ----- Hedonic window for every different taste -----
        dtaste_drive/dt = -taste_drive/tau_drive_win : 1

        # High threshold of the Hedonic window
        dav_over/dt = (-av_over + 0.5*((taste_drive - thr_hi) + abs(taste_drive - thr_hi)))/tau_av_win : 1
        # Low threshold of the Hedonic window
        dav_under/dt = (-av_under + 0.5*((thr_lo - taste_drive) + abs(thr_lo - taste_drive)))/tau_av_win : 1

        # Over time adapting HIGH
        dthr_hi/dt = (-(thr_hi - thr0_hi)
                      + k_hab_hi*av_over*(1 + eta_da_win*da_gate)
                      - k_sens_hi*av_over*(1 - da_gate)) / tau_thr_win : 1
        # Over time adapting LOW
        dthr_lo/dt = (-(thr_lo - thr0_lo)
                      + k_hab_lo*av_under
                      - k_sens_lo*av_over) / tau_thr_win : 1
        
        # Stateless parameters for the Hedonic window
        thr0_hi   : 1
        thr0_lo   : 1
        k_hab_hi  : 1
        k_sens_hi : 1
        k_hab_lo  : 1
        k_sens_lo : 1
        da_gate   : 1

        # ----- SPICY aversion -----
        dspice_drive/dt = is_spice * (-spice_drive / tau_sd_spice) : 1

        da_spice/dt = is_spice * (
            -a_spice / tau_a_spice
            + (k_a_spice / tau_a_spice) * 0.5 * ((spice_drive - thr_spice) + abs(spice_drive - thr_spice))
        ) : 1
    

        dthr_spice/dt = is_spice * (
            -(thr_spice - thr0_spice) / tau_thr_spice
            + (k_hab_spice  / tau_thr_spice) * a_spice * (1 + eta_da_spice * da_gate)
            - (k_sens_spice / tau_thr_spice) * a_spice * (1 - da_gate)
        ) : 1

        is_spice : 1
        thr0_spice : 1

    ''', 
    threshold='v > (Vth + theta + theta_bias + wfast)',
    reset='v = Vreset; s += 1; wfast += 0.3*mV',
    refractory=2*b.ms,
    method='euler',
    namespace={
        # LIF neuron constants
        'C': C, 'gL': gL, 'EL': EL,
        'Ee': Ee, 'Ei': Ei,
        'taue': taue, 'taui': taui,
        'Vth': Vth, 'Vreset': Vreset,
        'tau_rate': tau_rate, 'tau_theta': tau_theta,
        'rho_target': rho_target,
        # Hedonic window constants
        'tau_drive_win': tau_drive_win,
        'tau_av_win':    tau_av_win,
        'tau_thr_win':   tau_thr_win,
        'eta_da_win':    eta_da_win,
        # Dynamic SPICY namespaces
        'tau_sd_spice': tau_sd_spice, 'tau_a_spice': tau_a_spice, 'tau_thr_spice': tau_thr_spice,
        'k_a_spice': k_a_spice, 'k_hab_spice': k_hab_spice, 'eta_da_spice': eta_da_spice,
        'k_sens_spice': k_sens_spice
    }
)
# initializing theta bias because 5-HT is going to high the thresholds during aversive episode
taste_neurons.theta_bias[:] = 0 * b.mV # initial bias 0 mV

# 5. Monitors
mon_clock = b.Clock(dt=5*b.ms)
spike_mon = b.SpikeMonitor(taste_neurons) # monitoring spikes and time
state_mon = b.StateMonitor(taste_neurons, 'v', record=True, clock=mon_clock) # monitoring membrane potential

# 6. Poisson INPUT neurons and STDP + eligibility trace synapses
# 1) Labelled stimulus (yes plasticity) -> stimulus Poisson (labelled)
pg = b.PoissonGroup(num_tastes, rates=np.zeros(num_tastes)*b.Hz)

# 2) Neutral noise (no plasticity) -> background Poisson (sorrounding ambient)
baseline_hz = 0.5  # 0.5-1 Hz
pg_noise = b.PoissonGroup(num_tastes, rates=baseline_hz*np.ones(num_tastes)*b.Hz)

# Global Division Inhibition GDI neuron integrator => only one 
gdi_pool = b.NeuronGroup(1, 'dx/dt = -x/tau_gdi : 1',
                         method='euler',
                         namespace={'tau_gdi': tau_gdi})
gdi_pool.x = 0.0

# linking GDI value to the output neurons
taste_neurons.gdi = b.linked_var(gdi_pool, 'x')

# GDI Synapses definition (Feedforward + Feedback) to increment GDI pool value on spikes from both input and output neurons 
# funzionalità: ad ogni spike in ingresso o in uscita, incrementa il valore di GDI di una quantità fissa
# Feedforward: input Poisson -> to GDI
S_ff_gdi = b.Synapses(pg, gdi_pool, 
                      on_pre='x_post += k_e2g_ff', 
                      namespace={'k_e2g_ff': k_e2g_ff})
S_ff_gdi.connect('i != unknown_id')

# Feedback: output neurons -> to GDI
S_fb_gdi = b.Synapses(taste_neurons, gdi_pool, 
                      on_pre='x_post += k_e2g_fb', 
                      namespace={'k_e2g_fb': k_e2g_fb})
S_fb_gdi.connect('i != unknown_id')

# Monitoring GDI
gdi_mon = b.StateMonitor(gdi_pool, 'x', record=[0], clock=mon_clock)

# Triplet STDP (Pfister & Gerstner 2006) + eligibility trace decay + STP (STF/STD) Tsodyks-Markram
# Tracce:
#   x, xbar  -> pre (tau_x_minus, tau_xbar_minus)
#   y, ybar  -> post (tau_y_plus,  tau_ybar_plus)
# Contributi a elig:
#   on_pre  : A2p * y_post      + A3p * y_post * ybar_post
#   on_post : A2m * x_pre       + A3m * x_pre  * xbar_pre
#
# NB: accumuliamo in 'elig' (che poi decresce con Te). Il segno finale lo d� il rinforzo r
#     (positivo -> LTP; negativo -> LTD)
S = b.Synapses(
    pg, taste_neurons,
    model='''
        w            : 1
        x_stp        : 1                 # risorsa disponibile (0..1)
        u            : 1                 # utilizzo corrente (0..1)
        u0           : 1                 # set-point di u
        uinc         : 1                 # incremento per spike
        tau_rec      : second            # STD recovery
        tau_facil    : second            # STF decay
        ex_scale_stp : 1                 # gain STP

        dx/dt     = -x/tau_x_minus        : 1 (event-driven)   # pre  (veloce, LTD)
        dxbar/dt  = -xbar/tau_xbar_minus  : 1 (event-driven)   # Triplet pre  (lenta)
        dy/dt     = -y/tau_y_plus         : 1 (event-driven)   # post (veloce, LTP)
        dybar/dt  = -ybar/tau_ybar_plus   : 1 (event-driven)   # Triplet post (lenta)

        delig/dt  = -elig/Te              : 1 (clock-driven)

        stdp_on    : 1
        ex_scale   : 1

        gamma_gdi  : 1 (shared)
    ''',
    on_pre='''
        u     = u + uinc * (1 - u)
        ge_post += (w * u * x_stp * g_step_exc * ex_scale * ex_scale_stp) / (1.0 + gamma_gdi * gdi_eff_post)
        x_stp = x_stp * (1 - u)

        x    += 1.0
        xbar += 1.0

        elig += stdp_on * ( (A2p * y + A3p * y * ybar) * (1 - w) )
    ''',
    on_post='''
        y    += 1.0
        ybar += 1.0

        elig += stdp_on * ( (A2m * x + A3m * x * xbar) * w )
    ''',
    method='exact',
    namespace={
        'g_step_exc': g_step_exc, 'Te': Te,

        # time constants per le tracce (ms -> second via unit� Brian2)
        'tau_y_plus':     33*b.ms,
        'tau_ybar_plus': 200*b.ms,
        'tau_x_minus':    16*b.ms,
        'tau_xbar_minus': 66*b.ms,

        # guadagni pair/triplet
        'A2p':  0.0065,   # pair LTP
        'A3p':  0.0070,   # triplet LTP
        'A2m': -0.0070,   # pair LTD
        'A3m': -0.0023,   # triplet LTD
    }
)

# Continual recovery between spikes: x grow up, u go back to U0
S.run_regularly('''
    x_stp += (1 - x_stp) * (dt / tau_rec)
    u     += (u0 - u)    * (dt / tau_facil)
''')

# states initialization
taste_neurons.v[:] = EL
taste_neurons.s[:] = 0
taste_neurons.theta[:] = theta_init
taste_neurons.homeo_on = 1.0 # ON during training

# dynamic SPICY states initialization
sl_spice = taste_slice(spicy_id)
taste_neurons.is_spice[:]   = 0
taste_neurons.is_spice[sl_spice] = 1
taste_neurons.thr0_spice[sl_spice]   = profile['thr0_spice_hub']
taste_neurons.thr_spice[sl_spice]  = 0.0
taste_neurons.spice_drive[sl_spice] = 0.0
taste_neurons.a_spice[sl_spice]     = 0.0
taste_neurons.da_gate[sl_spice]     = 0.0

# Hedonic window initialization
taste_neurons.taste_drive[:] = 0.0
taste_neurons.av_over[:]  = 0.0
taste_neurons.av_under[:] = 0.0
taste_neurons.thr_hi[:]   = 0.0
taste_neurons.thr_lo[:]   = 0.0
taste_neurons.thr0_hi[:]  = 0.0
taste_neurons.thr0_lo[:]  = 0.0
# rewards for each individual -> they will be overwrite with every new individual
taste_neurons.k_hab_hi[:]  = 0.002
taste_neurons.k_sens_hi[:] = 0.001
taste_neurons.k_hab_lo[:]  = 0.0015
taste_neurons.k_sens_lo[:] = 0.0005

# initial thresholds for Hedonic window
#taste_neurons.thr0_spice = profile['thr0_spice_hub'] # initial SPICY threshold

# initializing GDI
taste_neurons.gdi_center = 0.1
taste_neurons.gdi_half   = 0.80 

# Diagonal or dense connection mode for SYNAPSES except for UNKNOWN with population method
if NEURONS_PER_TASTE == 1:
    if connectivity_mode == "diagonal":
        S.connect('i == j and i != unknown_id')
    else:
        S.connect('i != unknown_id and j != unknown_id')
else:
    if connectivity_mode == "diagonal":
        # solo diagonale a popolazioni
        for tx in range(num_tastes):
            if tx == unknown_id: 
                continue
            sl = taste_slice(tx)
            S.connect(i=tx, j=np.arange(sl.start, sl.stop))
    else:  # "dense" popolazionale: p -> slice(q) per tutti i q
        all_posts = {q: np.arange(*taste_slice(q).indices(TOTAL_OUT)) for q in range(num_tastes)}
        for ps in range(num_tastes):
            if ps == unknown_id:
                continue
            for q in range(num_tastes):
                if q == unknown_id:
                    continue
                S.connect(i=ps, j=all_posts[q])

# init weights
if NEURONS_PER_TASTE == 1 and connectivity_mode == "dense":
    # initial advantage for true connections, minimal cross-talk
    S.w['i==j'] = '0.35 + 0.25*rand()'  # 0.30 0.50 value
    S.w['i!=j'] = '0.01 + 0.03*rand()'  # 0.02 0.06 value
else:
    S.w = '0.2 + 0.8*rand()'

# if STP for TRAINING is ON => initialize it
if use_stp:
    S.x_stp[:]     = 1.0
    S.u[:]         = stp_u0
    S.u0[:]        = stp_u0
    S.uinc[:]      = stp_uinc
    S.tau_rec[:]   = stp_tau_rec
    S.tau_facil[:] = stp_tau_facil

    # static corrective gain => drive average as pre-STP
    # to avoid saturation: using STD steady-state formula a r_ref and not considering STF (for now it's very small).
    ux_ss = stp_u0 / (1.0 + stp_u0 * stp_r_ref * float(stp_tau_rec / b.second))
    S.ex_scale_stp[:] = 1.0 / max(1e-3, ux_ss)
else:
    S.ex_scale_stp[:] = 1.0

# slow (consolidated) weights buffer for bio-early stopping
w_slow = S.w[:].copy()
# keep a slow snapshot for the best too
w_slow_best = w_slow.copy()
# scaling factor for noradrenaline
S.ex_scale = 1.0
# Diagonal synapses index
if NEURONS_PER_TASTE == 1:
    ij_to_si = {}
    Si = np.array(S.i[:], dtype=int)
    Sj = np.array(S.j[:], dtype=int)
    for k in range(len(Si)):
        ij_to_si[(int(Si[k]), int(Sj[k]))] = int(k) # synapse index
    # Available diagonal index in 'diagonal' and 'dense'
    diag_idx = {k: ij_to_si[(k, k)] for k in range(num_tastes-1) if (k, k) in ij_to_si} # dictionary to map the synapse couples i->j
else: # within population mode taking all the neurons for the related population
    diag_idx = {k: diag_indices_for_taste(k) for k in range(num_tastes-1)}

# Background synapses (ambient excitation) with GDI
S_noise = b.Synapses(
    pg_noise, taste_neurons,
    model='gamma_gdi : 1 (shared)',
    on_pre='ge_post += (g_step_bg) / (1.0 + gamma_gdi * gdi_eff_post)',
    namespace=dict(g_step_bg=g_step_bg)
)

# Weak non-plastic uniform drive -> UNKNOWN (rifiuto guidato dall'energia diffusa)
#S_unk = b.Synapses(pg, taste_neurons, on_pre='ge_post += 0.30*nS')  # small but on all tastes
# Weak plastic uniform drive gated -> UNKNOWN (data driven from diffuse energy)
S_unk = b.Synapses(pg, taste_neurons,
                   model='gain_unk : 1 (shared)',
                   on_pre='ge_post += gain_unk * (0.30*nS)') # small but only on real UNKNOWN tastes

# Lateral inhibition for WTA (Winner-Take-All) with 5-HT modulation
inhibitory_S = b.Synapses(
    taste_neurons, taste_neurons,
    model='''
        g_step_inh : siemens (shared)
        inh_scale  : 1
    ''',
    on_pre='gi_post += g_step_inh * inh_scale',
    delay=0.2*b.ms,
    namespace={'npt': NEURONS_PER_TASTE, 'unk': unknown_id}
)
# Just inter-pop connection without UNKNOWN -> neurons of the same population cooperate and do not compete:
inhibitory_S.connect('i//npt != j//npt and i//npt != unk and j//npt != unk')
#inhibitory_S.connect('i != j') # inter-pop and intra-pop compete
inhibitory_S.inh_scale = 0.6 # with GDI installed less WTA inhibition
inhibitory_S.g_step_inh = g_step_inh_local

# every spike from SPICY gate increases SPICY neuron drive
S_spice_sensor = b.Synapses(
    pg, taste_neurons,
    on_pre='spice_drive_post += k_spike_spice',
    namespace={'k_spike_spice': k_spike_spice}
)
sl = taste_slice(spicy_id)
S_spice_sensor.connect(i=spicy_id, j=np.arange(sl.start, sl.stop))

w_mon = b.StateMonitor(S, 'w', record=True, clock=mon_clock) # if too heavy -> record[0]
weight_monitors.append((w_mon, S))

# Sensorial synapses for every taste neuron to module Hedonic window
S_drive = b.Synapses(pg, taste_neurons,
                     on_pre='taste_drive_post += k_spike_drive',
                     namespace={'k_spike_drive': k_spike_drive})

# Connections managament for NOISE, DRIVE and UNKNOWN
if NEURONS_PER_TASTE == 1:
    S_noise.connect('i == j and i != unknown_id')
    S_unk.connect('j == unknown_id and i != unknown_id')
    S_drive.connect('i == j and i != unknown_id')
else:
    # i → popolazione di i (tranne UNKNOWN)
    for ts in range(num_tastes):
        if ts == unknown_id:
            continue
        sl = taste_slice(ts)
        S_noise.connect(i=ts, j=np.arange(sl.start, sl.stop))
        S_drive.connect(i=ts, j=np.arange(sl.start, sl.stop))
    # verso UNKNOWN: j in popolazione UNKNOWN
    slu = taste_slice(unknown_id)
    for ts in range(num_tastes):
        if ts == unknown_id:
            continue
        S_unk.connect(i=ts, j=np.arange(slu.start, slu.stop))

# GDI Synapses initialization
S.gamma_gdi = gamma_gdi_0
S_noise.gamma_gdi = gamma_gdi_0
# UNKNOWN demand gain initialization
S_unk.gain_unk = 0.0

# DA, 5-HT, NE, HI, ACh, GABA neuromodulators that decay over time
mod = b.NeuronGroup(
    1,
    model='''
        dDA_f/dt = -DA_f/tau_DA_phasic : 1   # phasic
        dDA_t/dt = -DA_t/tau_DA_tonic  : 1   # tonic
        dHT/dt = -HT/tau_HT : 1
        dNE/dt = -NE/tau_NE : 1
        dHI/dt = -HI/tau_HI : 1
        dACH/dt = -ACH/tau_ACH : 1
        dGABA/dt = -GABA/tau_GABA : 1

        # hungry and need of that type of food
        dHUN/dt  = -HUN/tau_HUN : 1

        # too much of the same kind of food
        dSAT/dt  = -SAT/tau_SAT : 1

        # hydratation management to keep the level of the taste inside the Hedonic window
        dH2O/dt  = -H2O/tau_H2O : 1
    ''',
    method='exact',
    namespace={'tau_DA_phasic': tau_DA_phasic, 'tau_DA_tonic': tau_DA_tonic, 
               'tau_HT': tau_HT, 'tau_NE': tau_NE, 
               'tau_HI' : tau_HI, 'tau_ACH' : tau_ACH, 'tau_GABA' : tau_GABA,
               'tau_HUN': 60*b.second, 'tau_SAT': 120*b.second, 'tau_H2O': 90*b.second}
)
mod.DA_f = 0.0
mod.DA_t = 0.0
mod.HT = 0.0
mod.NE = 0.0
mod.HI = 0.0
mod.ACH  = 0.0
mod.GABA = 0.0
mod.HUN = 0.2
mod.SAT = 0.0
mod.H2O = 0.2

# 8. Building the SNN network and adding levels
net = b.Network(
    taste_neurons,
    pg,
    pg_noise, # imput neurons noise introduced
    S,
    S_noise,
    S_drive, # Hedonic window synapses
    inhibitory_S,
    spike_mon,
    state_mon,
    mod, # neuromodulator neurons installed into the net
    S_spice_sensor, # to monitor dynamic SPICY
    gdi_pool,    # GDI management
    S_ff_gdi,
    S_fb_gdi
)

# monitoring all neuromodulators and aversion to SPICY
net.add(w_mon)
inh_mon = b.StateMonitor(inhibitory_S, 'inh_scale', record=[0], clock=mon_clock)
net.add(inh_mon)
theta_mon = b.StateMonitor(taste_neurons, 'theta', record=[0], clock=mon_clock)
s_mon = b.StateMonitor(taste_neurons, 's', record=[0], clock=mon_clock)
net.add(theta_mon, s_mon)
mod_mon = b.StateMonitor(mod, ['DA_f','DA_t','HT','NE','HI','ACH','GABA'], record=[0], clock=mon_clock)
net.add(mod_mon)
# Hedonic window for SPICY nociceptive
spice_mon = b.StateMonitor(taste_neurons, ['spice_drive','thr_spice','a_spice','da_gate'],
                           record=np.arange(taste_slice(spicy_id).start, taste_slice(spicy_id).stop))
net.add(spice_mon)
# Hedonic window monitor
hed_mon = b.StateMonitor(taste_neurons, ['taste_drive','thr_hi','thr_lo','av_over','av_under','da_gate'], record=[0], clock=mon_clock)
net.add(hed_mon)
# Monitoring all the GDI states
ge_mon = b.StateMonitor(taste_neurons, ['ge','gdi_eff'], record=[0], clock=mon_clock)
net.add(ge_mon)
net.add(gdi_mon)
# unknown gate without learning it
net.add(S_unk)

################## DATASET PREPARATION ##################

# 9. Prepare stimuli list (TRAIN / VAL / TEST con split disgiunto + DAF)
rng = np.random.default_rng(123)
# iperparametri split
PAIR_SPLIT   = (0.6, 0.2, 0.2)   # train, val, test fractions
TRIPLE_SPLIT = (0.6, 0.2, 0.2)
PURE_VAL_PER_CLASS  = 20
PURE_TEST_PER_CLASS = 20
# Data Augmentation Factor 
DAF_MIX  = 1   # quante varianti augment per ogni mix base (coppie+triple) in TRAIN
DAF_PURE = 2   # 1 = niente extra sui puri (puoi portarlo a 2 se vuoi)
# enumerazione coppie e triple disgiunte
pairs = [(ia, ja) for ia in range(unknown_id) for ja in range(ia+1, unknown_id)]
triples = []
for ia in range(unknown_id):
    for ja in range(ia+1, unknown_id):
        for k in range(ja+1, unknown_id):
            triples.append((ia, ja, k))
rng.shuffle(pairs)
rng.shuffle(triples)

def _split(lst, split):
    n = len(lst)
    a = int(split[0]*n)
    b = int(split[1]*n)
    return lst[:a], lst[a:a+b], lst[a+b:]

pairs_tr, pairs_val, pairs_te = _split(pairs, PAIR_SPLIT)
trip_tr,  trip_val,  trip_te  = _split(triples, TRIPLE_SPLIT)

# numero di ripetizioni per bilanciare il training
# PRIMA DI costruire mixture_train, fai un warmup solo puri
PURE_WARMUP_EPOCHS = 5
pure_warmup = []
for _ in range(PURE_WARMUP_EPOCHS):
    for taste_id in range(unknown_id):
        vs, ids, lab = make_mix([taste_id], amp=np.random.randint(240, 301))
        pure_warmup.append((vs, ids, lab + " (warmup)"))

# PURE: Train/Val/Test
# a. Train
pure_train = []
for taste_id in range(unknown_id):
    for _ in range(n_repeats * max(1, DAF_PURE)):
        vs, ids, lab = make_mix([taste_id], amp=np.random.randint(220, 301))
        pure_train.append((vs, ids, lab + " (train)"))

# b. Val
pure_val = []
for taste_id in range(unknown_id):
    for _ in range(PURE_VAL_PER_CLASS):
        vs, ids, lab = make_mix([taste_id], amp=rng.integers(220, 321))
        pure_val.append((vs, ids, lab + " [VAL]"))

# c. Test
pure_test = []
for taste_id in range(unknown_id):
    for _ in range(PURE_TEST_PER_CLASS):
        vs, ids, lab = make_mix([taste_id], amp=rng.integers(220, 321))
        pure_test.append((vs, ids, lab + " [TEST]"))

# MIX: train con augmentation, val/test “puliti” -> niente augment
mixture_train = []

# coppie train (+DAF_MIX varianti ciascuna)
for (ia, ja) in pairs_tr:
    for _ in range(max(1, DAF_MIX)):
        va, ids, lab = augment_mix([ia, ja], amp=rng.integers(200, 321), rng=rng)
        mixture_train.append((va, ids, lab))

# triple train (+DAF_MIX varianti ciascuna)
for (ia, ja, ka) in trip_tr:
    for _ in range(max(1, 3)):   # DAF_MIX = 3 per le triple
        va, ids, lab = augment_mix([ia, ja, ka], amp=rng.integers(200, 321), rng=rng)
        mixture_train.append((va, ids, lab))

# coppie difficili in train (es: BITTER+SALTY) con augmentation
for _ in range(4*n_repeats):
    mixture_train.append(augment_mix([1, 2], amp=240, rng=rng))
    mixture_train.append(augment_mix([2, 1], amp=240, rng=rng))

# alcune triple difficili in train
for _ in range(n_repeats):
    mixture_train.append(augment_mix([1, 2, 4], amp=220, rng=rng))
    mixture_train.append(augment_mix([2, 6, 1], amp=220, rng=rng))

# coppie asimmetriche ad alto SNR in train
for _ in range(2*n_repeats):
    for (ia, ja) in [(0,4), (1,2), (3,5), (2,6)]:
        va, ids, lab = make_asymmetric_pair(ia, ja, amp_hi=rng.integers(240, 321), rng=rng)
        va = jitter_active(va, frac=0.15, rng=rng)
        va = global_gain(va, lo=0.9, hi=1.15, rng=rng)
        mixture_train.append((va, ids, lab + " (train)"))

# più puri per SWEET/SOUR/FATTY
for _ in range(2*n_repeats):
    for ts in [0,3,5]:
        pure_train.append(make_mix([ts], amp=rng.integers(240, 321)))

# coppie asimmetriche focalizzate
for _ in range(2*n_repeats):
    for (ia, ja) in [(0,5), (0,3), (3,5), (3,4), (5,4)]:
        va, ids, lab = make_asymmetric_pair(ia, ja, amp_hi=rng.integers(260, 321), rng=rng)
        va = jitter_active(va, frac=0.15, rng=rng)
        va = global_gain(va, lo=0.9, hi=1.15, rng=rng)
        mixture_train.append((va, ids, lab + " (train+)"))

# triple asimmetriche ad alto SNR in train
for (ia,ja,ka) in trip_tr:
    for _ in range(3):  # aumenta DAF solo triple
        mixture_train.append(make_asymmetric_triple(ia,ja,ka, amp_hi=rng.integers(260, 321), rng=rng))

# new random couples of noisy tastes and mixtures to stress more the net
extra_mixes = [
    [1,4,6], [1,5,6], [2,5,6], [0,6,3], [0,2,4], [3,4,5], [0,1,5], [0,3,2,5,6], [1,6,2],
    [2,3,4,5], [3,5,6,2], [1,3,4,6], [0,2,3,4,5]
]
for _ in range(2*n_repeats):  # repeat a bit but not too much or training will be too long
    for mix in extra_mixes:
        mixture_train.append(noisy_mix(mix, amp=np.random.randint(200, 321)))

# extremely difficult couples
for _ in range(2*n_repeats):
    for (ax,bs) in [(0,4),(0,3),(2,6),(4,6)]:  # SWEET-UMAMI, SWEET-SOUR, SALTY-SPICY, UMAMI-SPICY
        va, ids, lab = make_asymmetric_pair(ax, bs, amp_hi=rng.integers(260, 321), rng=rng)
        va = jitter_active(va, frac=0.12, rng=rng)
        va = global_gain(va, lo=0.9, hi=1.15, rng=rng)
        mixture_train.append((va, ids, lab + " (train++)"))

# new random quadruples and quintuples
extra_mixes_q = [
    [1,4,6,3], [1,5,6,4,2], [2,5,6,1], [0,1,6,3,5], [0,2,4,3], [3,4,5,6], [0,1,5,2], [0,1,6,5,3], [1,6,2,4],
    [1,2,3,4], [3,6,1,2], [1,2,4,5,6], [0,1,2,4,6]
]
for _ in range(2*n_repeats):  # repeat a bit but not too much or training will be too long
    for mix in extra_mixes_q:
        mixture_train.append(noisy_mix(mix, amp=np.random.randint(200, 260)))

# adding specific difficult pairs in training set to improve learning
for _ in range(2*n_repeats):
    mixture_train.append(augment_mix([0,4], amp=250, rng=rng))  # SWEET+UMAMI
    mixture_train.append(augment_mix([0,3], amp=250, rng=rng))  # SWEET+SOUR
    mixture_train.append(augment_mix([3,4], amp=240, rng=rng))  # SOUR+UMAMI

# Coppie asimmetriche che includano sempre 2 e/o 3
hard_pairs = [(2,3), (2,6), (3,5), (0,3), (2,4)]
for _ in range(2*n_repeats):
    for ia, ja in hard_pairs:
        va, ids, lab = make_asymmetric_pair(ia, ja, amp_hi=rng.integers(260, 321), rng=rng)
        va = jitter_active(va, frac=0.15, rng=rng)
        va = global_gain(va, lo=0.9, hi=1.15, rng=rng)
        mixture_train.append((va, ids, lab + " (train)"))

#print(mixture_train[0])

# VAL: coppie e triple NON viste in train (no augment)
# pairs
mixture_val = [] # empty list for validation pairs
for (ia, ja) in pairs_val:
    va, ids, lab = make_mix([ia, ja], amp=rng.integers(220, 321))
    mixture_val.append((va, ids, lab + " [VAL]"))

# triples
for (ia, ja, ka) in trip_val:
    va, ids, lab = make_mix([ia, ja, ka], amp=rng.integers(180, 341))
    mixture_val.append((va, ids, lab + " [VAL]"))

# some extra difficult mixtures in val never seen before
extra_mixes_val = [
    [0,2,5], [1,5,6], [2,5,6], [0,6,3], [0,2,6], [3,6,5], [0,1,5], [1,3,2,5], [1,4,2],
    [2,0,3,5], [3,1,0,2], [1,3,4,5], [0,2,3,4,1]
]
# repeat a bit but not too much or training will be too long
for mix in extra_mixes_val:
    mixture_val.append(make_mix(mix, amp=np.random.randint(200, 321)))

# new random quadruples and quintuples for VAL
extra_mixes_q = [
    [2,3,4,5], [1,4,5,6,2], [3,5,6,1], [1,6,0,5], [0,2,4,3,5], [6,5,2,1], [3,4,1,6], [1,2,6,4,3], [5,6,4,2],
    [1,2,3,4,5], [2,4,3,1], [5,2,6,3], [6,1,2,4,5]
]
# repeat a bit but not too much or training will be too long
for mix in extra_mixes_q:
    mixture_val.append(make_mix(mix, amp=np.random.randint(200, 260)))

# TEST: coppie e triple NON viste in train/val (no augment)
mixture_test = []
for (ic, jc) in pairs_te:
    vc, ids, lab = make_mix([ic, jc], amp=rng.integers(220, 321))
    mixture_test.append((vc, ids, lab + " [TEST]"))

for (ic, jc, kc) in trip_te:
    vc, ids, lab = make_mix([ic, jc, kc], amp=rng.integers(180, 341))
    mixture_test.append((vc, ids, lab + " [TEST]"))

# some extra difficult mixtures in test never seen before
extra_mixes_test = [
    [0,2,5], [1,5,6], [2,5,6], [0,6,3], [0,2,6], [3,6,5], [0,1,5], [1,3,2,5], [1,4,2],
    [2,0,3,5], [3,1,0,2], [1,3,4,5], [0,2,3,4,1]
]
# repeat a bit but not too much or training will be too long
for mix in extra_mixes_test:
    mixture_test.append(make_mix(mix, amp=np.random.randint(200, 321)))

# new random quadruples and quintuples for TEST
extra_mixes_q = [
    [2,3,4,5], [1,4,5,6,2], [3,5,6,1], [1,6,0,5], [0,2,4,3,5], [6,5,2,1], [3,4,1,6], [1,2,6,4,3], [5,6,4,2],
    [1,2,3,4,5], [2,4,3,1], [5,2,6,3], [6,1,2,4,5]
]
# repeat a bit but not too much or training will be too long
for mix in extra_mixes_q:
    mixture_test.append(make_mix(mix, amp=np.random.randint(200, 260)))

# OOD/NULL/NEAR-OOD: val e test più “larghi”
val_stimuli = []
test_stimuli = []

# VAL OOD/NULL/NEAR-OOD
for _ in range(20):
    val_stimuli.append(make_null())
    val_stimuli.append(make_ood_diffuse())
    val_stimuli.append(make_ood_many(k=rng.integers(3,6)))
    val_stimuli.append(make_near_ood())

# TEST OOD/NULL/NEAR-OOD (più ricchi)
for _ in range(20):
    test_stimuli.append(make_null())
    test_stimuli.append(make_ood_diffuse())
    test_stimuli.append(make_ood_many(k=rng.integers(3,6)))
    test_stimuli.append(make_near_ood())

# compone liste finali
# 1. Train (anche warmup incluso)
training_stimuli = pure_warmup + pure_train + mixture_train
rng.shuffle(training_stimuli)
# 2. Val
val_stimuli   += pure_val + mixture_val
rng.shuffle(val_stimuli)
# 3. Test
test_stimuli  += pure_test + mixture_test
rng.shuffle(test_stimuli)

################## TRAINING PHASE ##################

# decoder parameters initialization
pos_counts = {idx: [] for idx in range(num_tastes-1)}
neg_counts = {idx: [] for idx in range(num_tastes-1)}
ema_neg_m1 = np.zeros(num_tastes-1)  # E[x] neg
ema_neg_m2 = np.zeros(num_tastes-1)  # E[x^2] neg
ema_pos_m1 = np.zeros(num_tastes-1)  # E[x] pos
ema_pos_m2 = np.zeros(num_tastes-1)  # E[x^2] pos

# SPICY initialization
# profile set in the group
taste_neurons.k_hab_hi[:unknown_id]  = profile['k_hab_hi']
taste_neurons.k_sens_hi[:unknown_id] = profile['k_sens_hi']
taste_neurons.k_hab_lo[:unknown_id]  = profile['k_hab_lo']
taste_neurons.k_sens_lo[:unknown_id] = profile['k_sens_lo']

# apply internal state bias before starting training loop
apply_internal_state_bias(profile, mod, taste_neurons) # initial internal-state bias for all tastes
apply_spicy_state_bias(profile, mod, taste_neurons) # SPICY bias for nociceptive taste
# in the beginning thr = thr0
taste_neurons.thr_hi[:] = taste_neurons.thr0_hi[:]
taste_neurons.thr_lo[:] = taste_neurons.thr0_lo[:]

# 10. Main "always-on" loop
print("\nStarting TRAINING phase...")
S.stdp_on[:] = 1.0
S.x[:] = 0; S.xbar[:] = 0
S.y[:] = 0; S.ybar[:] = 0
S.elig[:]  = 0
ema_cop_m1 = np.zeros(num_tastes-1)  # E[x] quando la classe � presente in mix (co-presenza)
ema_cop_m2 = np.zeros(num_tastes-1)  # E[x^2]
n_noti = unknown_id   # max available tastes
# reset GDI
taste_neurons.v[:] = EL
sim_t0 = time.perf_counter()
step = 0
total_steps = len(training_stimuli) # pure + mixture
#  col_norm_target
if use_col_norm and NEURONS_PER_TASTE == 1 and connectivity_mode == "dense" and col_norm_target is None:
    # expected fan-in: all the pre-synaptics except for UNKNOWN
    fanin = (num_tastes - 1)
    init_mean = float(np.mean(S.w[:])) if len(S.w[:]) > 0 else 0.5
    col_norm_target = init_mean * fanin
    # target clamp
    col_norm_target = float(np.clip(col_norm_target, 0.5*fanin*0.2, 1.5*fanin*0.8))
    if verbose_rewards:
        print(f"col_norm_target auto={col_norm_target:.3f} (fanin={fanin}, init_mean={init_mean:.3f})")

################ TRIAL CYCLE ##################

# ablation conditions
# set_ach(False)   # ablation ACh
# set_gaba(False)  # ablation GABA
for input_rates, true_ids, label in training_stimuli:
    step += 1 
    # early soft clamp of GDI for first 300 trials to avoid excessive inhibition at start
    # because weights are still low and GDI can grow up too much
    # this is important especially if the initial weights are high
    # after 300 trials the GDI is free to grow up as needed
    if step <= 300: 
        S.gamma_gdi = min(S.gamma_gdi, 0.10)
        S_noise.gamma_gdi = S.gamma_gdi
    
    # progress bar + chrono + ETA
    frac   = step / total_steps
    filled = int(frac * progress_bar_len)
    bar    = '�'*filled + '�'*(progress_bar_len - filled)

    elapsed = time.perf_counter() - sim_t0
    eta = (elapsed/frac - elapsed) if frac > 0 else 0.0

    if len(true_ids) == 1:
       reaction = taste_reactions[true_ids[0]]
       msg = (f"[{bar}] {int(frac*100)}% | Step {step}/{total_steps} | {label} | {reaction}"
           f" | t={fmt_mmss(elapsed)} | ETA={fmt_mmss(eta)}")
    else:
       msg = (f"[{bar}] {int(frac*100)}% | Step {step}/{total_steps} | {label} (mixture)"
           f" | t={fmt_mmss(elapsed)} | ETA={fmt_mmss(eta)}")

    # print the bar
    pbar_update(msg)

    # Before the stimulus, update internal-state bias and reset the taste variables:
    taste_neurons.taste_drive[:] = 0.0
    taste_neurons.av_over[:]  = 0.0
    taste_neurons.av_under[:] = 0.0
    # GDI reset per-trial
    if gdi_reset_each_trial:
        gdi_pool.x[:] = 0.0

    # ACh must to be high during in training for efficient plasticity
    mod.ACH[:] = ach_train_level

    # after the increasing, neuromodulators must decay
    DA_now = float(mod.DA_f[0] + k_tonic_DA * mod.DA_t[0])  # DA with tonic tail included
    HT_now = float(mod.HT[0])
    NE_now = float(mod.NE[0])
    HI_now = float(mod.HI[0])
    ACH_now = float(mod.ACH[0])
    GABA_now = float(mod.GABA[0])

    # just the effectives
    masked = np.zeros_like(input_rates)
    masked[true_ids] = input_rates[true_ids]
    # Per trial OOD/NEAR-OOD: usa l'intero vettore (UNKNOWN resta già a 0)
    # OVERSAMPLING dinamico ai canali ATTIVI (TRAIN only)
    masked_boosted = masked.copy()
    if USE_DYNAMIC_OVERSAMPLING:   # currently "False"
        act = (masked_boosted[:unknown_id] > 0)

        if step > BOOST_APPLY_GUARD_STEPS and np.any(act):
            wk = CLASS_BOOST[:unknown_id].copy()
            # centra sui soli canali attivi per non alterare l'energia del trial
            sk = float(np.mean(wk[act]))
            if sk > 0:
                wk /= sk
                wk = np.clip(wk, 0.85, 1.20)  # pinna l'effetto per stabilità
            masked_boosted[:unknown_id] *= wk
        else:
            masked_boosted = masked  # nei primissimi step non boostare

    # MORE BIOLOGICAL OVERSAMPLING -> Meta-plasticity Homeostatic/Attention population-level aware
    # (5-HT/HI neuromodulators modulate the attentional state gated) with attentional gain control
    # state gating guidato da 5-HT/HI (bias globale già presente)
    # Neuromodulator-gated population-level intrinsic homeostasis metaplasticity controller
    base_bias_mv = (k_theta_HT * HT_now + k_theta_HI * HI_now)

    # homeostatic attentional bias per class (only training)
    if USE_ATTENTIONAL_BIAS:   # currently "True"
        # usa il vettore CLASS_BOOST come "need" (EMA già aggiornato nel loop precedente)
        need_vec = CLASS_BOOST[:unknown_id].copy()
        # centra rispetto alla media per evitare drift globale
        need_vec /= float(np.mean(need_vec) + 1e-9)
        # quanto ogni classe è "in debito" ( >1 => più bisogno)
        debito = np.maximum(0.0, need_vec - 1.0)

        # normalizza per stabilità (opzionale, mantiene range in [0,1])
        if debito.max() > 0:
            debito = debito / debito.max()

        # applica un piccolo abbassamento soglia sui neuroni di quel gusto
        for ta in range(unknown_id):
            sl = taste_slice(ta)
            bias_ta = base_bias_mv - ATTN_BIAS_GAIN_MV * debito[ta]
            bias_ta = float(np.clip(bias_ta, -ATTN_BIAS_CAP_MV, ATTN_BIAS_CAP_MV))
            taste_neurons.theta_bias[sl] = bias_ta * b.mV
    else:
        taste_neurons.theta_bias[:] = base_bias_mv * b.mV

    # 1) training stimulus with masking on no-target neurons
    if USE_GDI:
        # no rates normalization with GDI
        set_stimulus_vector(masked_boosted, include_unknown=False)
    else:
        set_stimulus_vect_norm(masked_boosted, total_rate=BASE_RATE_PER_CLASS * len(true_ids), include_unknown=False)
    
    # reward gain and WTA addicted to NE/HI/ACh
    S.ex_scale = (1.0 + k_ex_NE * NE_now) * (1.0 + k_ex_HI * HI_now) * (1.0 + k_ex_ACH * ACH_now)
    # initializing the rewarding for GDI
    gamma_val = gamma_gdi_0 * (1.0 + 0.5*NE_now) * (1.0 - 0.3*HI_now)
    gamma_val = max(0.0, min(gamma_val, 0.5))  # clamp to max-limit gamma
    # cap adattivo in base a "diffusione" dell'input
    inp_energy = float(np.sum(masked_boosted[:unknown_id]))
    pmr_in = (float(np.max(masked_boosted[:unknown_id])) / (inp_energy + 1e-9)) if inp_energy > 0 else 0.0
    cap_boost  = float(np.interp(pmr_in, [0.25, 0.45], [0.95, 0.60]))
    if inp_energy < 1e-6:
        cap_boost = 0.10  # cap molto stretto con energia nulla
    gamma_val *= (1.0 + 0.15 * np.tanh(inp_energy/800.0))  # più diffuso ⇒ più squeeze
    gamma_val = min(gamma_val, cap_boost)
     # s.ex_scale with STP and warmup
    if use_stp and (step <= stp_warmup_trials):
        S.ex_scale *= 1.10
        gamma_val = min(gamma_val, 0.06)
    # STP warmup trials to avoid collapsing on a few spikes
    S.gamma_gdi = gamma_val
    S_noise.gamma_gdi = gamma_val

    # inhibition with 5-HT/GABA/HI -> whereas WTA more aggressive when 5-HT is higher, because aversion and fear must to influence the behiaviour during the train over and over
    _inh = (1.0 + k_inh_HT * HT_now + k_inh_NE * NE_now + k_inh_HI * HI_now + k_inh_GABA * GABA_now)
    inhibitory_S.inh_scale = max(0.3, _inh) # clamp to avoid errors

    # environment noise reduction with ACh, NE and HI, HT (clamp e0.05 Hz)
    ne_noise_scale = max(0.05, 1.0 - k_noise_NE * NE_now)
    hi_noise_scale = (1.0 + k_noise_HI * HI_now)
    ach_noise_scale = max(0.05, 1.0 - k_noise_ACH * ACH_now)
    pg_noise.rates = baseline_hz * ne_noise_scale * hi_noise_scale * ach_noise_scale * np.ones(num_tastes) * b.Hz

    # state gating guided by 5-HT because threshold has to be bigger if HT is higher -> behaviour of caution
    # 5-HT increase bias | HI decrease bias
    # NON SCOMMENTARE LA RIGA SOTTO SE USO ATTENTION BIAS O SOVRASCRIVO I BIAS APPENA CALCOLATI
    #taste_neurons.theta_bias[:] = (k_theta_HT * HT_now + k_theta_HI * HI_now) * b.mV
    
    # 2) spikes counting during trial
    prev_counts = spike_mon.count[:].copy()
    net.run(training_duration)
    diff_counts = spike_mon.count[:] - prev_counts

    # population aggregation
    dc_pop = population_scores_from_counts(diff_counts)
    # OOD/NULL hard-negative mining on-the-fly
    #E = float(np.sum(diff_counts[:unknown_id]))
    #pmr = (float(diff_counts[:unknown_id].max()) / (E + 1e-9)) if E>0 else 0.0
    E = float(np.sum(dc_pop[:unknown_id]))
    pmr = (float(dc_pop[:unknown_id].max()) / (E + 1e-9)) if E>0 else 0.0

    ######### REWARDING PHASE #########
    # trial diffuso/ambiguo => usa gate negativo più severo
    is_diffuse_train = (pmr < 0.45)
    if is_diffuse_train:
        j_all = np.asarray(S.j[:], int)
        for q in range(unknown_id):
            # "FP gate" sul training window (EMA-neg già disponibile)
            neg_mu  = float(ema_neg_m1[q])
            neg_sd  = float(ema_sd(ema_neg_m1[q], ema_neg_m2[q]))
            fp_gate_train = max(min_spikes_for_known, neg_mu + 2.2*neg_sd)  # leggermente più severo del test

            #if float(diff_counts[q]) >= fp_gate_train:
            if float(dc_pop[q]) >= fp_gate_train:
                # punizione proporzionale all'eccesso + cautela (5-HT)
                #severity = float(diff_counts[q]) / (fp_gate_train + 1e-9)
                severity = float(dc_pop[q]) / (fp_gate_train + 1e-9)
                # boost leggero se � SPICY (perché tende a vincere nei diffusi)
                #spice_boost = 1.20 if q == spicy_id else 1.0
                #r_off = - spice_boost * beta_offdiag * (1.0 + ht_gain * float(mod.HT[0])) * min(2.0, severity)
                r_off = - 1.15 * beta_offdiag * (1.0 + ht_gain * float(mod.HT[0])) * min(2.0, severity)
                # penalizza TUTTE le sinapsi *� q* (riduce l'attrattore spuriamente -> caldo)
                #idx = np.where(np.asarray(S.j[:], int) == q)[0]
                sl_q  = taste_slice(q)
                idx   = np.where((j_all >= sl_q.start) & (j_all < sl_q.stop))[0]
                if idx.size:
                    S.w[idx] = np.clip(S.w[idx] + r_off * S.elig[idx], 0.0, 1.0)
                    S.elig[idx] = 0.0
        # piccolo boost GABA per stabilit�
        mod.GABA[:] += 0.3 * gaba_pulse_stabilize

    # EMA vectors for decoder
    is_mix_trial = (len(true_ids) >= 2)
    for idx in range(num_tastes-1):
        if idx in true_ids and is_mix_trial:
            # population support
            ema_cop_m1[idx], ema_cop_m2[idx] = ema_update(
                ema_cop_m1[idx], ema_cop_m2[idx], float(dc_pop[idx]), ema_lambda
            )

    #print(f"GDI end: x={float(gdi_pool.x[0]):.3f}, eff={float(taste_neurons.gdi_eff[0]):.3f}")

    # fear/aversion only if the generic taste stimulous overcomes the threshold
    known = slice(0, taste_slice(unknown_id).start)
    drv = np.array(taste_neurons.taste_drive[known])
    thr = np.array(taste_neurons.thr_hi[known])
    if (drv > thr).any():
       mod.HT[:] += 0.2

    # fear/aversion only if the SPICY stimulous overcomes the threshold
    drv_now = float(np.mean(taste_neurons.spice_drive[taste_slice(spicy_id)]))
    thr_now = float(np.mean(taste_neurons.thr_spice[taste_slice(spicy_id)]))
    #drv_now = float(taste_neurons.spice_drive[spicy_id])
    #thr_now = float(taste_neurons.thr_spice[spicy_id])
    if drv_now > thr_now:
        mod.HT[:] += 0.25   # manage the aversion

    # to manage GABA during trial if there are too many spikes -> stabilizing the net
    #total_spikes  = float(np.sum(diff_counts[:unknown_id]))
    #active_neurs  = int(np.sum(diff_counts[:unknown_id] > 0))
    total_spikes  = float(np.sum(dc_pop[:unknown_id]))
    active_neurs  = int(np.sum(dc_pop[:unknown_id] > 0))
    if (active_neurs > gaba_active_neurons) or (total_spikes > gaba_total_spikes):
        mod.GABA[:] += gaba_pulse_stabilize

    # collect all positive and negative counts
    for idx in range(num_tastes-1):
        # population support
        if idx in true_ids:
            pos_counts[idx].append(int(dc_pop[idx]))
        else:
            neg_counts[idx].append(int(dc_pop[idx]))
    # updating EMA decoder parameters during online training
    for idx in range(num_tastes-1):
        # population support
        if idx in true_ids:
            ema_pos_m1[idx], ema_pos_m2[idx] = ema_update(ema_pos_m1[idx], ema_pos_m2[idx],
                                                  float(dc_pop[idx]), ema_lambda)
        else:
            ema_neg_m1[idx], ema_neg_m2[idx] = ema_update(ema_neg_m1[idx], ema_neg_m2[idx],
                                                  float(dc_pop[idx]), ema_lambda)

    # soft-pruning off-diagonali con bassa utilità (ogni 10 step)
    if step % 10 == 0 and NEURONS_PER_TASTE == 1 and connectivity_mode == "dense":
        w_all = np.asarray(S.w[:], float)
        i_all = np.asarray(S.i[:], int)
        j_all = np.asarray(S.j[:], int)
        mask_off = (i_all != j_all) & (j_all != unknown_id) & (i_all != unknown_id)
        # light decay
        factor = 0.995 if ema_perf < 0.45 else 0.992
        w_all[mask_off] *= factor
        S.w[:] = np.clip(w_all, 0.0, 1.0)

    if dc_pop.max() <= 0:
        print("\nThere's no computed spike, skipping rewarding phase...")
        # neuromodulators cooldown
        mod.NE[:] *= 0.98
        mod.HI[:] *= 0.98
        net.run(pause_duration)
        S.elig[:] = 0
        continue

    ############### DECODER + WINNER SELECTION + REINFORCEMENT #############

    # A3: TP/FP threshold for each class
    #scores = diff_counts.astype(float)
    scores = population_scores_from_counts(diff_counts) # neurons population management
    scores[unknown_id] = -1e9
    mx = scores.max()
    rel = threshold_ratio * mx

    tp_gate = np.zeros(num_tastes-1, dtype=float)
    fp_gate = np.zeros(num_tastes-1, dtype=float)

    # weight arrays
    i_all = np.asarray(S.i[:], int)
    j_all = np.asarray(S.j[:], int)
    w_all = np.asarray(S.w[:], float)
    for idx in range(num_tastes-1):
       # negative floor -> FP conservative threshold for EMA
       neg_mu_i = float(ema_neg_m1[idx])
       neg_sd_i = float(ema_sd(ema_neg_m1[idx], ema_neg_m2[idx]))
       thr_ema_i = neg_mu_i + k_sigma * neg_sd_i

       # positive -> strong TP threshold
       pos_sd_i = float(ema_sd(ema_pos_m1[idx], ema_pos_m2[idx]))
       tp_gate_i = max(min_spikes_for_known * ema_factor,
                ema_pos_m1[idx] - ema_factor * pos_sd_i)
       fp_gate_i = max(thr_ema_i, rel)

       # to avoid FP on these two
       if idx in (0,1):  # SWEET=0, BITTER=1
            fp_gate_i *= 1.08  # +8% scalar value on SWEET and BITTER
       if idx in (spicy_id, 4):  # SPICY e UMAMI
            fp_gate_i *= 1.20     # +20% soglia FP
       # absolute max for hot classes
       fp_gate_i = max(fp_gate_i, 0.20 * mx + 4)
       if idx in (0,1,2,3): # boost TP to weak tastes
            tp_gate_i *= 0.85
       # safety clamp for infinite values
       if not np.isfinite(tp_gate_i): tp_gate_i = 0.0
       if not np.isfinite(fp_gate_i): fp_gate_i = 0.0
       # effective gates TP and FP
       tp_gate[idx] = tp_gate_i
       fp_gate[idx] = fp_gate_i

       # LTD mirata sulle colonne “calde” che NON sono nel target
       for q in range(unknown_id):
            if q in true_ids:
                continue
            # se la classe q ha “vinto” troppo
            if float(scores[q]) >= fp_gate[q]:
                sl_q = taste_slice(q)
                mask_col = (j_all >= sl_q.start) & (j_all < sl_q.stop)
                excess = float(scores[q]) / (fp_gate[q] + 1e-9)
                w_all[mask_col] -= beta_offdiag * 0.12 * min(2.0, excess) * w_all[mask_col]
       S.w[:] = np.clip(w_all, 0.0, 1.0)

       # bisogno per classe: TP sotto soglia sui veri; FP sopra soglia sui falsi
       need = np.zeros(unknown_id, dtype=float)
       for ta in range(unknown_id):
            if ta in true_ids:
                need[ta] = max(0.0, float(tp_gate[ta]) - float(dc_pop[ta]))
            else:
                need[ta] = max(0.0, float(scores[ta]) - float(fp_gate[ta]))

       # normalizza il need per stabilità
       m = float(np.mean(need) + 1e-9)
       need_norm = need / m

       # aggiorna il vettore boost (EMA) e clampa
       CLASS_BOOST[:] = (1.0 - BOOST_LAM) * CLASS_BOOST + BOOST_LAM * (1.0 + BOOST_GAIN * need_norm)
       CLASS_BOOST[:] = np.clip(CLASS_BOOST, BOOST_CAP[0], BOOST_CAP[1])

    # selecting all the spiking winning neurons >= threshold_ratio
    sorted_idx = np.argsort(scores)[::-1]
    top = scores[sorted_idx[0]]
    second = scores[sorted_idx[1]] if len(sorted_idx) > 1 else 0.0
    winners = []
    if top >= min_spikes_for_known and second > 0 and (top / (second + 1e-9) >= top2_margin_ratio):
        winners = [int(sorted_idx[0])] # only one dominant taste -> single taste case
    else:
        thr = threshold_ratio * mx
        winners = [idx for idx,c in enumerate(scores) if c >= thr]
        if not winners:
            winners = [int(np.argmax(scores))] # winner tastes > 1 -> mixture case

    # burst NE
    ambiguous = (second > 0 and top/(second + 1e-9) < 1.3) or (len(winners) > 2)
    if ambiguous:
        mod.NE[:] += ne_pulse_amb
        mod.HI[:] += hi_pulse_nov
        inhibitory_S.inh_scale[:] = np.maximum(0.8, inhibitory_S.inh_scale[:] * 0.85)
        thr_rel = 0.18 * top
        for jas in range(unknown_id):
           if jas != sorted_idx[0]:
                if float(scores[jas]) >= max(thr_rel, 0.4 * tp_gate[jas]):
                    if jas not in winners:
                        winners.append(jas)
    
    # absolute minimum on any co-winners:
    if len(winners) > 1:
        winners = [wa for wa in winners if scores[wa] >= max(0.25*top, min_spikes_for_known*0.5)]
        if not winners:
            winners = [int(np.argmax(scores))]

    # SPICY aversion management (if SPICY is present among true or winner ids)
    is_spicy_present = (spicy_id in true_ids) or (spicy_id in winners)

    if is_spicy_present:
        happened, p_now = spicy_aversion_triggered(
            taste_neurons, mod, spicy_id,
            p_base=p_aversion_base,
            slope=p_aversion_slope,
            cap=p_aversion_cap,
            trait=profile['spicy_aversion_trait'],
            k_hun=profile['k_hun_spice'],
            k_h2o=profile['k_h2o_spice']
        )

        if happened:
            mod.HT[:]  += ht_pulse_aversion
            mod.DA_f[:] *= da_penalty_avers
            mod.DA_t[:] *= da_penalty_avers
            sl = taste_slice(spicy_id)
            taste_neurons.thr_spice[sl] = taste_neurons.thr_spice[sl] + thr_spice_kick
            if verbose_rewards:
                print(f"[SPICY-AVERSION] p={p_now:.2f} → HT+{ht_pulse_aversion}, DA×{da_penalty_avers}, thr_spice+={thr_spice_kick:.3f}")

    # ANALYSIS of the scores distribution to detect mix-like patterns
    # prima di applicare i rinforzi, analizza il pattern di punteggi
    # stai per decidere i winner e dare rinforzi. Allentare WTA/GDI prima del rinforzo permette a più classi vere di sparare insieme nei mix (e poi ricevere reward).
    E = float(np.sum(scores[:unknown_id]))
    top = float(np.max(scores[:unknown_id])) if E > 0 else 0.0
    PMR = top / (E + 1e-9) if E > 0 else 0.0
    p = scores[:unknown_id] / (E + 1e-9) if E > 0 else np.zeros(unknown_id)
    p = np.clip(p, 1e-12, 1.0); p /= p.sum()
    H = float(-(p*np.log(p)).sum())

    # Dopo aver calcolato tp_gate/fp_gate
    is_mix_like = (0.35 <= PMR <= 0.60) and (0.8 <= H <= 1.4)
    if is_mix_like: # mix_like lo uso in training, mixture_like in test
        tp_gate = np.maximum(min_spikes_for_known * 0.6, tp_gate * mixture_thr_relax)  # mixture_thr_relax 0.55 già nel codice
        # meno WTA e meno squeeze GDI per far coesistere più gusti
        inhibitory_S.inh_scale[:] = np.maximum(0.30, float(inhibitory_S.inh_scale[0]) * 0.55)
        gamma_val = min(gamma_val, 0.04)
        S.gamma_gdi = gamma_val
        S_noise.gamma_gdi = gamma_val

    # total scores printing
    order = np.argsort(scores)
    dbg = [(taste_map[idx], int(scores[idx])) for idx in order[::-1]]

    # if there is hedonic window dopaminergic reinforcement => DA is modulated by the hedonic window itself keeping states in mind
    if use_hedonic_da:
        av_out = np.array(taste_neurons.av_over[:unknown_id]) + np.array(taste_neurons.av_under[:unknown_id])
        g_win = 1.0 / (1.0 + hed_gate_k * np.maximum(av_out, 0.0))  # 0..1
        hed_fb_vec = state_hed_fallback_vec(mod, base=hed_fallback) # fallback for every different state
    else:
        g_win = np.ones(unknown_id)
        hed_fb_vec = np.full(unknown_id, hed_fallback)

    # biologically plausible dopaminergic latency simulation -> part of eligibility decay
    gwin_ema = (1.0 - meta_lambda) * gwin_ema + meta_lambda * np.asarray(g_win[:unknown_id])
    net.run(dopamine_latency)

    # To monitor the effect of oversampling (dynamic or static), log for taste
    #print("\n")
    log_population_stats(diff_counts, step=step, label="PRE-REWARD")

    # EARLY-STOPPING PERFORMANCE GATE: spegne il consolidamento se il trial è "scarso"
    # jacc è lo stesso che già calcoli (più sotto per i burst), ma lo possiamo stimare subito:
    Ts = set(true_ids)
    Ps = set(winners)
    jacc_proxy = (len(Ts & Ps) / len(Ts | Ps)) if (Ts | Ps) else 1.0

    # EMA della performance per "best snapshot" and EARLY STOPPING
    ema_perf = (1.0 - perf_alpha) * ema_perf + perf_alpha * jacc_proxy
    print(f"EMA PERF per-class: {ema_perf}\n")
        
    # questo gate moltiplica ogni rinforzo r (vedi A4) ⇒ se gate=0, nessun consolidamento
    perf_gate = 1.0 if (jacc_proxy >= DA_GATE_JACC) else 0.0
    # aggiorna storico e stima della varianza della curva EMA
    ema_hist.append(float(ema_perf))
    if len(ema_hist) >= 8:
        _arr = np.asarray(ema_hist, float)
        mu   = float(_arr.mean())
        ema_perf_sd = float(np.sqrt(np.mean(((_arr - mu) ** 2))))
    else:
        ema_perf_sd = 0.0

    # soglia di “vero miglioramento” guidata dal rumore e dagli step dinamici
    IMPROVE_EPS = max(0.002, 0.25 * ema_perf_sd)  # 25% del jitter recente, min 0.002
    MIN_STEPS_BEFORE_STOP = max(200, int(0.15 * total_steps)) # step dinamici da considerare ad ogni trial

    # 4) 3-factors training reinforcement multi-label learning dopamine rewards for the winner neurons
    # A4: DIAGONAL: reward TP, punish big FP
    for idx in range(num_tastes-1):
        idx_list = diag_indices_for_taste(idx)  # indici sinapsi verso la popolazione "idx"
        if idx_list.size == 0:
            continue

        spikes_i = float(dc_pop[idx])  # coerenza: usi la metrica "pop"

        # calcolo r
        r = 0.0
        if idx in true_ids:
            # TP forte
            if spikes_i >= tp_gate[idx]:
                ht_eff = min(HT_now, 0.5)
                r = (alpha * (1.0 + da_gain * DA_now) * (1.0 + ach_plasticity_gain * ACH_now)) / (1.0 + ht_gain * ht_eff)
                r *= (1.0 + ne_gain_r * NE_now) * (1.0 + hi_gain_r * HI_now)
                conf = np.clip((top - second) / (top + 1e-9), 0.0, 1.0)
                r *= (0.5 + 0.5 * conf)  # ∈ [0.5, 1.0]
                if use_hedonic_da:
                    hed_mult = hed_fb_vec[idx] + (1.0 - hed_fb_vec[idx]) * float(g_win[idx])
                    r *= hed_mult
                r *= meta_scale(idx)
            # TP debole (micro-reward)
            else:
                pos_mu_i = float(ema_pos_m1[idx]); pos_mu_t = max(1.0, pos_mu_i)
                z_i = spikes_i / pos_mu_t
                if z_i >= 0.40:
                    r = 0.25 * alpha * (1.0 + da_gain * DA_now) / (1.0 + ht_gain * HT_now)
                    r *= (1.0 + ne_gain_r * NE_now) * (1.0 + hi_gain_r * HI_now)
                    if use_hedonic_da:
                        hed_mult = hed_fb_vec[idx] + (1.0 - hed_fb_vec[idx]) * float(g_win[idx])
                        r *= hed_mult
                    r *= meta_scale(idx)
        else:
            # FP forte (dopo warmup)
            if step > fp_gate_warmup_steps and spikes_i >= fp_gate[idx]:
                r = - beta * (1.0 + ht_gain * HT_now)
                r *= (1.0 + ne_gain_r * NE_now)

        # applicazione unica del rinforzo con gating
        # perf_gate ∈ {0,1} o [0,1] (dipende da come lo calcoli); PLAST_GLOBAL ∈ [0,1]
        r_final = perf_gate * PLAST_GLOBAL * r
        if r_final != 0.0 and abs(r_final) * np.max(S.elig[idx_list]) >= 1e-6:
            # vettorizzato su tutta la popolazione di "idx"
            # S.elig è già il 3° fattore (eligibility trace)
            delta_vec = r_final * S.elig[idx_list]
            # applica e clampa
            S.w[idx_list] = np.clip(S.w[idx_list] + delta_vec, 0.0, 1.0)
            # svuota le tracce consumate
            S.elig[idx_list] = 0.0

    # A5: OFF-DIAGONAL: punish p->q when q is big FP
    if use_offdiag_dopamine:
        for p in true_ids:
            for q in range(num_tastes-1):
                if q == p:
                    continue

                # (E): se q è VERO ma "debole", penalità morbida p->q
                # "debole" = sotto metà della propria soglia di classe (scala già test-time)
                if q in true_ids:
                    #spikes_q = float(diff_counts[q])
                    spikes_q = float(dc_pop[q])
                    # usa la soglia positiva online
                    weak_q = (spikes_q < 0.5 * max(min_spikes_for_known, tp_gate[q]))
                    # quando una classe vera è debole, è il momento giusto per sgonfiare le connessioni parassite p�q che la disturbano.
                    if weak_q:
                        undershoot = max(0.0, 0.5*max(min_spikes_for_known, tp_gate[q]) - spikes_q)
                        scale = 1.0 + 0.5 * (undershoot / (0.5*max(1.0, tp_gate[q])))
                        # penalità morbida (>> più piccola della FP severa)
                        r_off_soft = (- 0.25 * beta_offdiag
                            * (1.0 + ht_gain * HT_now)
                            * (1.0 + ne_gain_r * NE_now))
                        r_off_soft *= scale
                        r_off_soft *= meta_scale(p)

                        # symmetry with population support
                        idx_list = diag_indices_for_taste(p, q)
                        if idx_list.size == 0:
                            continue

                        for si in idx_list:
                            delta_soft = r_off_soft * float(S.elig[si])
                            if delta_soft != 0.0:
                                S.w[si] = float(np.clip(S.w[si] + delta_soft, 0, 1))
                            S.elig[si] = 0.0

                        sj_list = diag_indices_for_taste(q, p)
                        for sj in sj_list:
                            delta_sym = 0.5 * r_off_soft * float(S.elig[sj])
                            if delta_sym != 0.0:
                                S.w[sj] = float(np.clip(S.w[sj] + delta_sym, 0, 1))
                            S.elig[sj] = 0.0

                    # in ogni caso, se q è vero abbiamo finito qui
                    continue

                # caso originale FP severo su q
                #if step > fp_gate_warmup_steps and float(diff_counts[q]) >= fp_gate[q]:
                if step > fp_gate_warmup_steps and float(dc_pop[q]) >= fp_gate[q]:

                    # OFF-diagonal FP severity on q (how much the threshold is exceeded)
                    #severity_q = float(diff_counts[q]) / float(fp_gate[q] + 1e-9)
                    severity_q = float(dc_pop[q]) / float(fp_gate[q] + 1e-9)
                    severity_q = float(np.clip(severity_q, 1.0, 2.0))  # 1x..2x

                    # trial confidence
                    conf = np.clip((top - second) / (top + 1e-9), 0.0, 1.0)

                    # true class 'p' hedonic scaling
                    if use_hedonic_da:
                        hed_mult_p = float(hed_fb_vec[p] + (1.0 - hed_fb_vec[p]) * float(g_win[p]))
                    else:
                        hed_mult_p = 1.0

                    # negative off-diagonal reward related to the p class and q severity
                    r_off = - beta_offdiag * (1.0 + ht_gain * HT_now) * (1.0 + ne_gain_r * NE_now)
                    r_off *= severity_q * (0.5 + 0.5 * conf) * hed_mult_p
                    r_off *= meta_scale(p)
                    
                    # population support
                    idx_list = diag_indices_for_taste(p, q)
                    if idx_list.size == 0:
                        continue

                    for si in idx_list:
                        delta = r_off * float(S.elig[si])
                        if delta != 0.0:
                            old_w = float(S.w[si])
                            S.w[si] = float(np.clip(S.w[si] + delta, 0, 1))
                            if verbose_rewards and step % 10 == 0:
                                print(f"  offdiag[cls] - {taste_map[p]}->{taste_map[q]} | "
                                    f"spk_q={float(dc_pop[q]):.1f} fp_q={fp_gate[q]:.1f} sev={severity_q:.2f} hed={hed_mult_p:.2f} "
                                    f"w={delta:+.4f}  w:{old_w:.3f}->{float(S.w[si]):.3f}\n")
                        S.elig[si] = 0.0
                    
                    # light symmetrical pattern q->p with population support
                    sj_list = diag_indices_for_taste(q, p)
                    #sj_sym = ij_to_si.get((q, p), None)
                    #if sj_sym is not None:
                    for sj in sj_list:
                        delta_sym = 0.5 * r_off * float(S.elig[sj])
                        if delta_sym != 0.0:
                            S.w[sj] = float(np.clip(S.w[sj] + delta_sym, 0, 1))
                        S.elig[sj] = 0.0

                    # safety fallback -> if FP is very strong, normalization on 'q' column again
                    if severity_q > 1.5 and use_col_norm and NEURONS_PER_TASTE == 1 and connectivity_mode == "dense":
                        w_all = np.asarray(S.w[:], dtype=float)
                        i_all = np.asarray(S.i[:], dtype=int)
                        j_all = np.asarray(S.j[:], dtype=int)

                        idx_col = np.where(j_all == q)[0]
                        if idx_col.size:
                            col = w_all[idx_col].copy()

                            if col_floor > 0.0:
                                col = np.maximum(col, col_floor)

                            # diagonal bias for normalization
                            if diag_bias_gamma != 1.0:
                                dloc = np.where(i_all[idx_col] == q)[0]
                                if dloc.size:
                                    col[dloc[0]] *= float(diag_bias_gamma)

                            L1 = float(np.sum(col))
                            target = col_norm_target if col_norm_target is not None else L1
                            if L1 > 1e-12:
                                if L1 > target:
                                    scale = target / L1
                                    col = np.clip(col * scale, 0.0, 1.0)
                                elif col_allow_upscale and (L1 < col_upscale_slack * target):
                                    scale = min(col_scale_max, (target / L1))
                                    col = np.clip(col * scale, 0.0, 1.0)

                                # anti-crosstalk: lieve leak sugli off-diagonali (corretto: usa idx_col)
                                pre_idx = i_all[idx_col]
                                #hot = np.array([1,0,0,0,1,0,1], dtype=float)  # SWEET, UMAMI, SPICY
                                leak_hot = 0.96
                                post = q
                                for k_local, pre in enumerate(pre_idx):
                                    if pre != post:
                                        col[k_local] *= leak_hot  # 2% controlled leak

                            w_all[idx_col] = col
                            S.w[:] = w_all

    # burst neuromodulators DA and 5-HT for the next trial as in a human-inspired biology brain
    # quality = Jaccard(T, P)
    T = set(true_ids); P = set(winners)
    jacc = len(T & P) / len(T | P) if (T | P) else 1.0

    if jacc >= 0.67:
        # scaled reward
        mod.DA_f[:] += da_pulse_reward * jacc
        mod.DA_t[:] += da_tonic_tail * jacc
    elif jacc > 0.0:
        mod.DA_f[:] += 0.4 * da_pulse_reward * jacc # partial reward
        mod.HI[:] += 0.5 * hi_pulse_miss
    else:
        mod.HI[:] += hi_pulse_miss

    # tonic bias internal state (HUN/H2O)
    mod.DA_t[:] += k_hun_tonic * float(mod.HUN[0]) + k_h2o_tonic * float(mod.H2O[0])

    # if the prevision is good => reward to every taste in the trial
    if jacc >= 0.67:
        for tid in true_ids:
            if tid != unknown_id:
                taste_neurons.da_gate[taste_slice(tid)] = float(g_win[tid])
                #taste_neurons.da_gate[tid] = float(g_win[tid])  # 0..1 reinforcement applied on the hedonic window base
        net.run(reinforce_dur)
        for tid in true_ids:
            if tid != unknown_id:
                taste_neurons.da_gate[taste_slice(tid)] = 0.0
        # Slow consolidation (DA-gated, bio-like)
        DA_eff = float(mod.DA_f[0] + k_tonic_DA * mod.DA_t[0])
        if USE_SLOW_CONSOLIDATION and (DA_eff >= DA_THR_CONSOL):
        # capture a fraction of the current (fast) weights into the slow store
            w_slow += ETA_CONSOL * (S.w[:] - w_slow)

    # clamp among thresholds for stability
    eps_thr = 0.02
    hi = np.array(taste_neurons.thr_hi[:]); lo = np.array(taste_neurons.thr_lo[:])
    hi = np.maximum(hi, lo + eps_thr)
    taste_neurons.thr_hi[:] = hi
    
    # EARLY STOPPING tracking & bio-like stagnation plateau handling to avoid overfitting and optimize training duration
    if ema_perf > (best_score + IMPROVE_EPS):
        best_step  = step
        best_score = float(ema_perf)
        best_state = snapshot_state()             # snapshot completo (pesate + stati)
        if w_slow is not None:
            w_slow_best = w_slow.copy()           # se usi consolidamento lento

        patience = 0
        decays_done = 0          # nuovo ciclo “explore”: azzero i decay conteggiati
        cooldown_left = 0

        # re-opening plasticity in a slower way
        if 'stdp_on' in S.variables:
            curs = float(np.mean(S.stdp_on[:]))
            target_s = 1.0
            set_plasticity_scale(0.50 * curs + 0.50 * target_s)

        print(f"NEW BEST ema_perf: {best_score:.4f} at step {best_step} (+>{IMPROVE_EPS:.4f} increment)\n")

    else:
        # no best improvement
        patience += 1

        # 3a) soft-pull verso il best durante plateau prolungato (bio consolidamento)
        if USE_SOFT_PULL and patience >= max(W_EMA, PLATEAU_WINDOW) and best_state is not None:
            soft_pull_toward_best(best_state, rho=RHO_PULL)

        # 3b) Reduce-on-plateau biologico: decresci la plasticità a gradini con cooldown
        if patience >= PLATEAU_WINDOW:
            did_decay = False
            if cooldown_left > 0:
                cooldown_left -= 1
            else:
                # Applica un solo decay e avvia cooldown
                if decays_done < MAX_PLASTICITY_DECAYS:
                    new_scale = max(PLAST_GLOBAL_FLOOR, PLAST_GLOBAL * REDUCE_FACTOR)
                    if new_scale < PLAST_GLOBAL:  # evita no-op
                        set_plasticity_scale(new_scale)
                        decays_done += 1
                        cooldown_left = COOLDOWN   # reset cooldown reale
                        did_decay = True
                        print(f"[ReduceLROnPlateau] plasticity at step {step} → stdp_on≈{float(np.mean(S.stdp_on[:])):.3f} ↓ (decays={decays_done}/{MAX_PLASTICITY_DECAYS}).\n")
            # reset solo se è avvenuto il decay contrassegnato dal flag apposito
            if did_decay:
                patience = 0  

        # 3c) Early Stopping finale
        if (patience >= PATIENCE_LIMIT) and (step >= MIN_STEPS_BEFORE_STOP):
            print(f"\n[EARLY STOPPING] No plasticity best improvement for {PATIENCE_LIMIT} consecutive trials "
                f"(best={best_score:.4f} @ step {best_step}) — stopping training.")
            break

    # with many strong FP, increase 5-HT -> future caution in the next trial
    has_strong_fp = any(
            (idx not in true_ids) and (float(scores[idx]) >= fp_gate[idx])
            for idx in range(num_tastes-1)
    )
    if has_strong_fp:
        mod.HT[:] += ht_pulse_fp
        mod.NE[:] += 0.5 * ne_pulse_amb # arousal on strong FP
        mod.HI[:] += 0.3 * hi_pulse_nov
        mod.GABA[:] += 0.3 * gaba_pulse_stabilize

    # safe clip on theta for homeostasis
    theta_min, theta_max = -12*b.mV, 12*b.mV
    taste_neurons.theta[:] = np.clip((taste_neurons.theta/b.mV), float(theta_min/b.mV), float(theta_max/b.mV)) * b.mV

    # Column normalization (incoming synaptic scaling) with population management
    if use_col_norm and connectivity_mode == "dense":
        if (step % col_norm_every) == 0:
            col_norm_pop(target=col_norm_target, mode=col_norm_mode, temperature=col_norm_temp,
                     diag_bias_gamma=diag_bias_gamma, floor=col_floor,
                     allow_upscale=col_allow_upscale, scale_max=col_scale_max)
        
    # To monitor the effect of oversampling (dynamic or static), log for taste
    log_population_stats(diff_counts, step=step, label="POST-REWARD")  
    # 5) eligibility trace decay among trials
    net.run(pause_duration)
    S.elig[:] = 0

# best-snapshot CHECKPOINT restore states with the saved best_score until now
# not replacing weights because it's not biological-like method, but pushing weights to become their better version saved until now
if best_state is not None:
    # SOFT restore (bio-like)
    # 3a) blend fast weights toward best weights
    RHO_FINAL = 0.60  # 0..1, how much you lean to best fast weights
    S.w[:] = (1.0 - RHO_FINAL) * S.w[:] + RHO_FINAL * best_state['w']

    # 3b) mix in the slow consolidated trace for stability in test
    if USE_SLOW_CONSOLIDATION and (w_slow_best is not None):
        S.w[:] = (1.0 - BETA_MIX_TEST) * w_slow_best + BETA_MIX_TEST * S.w[:]
    
    # 3c) restore NON-weight state from the best snapshot (thresholds, modulators, traces, etc.)
    tmp_w = S.w[:].copy()
    restore_state_without(best_state)   # restores the rest of the state without overwrite w
    S.w[:] = tmp_w              # re-apply our blended weights
    # final print logs
    print(f"\n[BEST-CHECKPOINT] Restored bio-mixed state (ema_perf={best_score:.3f} @ step {best_step}).")
else:
    print("\n[BEST-CHECKPOINT] No best snapshot captured; proceeding with current state.")

# clean the bar
pbar_done()
print(f"Ended TRAINING phase! (elapsed: {fmt_mmss(time.perf_counter()-sim_t0)})\n")

# computing per-class thresholds
thr_per_class_train = np.zeros(num_tastes)
for idx in range(num_tastes-1):
    neg = np.asarray(neg_counts[idx], dtype=float)
    if neg.size == 0:
        neg = np.array([0.0])
    mu_n  = float(np.mean(neg))
    sd_n  = float(np.std(neg))
    thr_gauss = mu_n + k_sigma * sd_n
    thr_quant = float(np.quantile(neg, q_neg)) if np.isfinite(neg).any() else 0.0
    sd_ema = ema_sd(ema_neg_m1[idx], ema_neg_m2[idx])
    thr_ema = ema_neg_m1[idx] + k_sigma * sd_ema
    thr_per_class_train[idx] = max(float(min_spikes_for_known), thr_gauss, thr_quant, thr_ema)

# SOFT CAP dalle positive (EMA train)
# thr_from_pos = mu_pos - 0.5*sd_pos (clamp a min_spikes_for_known)
for c in range(unknown_id):
    mu_pos = float(ema_pos_m1[c])
    sd_pos = float(ema_sd(ema_pos_m1[c], ema_pos_m2[c]))
    thr_from_pos = max(float(min_spikes_for_known), mu_pos - 0.5*sd_pos)
    thr_per_class_train[c] = min(float(thr_per_class_train[c]), float(thr_from_pos))

# taglio secco iniziale -20% su tutte le classi
# (se vuoi sbloccare SUBITO le classi “fredde”; lascia commentato se non serve)
for c in range(unknown_id):
    thr_per_class_train[c] = 0.8 * float(thr_per_class_train[c])

print("Per-class thresholds @train (�+k�, quantile, EMA):",
      {taste_map[idx]: int(thr_per_class_train[idx]) for idx in range(num_tastes-1)})

# 3) Versione per il TEST: SOLO SCALING TEMPORALE
dur_scale = float(test_duration / training_duration)
thr_per_class_test = thr_per_class_train.copy()
thr_per_class_test[:unknown_id] *= dur_scale

# UNSUPERVISED LEARNING without knowing the label "a priori"
# Prototipo "forte" = aspettativa positiva media (scalata al test window)
proto_pos = np.maximum(ema_pos_m1 * dur_scale, 1.0)

# Prototipo "co-presenza" (più indulgente per i mix)
proto_cop = np.maximum(ema_cop_m1 * dur_scale, 1.0)

# Matrice dei prototipi: scegliamo un blend conservativo
#   - se il trial sarà "mix-like" useremo di più proto_cop,
#   - altrimenti proto_pos. Il blend preciso lo faremo nel punto di decisione.
P_pos = np.diag(proto_pos[:unknown_id])  # shape (C,C)
P_cop = np.diag(proto_cop[:unknown_id])

# 4) OOD/NULL calibration sul test window
PMR_thr, H_thr, gap_thr, ood_q = ood_calibration(n_null=96, n_ood=192, dur=test_duration, gap=0*b.ms, thr_vec=thr_per_class_test)
# clamp minimo delle soglie OOD
PMR_thr = max(PMR_thr, 0.22)   # prima 0.20
H_thr   = max(H_thr,   1.00)   # prima 0.95
gap_thr = max(gap_thr, 0.18)   # prima 0.22
# Overshoot OOD rispetto alla soglia corrente del TEST window
overshoot = np.maximum(0.0, ood_q - thr_per_class_test[:unknown_id])
# smorza overshoot con radice e riduci i gain
overshoot = np.sqrt(overshoot)
heat_gain = np.array([0.01, 0.03, 0.00, 0.03, 0.03, 0.00, 0.00]) # per-class heat gain vector
margin = heat_gain * overshoot
# cap to boost
pos_cap = 0.85 * np.maximum(ema_pos_m1 * dur_scale, 1.0)
thr_per_class_test[:unknown_id] = np.minimum(thr_per_class_test[:unknown_id], pos_cap)
#thr_per_class_test[:unknown_id] += margin # add some heat only where needed
# extra margin to SALTY/SOUR because they are more "leaky"
thr_per_class_test[2] = max(1.0, thr_per_class_test[2] - 1.5)  # SALTY
thr_per_class_test[3] = max(1.0, thr_per_class_test[3] - 1.0)  # SOUR
thr_per_class_test[5] = max(1.0, thr_per_class_test[5] - 0.5)  # FATTY
thr_per_class_test[6] = max(1.0, thr_per_class_test[6] - 1.0)  # SPICY
# debug stamp
print("[DBG] OOD quantiles per-class:", {taste_map[isa]: float(ood_q[isa]) for isa in range(unknown_id)})
print("[DBG] Final test thresholds per-class:", {taste_map[isa]: float(thr_per_class_test[isa]) for isa in range(unknown_id)})

# safety clamp finale con i positivi (scalati al test window)
min_floor_test = max(5, int(min_spikes_for_known * dur_scale))
# safety cap 1: non superare mai l’85-90% del positivo atteso (scalato al test window)
for idx in range(unknown_id):
    pos_mu_t = float(ema_pos_m1[idx]) * dur_scale
    pos_sd_t = float(ema_sd(ema_pos_m1[idx], ema_pos_m2[idx])) * dur_scale
    # cap duro: non oltre il 75% del firing medio atteso
    cap_hard = max(min_floor_test, 0.75 * max(1.0, pos_mu_t))
    thr_per_class_test[idx] = min(thr_per_class_test[idx], cap_hard)
    # cap soft: media 70% + 0.2*sd
    cap_soft = max(min_floor_test, 0.70 * max(1.0, pos_mu_t)) + 0.20 * pos_sd_t
    thr_per_class_test[idx] = min(float(thr_per_class_test[idx]), cap_soft)
# safety cap 2: non superare mai l’85% del positivo atteso meno 1 deviazione standard (scalato al test window)    
for idx in range(num_tastes-1):
    pos_mu_t = float(ema_pos_m1[idx]) * dur_scale
    pos_sd_t = float(ema_sd(ema_pos_m1[idx], ema_pos_m2[idx])) * dur_scale
    cap_soft = max(float(min_floor_test), 0.70 * max(1.0, pos_mu_t))  # 0.85 -> 0.80
    cap_soft += 0.30 * pos_sd_t                                      # 0.35 -> 0.30
    thr_per_class_test[idx] = min(float(thr_per_class_test[idx]), cap_soft)

print(f"[Open-Set] PMR_thr={PMR_thr:.3f}  H_thr={H_thr:.3f}  gap_thr={gap_thr:.3f}")
print("Per-class thresholds (final, after OOD calib):",
      {taste_map[idx]: int(thr_per_class_test[idx]) for idx in range(num_tastes-1)})

# Population weights after training logs
print("Target weights after training (population summary):")
for k in range(unknown_id):
    st = pop_stats(k)  # k -> pop(k)
    if st is None:
        print(f"  {taste_map[k]}→{taste_map[k]}: [no synapses]")
    else:
        print(f"  {taste_map[k]}→{taste_map[k]}: "
              f"N={st['N']}, μ={st['mean']:.3f}, σ={st['std']:.3f}, "
              f"min={st['min']:.3f}, max={st['max']:.3f}")
        
print("\nMean weight by class pair p→q (population-aware):")
for p in range(unknown_id):
    row = []
    for q in range(unknown_id):
        idx = diag_indices_for_taste(p, q)
        if idx.size:
            row.append(f"{np.mean(S.w[idx]):.3f}")
        else:
            row.append("---")
    print(f"{taste_map[p]:>6}: " + "  ".join(row))

# weights copying before TEST
print("\nUnsupervised TEST phase with STDP frozen")
w_before_test = S.w[:].copy()
test_w_mon = b.StateMonitor(S, 'w', record=True)
net.add(test_w_mon)

# 11. Freezing STDP, homeostatis and input conductance
print("Freezing STDP for TEST phase...")
USE_GDI = True # Toggle between True or False during test phase to control GDI behaviour
# to manage baseline_hz noise during test phase
use_test_noise = False
USE_ATTENTIONAL_BIAS = False
test_baseline_hz = baseline_hz if use_test_noise else 0.0
# Neuromodulator parameters in test
k_inh_HI_test   = -0.08
k_inh_HT_test = 0.4
k_theta_HI_test = -0.5
k_ex_HI_test    = 0.15
k_noise_HI_test = 0.15
ht_pulse_aversion_test = 0.5
taste_neurons.v[:] = EL
taste_neurons.ge[:] = 0 * b.nS
taste_neurons.gi[:] = 0 * b.nS
taste_neurons.s[:]  = 0
taste_neurons.wfast[:] = 0 * b.mV
# Intrinsic homeostasis frozen
taste_neurons.homeo_on = 0.0
th = taste_neurons.theta[:]
# to manage Hedonic window
taste_neurons.taste_drive[:] = 0.0
taste_neurons.av_over[:]  = 0.0
taste_neurons.av_under[:] = 0.0
taste_neurons.theta[:] = np.clip((th/b.mV), float(theta_min/b.mV), float(theta_max/b.mV)) * b.mV
S.x[:] = 0
S.xbar[:] = 0
S.y[:] = 0
S.ybar[:] = 0
S.elig[:] = 0
S_unk.gain_unk = 0.15
# NO NORMALIZATION during test
col_allow_upscale    = False
col_scale_max        = 1.10
col_norm_every       = 999999
th = th - np.mean(th) # centered
theta_min, theta_max = -10*b.mV, 10*b.mV
inhibitory_S.g_step_inh = g_step_inh_local
inhibitory_S.delay = 0.5*b.ms
# apply state bias
apply_internal_state_bias(profile, mod, taste_neurons)

# Set da valutare con la pipeline di TEST:
snap = snapshot_state() # weights/states saved
EVAL_SET = "VAL"   # "VAL" per validazione, "TEST" per test finale

# which set to evaluate
_eval_map = {"VAL": val_stimuli, "TEST": test_stimuli}
if EVAL_SET not in _eval_map:
    raise ValueError("EVAL_SET must be 'VAL' or 'TEST'.")

# alias: la pipeline sotto usa la variabile `test_stimuli`
test_stimuli = _eval_map[EVAL_SET]
print(f"\n[INFO] Starting valutation pipeline on set: {EVAL_SET} "
      f"({len(test_stimuli)} trials)")

# 12. TEST PHASE
print("\nStarting TEST phase...")
S.stdp_on[:] = 0.0
# no depression during TEST for STP
S.tau_rec[:]   = 400*b.ms
S.u0[:]        = 0.06
S.ex_scale_stp[:] = 1.0 
results = []
# low ACh in test phase
ach_test_level = 0.35
k_ex_ACH = 0.20   # leggermente meno del train, ma non troppo basso
mod.ACH[:] = ach_test_level
pg_noise.rates = 0 * b.Hz
test_t0 = time.perf_counter()  # start stopwatch TEST

min_spikes_for_known_base = 4 # soglia minima assoluta per gusti noti nella fase di test
#min_spikes_for_known_test = max(3, int(np.ceil(NEURONS_PER_TASTE))) # population mode on
min_spikes_for_known_test = max(3, int(np.ceil(dur_scale * 0.6 * NEURONS_PER_TASTE)))
print(f"[Decoder] dur_scale={dur_scale:.2f} -> min_spikes_for_known_test={min_spikes_for_known_test}")
print("Scaled per-class thresholds:",
      {taste_map[idxs]: int(thr_per_class_test[idxs]) for idxs in range(num_tastes-1)})
recovery_between_trials = 100 * b.ms  # refractory recovery

# Per-class: final threshold = max(train-scaled, OOD-quantile, floor EMA)
for idx in range(num_tastes-1):
    # floor dal negativo EMA sul window di test
    neg_mu  = float(ema_neg_m1[idx]) * float(test_duration/training_duration)
    neg_sd  = float(ema_sd(ema_neg_m1[idx], ema_neg_m2[idx])) * float(test_duration/training_duration)
    thr_ema_test = neg_mu + k_sigma * neg_sd

    # always growing up -> not tearing down
    thr_per_class_test[idx] = max(
        float(thr_per_class_test[idx]),  # train-scaled
        float(thr_ema_test)              # EMA neg sul window di test
    )

# minimum clamp = min_spikes_for_known_test
thr_per_class_test[:unknown_id] = np.maximum(thr_per_class_test[:unknown_id],
                                             float(max(3, NEURONS_PER_TASTE)))

exact_hits = 0
total_test = len(test_stimuli)
n_known = unknown_id                     
H_unif = np.log(n_known) # uniform entropy
all_scores = []
all_targets = []
# hyperparameters
sep_min    = max(0.12, 0.25 / np.sqrt(n_noti))
abs_margin_test = 0.0 # to avoid margin during test
for step, (_rates_vec, true_ids, label) in enumerate(test_stimuli, start=1):
    # reset GDI
    if gdi_reset_each_trial:
        gdi_pool.x[:] = 0.0
    taste_neurons.ge[:] = 0 * b.nS
    taste_neurons.gi[:] = 0 * b.nS
    taste_neurons.wfast[:] = 0 * b.mV
    # initializing SPICY during test
    sl_spice = taste_slice(spicy_id)
    taste_neurons.thr_spice[sl_spice] = taste_neurons.thr0_spice[sl_spice]  # reset per trial
    taste_neurons.spice_drive[sl_spice] = 0.0
    taste_neurons.a_spice[sl_spice]     = 0.0
    did_unsup_relabel = False
    unsup_log = None
    # progress bar + chrono + ETA
    frac   = step / total_test
    filled = int(frac * progress_bar_len)
    bar    = '�'*filled + '�'*(progress_bar_len - filled)

    elapsed = time.perf_counter() - test_t0
    eta = (elapsed/frac - elapsed) if frac > 0 else 0.0

    msg = (
      f"[{bar}] {int(frac*100)}% | Step {step}/{total_test} | Testing � {label}"
      f" | t={fmt_mmss(elapsed)} | ETA={fmt_mmss(eta)}"
    )
    pbar_update(msg)

    # deciding with "active" or "off" if there's need to applicate emotion test or not
    if test_emotion_mode != "off":
        DA_now = float(mod.DA_f[0] + k_tonic_DA * mod.DA_t[0])
        HT_now = float(mod.HT[0])
        NE_now = float(mod.NE[0])
        HI_now = float(mod.HI[0])
        ACH_now = float(mod.ACH[0]); 
        GABA_now = float(mod.GABA[0])
        # threshold
        taste_neurons.theta_bias[:] = (k_theta_HT * HT_now + k_theta_HI_test * HI_now) * b.mV
        # synaptic gain and inhibition as in training phase
        S.ex_scale = (1.0 + k_ex_NE * NE_now) * (1.0 + k_ex_HI_test * HI_now) * (1.0 + k_ex_ACH * ACH_now)
        _inh = 1.0 + k_inh_HT * HT_now + k_inh_NE * NE_now + k_inh_HI_test * HI_now + k_inh_GABA * GABA_now
        inhibitory_S.inh_scale = max(0.3, _inh) # same clamp to avoid explosions
        # noise
        ne_noise_scale = max(0.05, 1.0 - k_noise_NE * NE_now)
        ach_noise_scale = max(0.05, 1.0 - k_noise_ACH * ACH_now)
        pg_noise.rates = test_baseline_hz * ne_noise_scale * (1.0 + k_noise_HI_test * HI_now) * ach_noise_scale * np.ones(num_tastes) * b.Hz
    else:
        # neutral profile -> no modulators
        mod.DA_f[:] = mod.DA_t[:] = mod.HT[:] = mod.NE[:] = mod.HI[:] = 0.0
        inhibitory_S.inh_scale = 0.45
        S.ex_scale = 1.0
        pg_noise.rates = test_baseline_hz * np.ones(num_tastes) * b.Hz
        taste_neurons.theta_bias[:] = 0 * b.mV

    if len(true_ids) == 1 and true_ids[0] == unknown_id:
        # OOD/NULL � no normalization
        set_test_stimulus(_rates_vec)
    else:
        if USE_GDI:
            set_test_stimulus(_rates_vec)
        else:
            set_stimulus_vect_norm(_rates_vec, total_rate=BASE_RATE_PER_CLASS * len(true_ids), include_unknown=False)

    # initializing the rewarding for GDI
    # divisive gain test-time proporzionale all'energia di input (proxy: somma rates noti)
    input_energy = float(np.sum(_rates_vec[:unknown_id]))
    ref_rate = float(BASE_RATE_PER_CLASS)

    inp = np.asarray(_rates_vec[:unknown_id], dtype=float)
    pmr_in = (inp.max() / (inp.sum() + 1e-9)) if inp.sum() > 0 else 0.0

    cap_base   = 0.45
    cap_boost  = float(np.interp(pmr_in, [0.25, 0.45], [0.95, 0.60]))     # più diffuso -> più cap
    cap_energy = float(np.interp(input_energy / (ref_rate + 1e-9), [0.5, 2.0], [cap_base, 0.90]))
    cap        = min(cap_boost, cap_energy)

    gamma_val  = float(np.clip(gamma_gdi_0, 0.08, cap))
    S.gamma_gdi = gamma_val
    S_noise.gamma_gdi = gamma_val
    # 2) spikes counting during trials
    prev_counts = spike_mon.count[:].copy()
    net.run(test_duration)
    diff_counts = spike_mon.count[:] - prev_counts
    # after the stimulation to avoid zombie UNKNOWN gain during next trials
    S_unk.gain_unk = 0.0

    # inject 5-HT for the generic TASTE aversion
    known = slice(0, taste_slice(unknown_id).start)
    drv = np.array(taste_neurons.taste_drive[known])
    thr = np.array(taste_neurons.thr_hi[known])
    if (drv > thr).any():
        mod.HT[:] += 0.20

    # inject 5-HT for the SPICY aversion
    drv_now = float(np.mean(taste_neurons.spice_drive[taste_slice(spicy_id)]))
    thr_now = float(np.mean(taste_neurons.thr_spice[taste_slice(spicy_id)]))
    if drv_now > thr_now:
        mod.HT[:] += 0.25

    # 3) take the winners using per-class thresholds
    # strong decision with OOD and NULL
    scores = population_scores_from_counts(diff_counts) # neurons population management
    scores[unknown_id] = -1e9
    mx = scores.max()

    all_scores.append(scores[:unknown_id].astype(float))
    tgt = np.zeros(num_tastes-1, dtype=int)
    for tid in true_ids:
        if tid != unknown_id:
            tgt[tid] = 1
    all_targets.append(tgt)
 
    # estensione on-demand della finestra:
    '''If the trial is is_mixture_like but the co-tastes are a few spikes below your soft-abs,
    extend the test window by +50…80 ms for that trial only and recount. 
    This way, you can take advantage of the extra latency to bring out the co-tastes, 
    without affecting the overall OOD calibration.'''
    sorted_idx = np.argsort(scores)[::-1]
    top_idx = int(sorted_idx[0])
    top = float(scores[top_idx])
    second = float(scores[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
    sep = (top - second) / (top + 1e-9)  # relative separation
    pos_expect_test = np.maximum(ema_pos_m1 * float(test_duration / training_duration), 1e-6)

    # Entropy distribution
    E_local    = float(np.sum(scores[:unknown_id]))
    p_local    = scores[:unknown_id].astype(float)
    p_local    = p_local / (E_local + 1e-12)
    if E_local > 0:
        p_local /= E_local
    else:
        p_local[:] = 0.0
    H          = float(-(p_local * np.log(p_local + 1e-12)).sum())
    PMR        = top / (E_local + 1e-9)
    gap        = sep
    z = scores[:unknown_id] / np.maximum(pos_expect_test, 1.0)

    # GABA little decay
    if (E >= 3.0*min_spikes_for_known_test) and (gap < 0.14):
        mod.GABA[:] += 0.3 * gaba_pulse_stabilize

    # decide mix-like con questi valori "base"
    mix_pmr_lo, mix_pmr_hi = 0.26, 0.72
    mix_H_lo,   mix_H_hi   = 0.50, 1.85
    is_mixture_like = (mix_pmr_lo <= PMR <= mix_pmr_hi) and (mix_H_lo <= H <= mix_H_hi)
    did_extend = False

    # se è mix
    if is_mixture_like:
        # soft-abs che usi già per i mix
        soft_abs = np.maximum(mix_abs_pos_frac * np.maximum(pos_expect_test, 1.0),
                          float(min_spikes_for_known_test))  # vedi tuo calcolo preesistente

        # “borderline” = vicino alla soft-abs (entro il 20%)
        borderline = []
        for idxs in np.argsort(scores[:unknown_id])[::-1][:3]:  # controlla i top-3
            if scores[idxs] >= 0.8*soft_abs[idxs] and scores[idxs] < soft_abs[idxs]:
                borderline.append(idxs)

        # if the current taste is a bit upper the threshold
        if borderline:
            prev = spike_mon.count[:].copy()
            # ri-lancia lo stesso stimolo SENZA boost per +8s0 ms
            extra_b = 80 * b.ms
            # stimolo corrente è ancora attivo, allunghiamo la finestra di esecuzione di poco
            net.run(extra_b)
            diff_extra = (spike_mon.count[:] - prev).astype(float)
            scores = add_extra_scores(scores, diff_extra, unknown_id, reduce="sum") # scores updated with population
            did_extend = True

            # eventuale estensione e ricalcolo (solo se mix-like in caso borderline)
            sorted_idx = np.argsort(scores)[::-1]
            top_idx    = int(sorted_idx[0])
            top        = float(scores[top_idx])
            second     = float(scores[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
            sep        = (top - second) / (top + 1e-9)

            E   = float(np.sum(scores[:unknown_id]))
            p   = scores[:unknown_id].astype(float)
            if E > 0:
                p /= E
            else:
                p[:] = 0.0
            H   = float(-(p*np.log(p + 1e-12)).sum())
            PMR = top / (E + 1e-9)
            gap = sep
        
    # intervalli stretti mantengono l'open-set. Sono centrati sui dati attuali
    k_active = int(np.sum(scores[:unknown_id] >= min_spikes_for_known_test))
    n_strong = np.sum(scores[:unknown_id] >= min_spikes_for_known_test)
    is_mixture_like = is_mixture_like and (n_strong >= 2)

    # regola di rifiuto open-set (usa soglie sopra)
    gap_dyn = gap_thr
    if k_active in (2,3):
        gap_dyn = max(0.10, 0.75 * gap_thr)   # 0.18 → ~0.15

    # dopo aver calcolato PMR,H
    flags = [
        PMR is not None and PMR < PMR_thr,
        H   is not None and H   > H_thr,
        gap is not None and gap < gap_dyn,
    ]
    is_diffuse = sum(bool(x) for x in flags) >= 2

    # 6) Politica UNICA per l’inibizione e GDI in questo trial (niente doppi set)
    # base: usa il valore corrente come “ancora”
    inh = float(inhibitory_S.inh_scale[0]) if hasattr(inhibitory_S, 'inh_scale') else 0.45
    # più co-attivi => un po' più WTA
    if k_active >= 3:
        inh *= min(1.40, 1.0 + 0.12*(k_active - 2))
    # se diffuso, spingi ancora
    if is_diffuse:
        inh *= 1.12
    inhibitory_S.inh_scale[:] = float(np.clip(inh, 0.30, 1.40))

    # 7) Gating UNKNOWN “forte”: se il top ha z alto, spegni UNKNOWN (dopo eventuale estensione)
    # stimulus on target classes with UNKNOWN inputs
    set_unknown_gate(
        pmr=PMR, gap=gap, H=H,
        PMR_thr=0.58,           # 0.56–0.62
        gap_thr=0.14,           # 0.12–0.18
        H_thr=0.80*np.log(unknown_id)
    )
    if z[top_idx] >= 0.90:
        S_unk.gain_unk = 0.0

    # blend dinamico (0=solo P_pos, 1=solo P_cop)
    # più "mix-like" (PMR medio e H alto) ⇒ più peso a P_cop
    lam_blend = float(np.clip(np.interp(PMR, [0.30, 0.55], [0.0, 1.0]) * np.interp(H, [0.7, 1.5], [0.0, 1.0]), 0.0, 1.0))
    P_blend = (1.0 - lam_blend) * P_pos + lam_blend * P_cop
    
    # corsia soft per coppie: se i 2 top superano un assoluto “ragionevole”, accetta anche con gap basso
    if is_diffuse and is_mixture_like and k_active == 2:
        top2 = int(sorted_idx[1])
        abs_soft = max(min_spikes_for_known_test, 0.30 * (pos_expect_test[top_idx] + pos_expect_test[top2]) / 2.0)
        if scores[top_idx] >= abs_soft and scores[top2] >= abs_soft*0.88:
            # consentire il mix binario nonostante diffuso "borderline"
            winners = sorted([top_idx, top2], key=lambda ids: scores[ids], reverse=True)
            is_diffuse = False

    # expected cop value
    cop_expect_test = np.maximum(ema_cop_m1 * float(test_duration / training_duration), 1.0)

    # mixture_shortcut -> solo se mix-like e NON diffuso 
    mixture_shortcut = [] 
    if is_mixture_like and (not is_diffuse): 
        # soglia assoluta morbida proporzionale all'atteso positivo 
        soft_abs = np.maximum(mix_abs_pos_frac * np.maximum(pos_expect_test, 1.0), 
                          float(min_spikes_for_known_test)) # guardia relativa al top per evitare code spurie 
        rel_guard = norm_rel_ratio_test * top 
        for idx in range(unknown_id): 
            if (scores[idx] >= max(soft_abs[idx], rel_guard)) and (z[idx] >= max(0.18, z_rel_min)): 
                mixture_shortcut.append(idx) 
                z_min = (z_min_mix if (is_mixture_like and not is_diffuse) else z_min_base) if (not is_diffuse) else 0.20 
                if (k_active >= 3) and (PMR < max(0.65, 1.05*PMR_thr)): 
                    is_diffuse = True

    # NNLS Unsupervised Learning - Labeling Discovery
    winners = []
    did_unsup_relabel = False
    unsup_labels = []
    unsup_log = None
    # criteri per consentire quintuple:
    may_allow_k5 = (
        is_mixture_like and
        (gap is not None and gap_thr is not None and gap >= 1.00*gap_thr or
        PMR is not None and PMR_thr is not None and PMR >= 1.10*PMR_thr)
    )

    # Gestione NNLS dell'Unsupervised Learning
    should_try_nnls = ((winners == []) or (winners == [unknown_id])) #and (not is_diffuse)
    if should_try_nnls:
        # NNLS decoding before saying: "UNKNOWN" a priori
        abs_floor_vec = np.maximum(  # “soft-abs” per non far crollare i co-gusti
            0.22*pos_expect_test,
            0.18*cop_expect_test
        )
        cand_nnls, info_nnls = decode_by_nnls(
            scores=scores[:unknown_id],
            P_pos=P_pos, P_cop=P_cop,
            is_mixture_like=is_mixture_like,
            z_scores=z[:unknown_id],
            frac_thr=0.10,                     # 0.12–0.15
            z_min_guard=max(0.01, z_rel_min),
            allow_k4=True,
            allow_k5=may_allow_k5,
            gap=gap, gap_thr=gap_thr,
            pmr=PMR, pmr_thr=PMR_thr,
            abs_floor=abs_floor_vec,        
            nnls_iters=250, nnls_lr=None, l1_cap=1.0
        )

        if cand_nnls:
            # ordina preferendo score reale, poi peso NNLS
            w_nnls = info_nnls.get("w", None)
            if w_nnls is not None:
                cand_nnls = sorted(cand_nnls, key=lambda isa: (scores[isa], w_nnls[isa]), reverse=True)
            else:
                cand_nnls = sorted(cand_nnls, key=lambda isa: scores[isa], reverse=True)

            # cap finale: permetti fino a 5 solo se consentito globalmente
            allow_k5_globally = True  # to allow 5-mix
            k_cap_eff = min(k_cap, 5 if (allow_k5_globally and may_allow_k5) else 4)
            winners = cand_nnls[:k_cap_eff]

            did_unsup_relabel = True
            unsup_labels = [taste_map[isa] for isa in winners]
            unsup_log = dict(
                stage="nnls",
                err=float(info_nnls["err"]),
                cover=float(info_nnls["cover"]),
                k_abs=int(info_nnls["k_abs"]),
                k_est=int(info_nnls["k_est"]),
                thr_rel=float(info_nnls["thr_rel"])
            )

    # veto rapido su molti attivi diffusi senza considerare falsi allarmi
    nnls_k = len(winners) if (did_unsup_relabel and winners and winners[0] != unknown_id) else 0
    too_many_active_diffuse = (
        # blocca solo se > 5
        (k_active > 5) and
        # e la distribuzione non giustifica mix complessi
        ((PMR < 1.03*PMR_thr) or (H > 0.95*H_thr)) and
        # e non abbiamo un segnale affidabile di quintupla
        (not may_allow_k5) and
        # e NNLS non ha già “validato” (>=5) in modo controllato
        (nnls_k < 5)
    )
    if too_many_active_diffuse:
        winners = [unknown_id]
        # passa direttamente alla chiusura del trial:
        expected  = [taste_map[idxs] for idxs in true_ids]
        predicted = [taste_map[wx] for wx in winners]
        hit = set(winners) == set(true_ids)
        print(f"\n{label}\n  expected:   {expected}\n  predicted: {predicted}\n  exact:   {hit}")
        results.append((label, expected, predicted, hit, ""))
        net.run(recovery_between_trials)
        continue

    # pulizia del trial: blending factor 0..1 (0 = tutto su EMA+, 1 = tutto su soglia dura)
    # aumentiamo il blending (=> soglia più dura) se il trial è più ambiguo/diffuso
    clean_pmr = np.clip((PMR - PMR_thr) / max(1e-9, 0.85 - PMR_thr), 0.0, 1.0)  # PMR rispetto alla soglia OOD
    clean_gap = np.clip((gap - gap_thr) / max(1e-9, 0.50 - gap_thr), 0.0, 1.0)  # separazione relativa
    clean = 0.5*clean_pmr + 0.5*clean_gap

    # blending: più peso al co-positive se siamo in un mix pulito
    # clean  [0,1] (già calcolato): più è alto, più il trial è "pulito"
    mix_blend = np.interp(clean, [0.0, 1.0], [0.55, 0.95])  # 0.90 � 0.95
    frac_pos  = np.interp(clean, [0.0, 1.0], [0.68, 0.28])  # 0.34 � 0.30
    pos_exp_blend = (1.0 - mix_blend) * pos_expect_test + mix_blend * cop_expect_test

    # quanta frazione dell'aspettativa positiva usiamo come soglia "morbida"
    # - se cleanH1 (trial pulito) è frazione bassa (~0.60) -> più facile accettare noti
    # - se cleanH0 (trial ambiguo) è frazione alta (~0.90) -> più vicina alla soglia dura
    mix_min_abs = (
        max(4, int(0.6*min_spikes_for_known_test))
        if is_mixture_like and (not is_diffuse) else
        min_spikes_for_known_test
    )

    # rendi un filo più tenera la soglia sui mix
    if is_mixture_like and (not is_diffuse):
        frac_pos = max(0.38, frac_pos - 0.08)
        if (k_active in (2, 3)) and (mix_pmr_lo <= PMR <= mix_pmr_hi):
           frac_pos = max(0.34, frac_pos - 0.06)

    # soglia efficace per ogni classe nota
    # - se il trial è pulito, usiamo un blend di soglia dura e soglia morbida
    # - se il trial è diffuso, usiamo la soglia dura (OOD-based)
    if not is_diffuse:
        thr_eff = np.minimum(thr_per_class_test[:unknown_id], frac_pos * pos_exp_blend)
        thr_eff = np.maximum(
            thr_eff,
            np.maximum(float(mix_min_abs), dyn_abs_min_frac * np.maximum(pos_expect_test, 1.0))
        )
    else:
        thr_eff = np.maximum(thr_per_class_test[:unknown_id], float(mix_min_abs))
    
    # data-driven thresholds
    top_pass_strict = (
        (scores[top_idx] >= 0.95 * thr_per_class_test[top_idx]) and
        (z[top_idx]     >= z_min) and
        (gap            >= 0.95 * gap_thr)
    )

    # 6) Penalità “classi calde” SOLO in diffuso (overshoot/neg_sd già disponibili dalla calibrazione)
    if is_diffuse:
        neg_sd = np.array([ema_sd(ema_neg_m1[isd], ema_neg_m2[isd]) for isd in range(unknown_id)])
        heat = np.clip((overshoot / (neg_sd + 1e-6)), 0.0, 4.0)
        thr_eff[:unknown_id] *= (1.0 + 0.45 * heat)
        # guardia extra su FATTY come nel tuo codice
        if top_idx == fatty_id and (z[fatty_id] < 1.45 or PMR < 0.66):
            winners = [unknown_id]

    # 7) Stima locale di k_est (da HHI), poi k_cap
    HHI = float((p_local**2).sum()) if E_local > 0 else 1.0
    k_est = int(np.clip(round(1.0 / max(HHI, 1e-9)), 1, unknown_id))

    k_cap_base = int(np.clip(k_est, 2, n_noti))
    bonus = 0 if k_active < 2 else (2 if clean >= 0.70 else (1 if clean >= 0.60 else 0))
    k_cap = min(max(k_cap_base + bonus, 3), k_active, 9)  # ≥3 quando il mix è pulito

    #  WINNERS SELECTION
    # A) Rejection diretto se diffuso e top non “strict”
    if is_diffuse and not top_pass_strict:
        winners = [unknown_id]
    else:
        # B) candidati "stretti": soglia assoluta vera + z
        strict_winners = [
            idx for idx in range(unknown_id)
            if (scores[idx] >= (thr_eff[idx] + abs_margin_test)) and (z[idx] >= z_min)
        ]
        winners = list(strict_winners)

         # C) Mixture shortcut (solo top forte e NON diffuso)
        top_known = (scores[top_idx] >= thr_eff[top_idx]) and (sep >= gap_thr)
        if top_known and (not is_diffuse) and mixture_shortcut:
            add = [c for c in mixture_shortcut if c != top_idx and c not in winners]
            if add:
                add.sort(key=lambda ia: scores[ia], reverse=True)
                add = add[:max(0, k_cap - 1)]
                # guardrail energetico
                if scores[add].sum() >= 0.20 * E:
                    winners.extend(add)
        
        # D) Negative rel gate
        if use_rel_gate_in_test and top_known and (not is_diffuse):
            rel_thr = rel_gate_ratio_test * top
            co_abs_soft = np.maximum(0.32 * thr_eff, 0.30 * cop_expect_test)
            co_abs_cap = 1.05 * rel_thr
            co_abs = np.minimum(co_abs_soft, co_abs_cap)

            E_abs = E
            co_abs_energy_min = np.clip(0.05 * E_abs, 2.0, 10.0)
            nrel = scores[:unknown_id] / (top + 1e-9)

            cand = [idx for idx in range(unknown_id)
                if (idx != top_idx) and (z[idx] >= z_rel_min) and
                   ((nrel[idx] >= norm_rel_ratio_test and scores[idx] >= min_norm_abs_spikes) or
                    (scores[idx] >= max(rel_thr, co_abs[idx]))) and
                   (scores[idx] >= co_abs_energy_min)]
            cand.sort(key=lambda idx: scores[idx], reverse=True)

            add_max = max(0, k_cap - 1)  # top già incluso
            add = cand[:min(add_max, len(cand))]
            if add and scores[add].sum() >= 0.22 * E:
                winners.extend(add)

            # Weak co-taste rescue (solo se NON diffuso e top � forte)
            if top_pass_strict:
                rescue = []
                abs_co_min = 0.25 * min_spikes_for_known_test      # co-gusto davvero debole ma reale
                rel_min    = 0.05                                  # quota minima vs top
                z_min_resc = 0.20                                  # z minimo del co-gusto

                for idx in range(unknown_id):
                    if idx == top_idx or (idx in winners):
                        continue
                    if (z[idx] >= z_min_resc and
                        scores[idx] >= abs_co_min and
                        (scores[idx] / (top + 1e-9) >= rel_min)):
                        rescue.append(idx)

                # tieni al massimo 1 co-gusto extra, e solo se contribuisce davvero
                rescue.sort(key=lambda js: scores[js], reverse=True)
                if rescue and scores[rescue[0]] >= max(0.18 * E, 2.0):
                    winners.append(rescue[0])

            # Top-only fallback severo in scene borderline
            if (k_active >= 3) and (clean < 0.35):
                winners = [top_idx] if top_pass_strict else [unknown_id]  # 0.35->0.40
            
            # NEW: cap finale sul numero totale di vincitori (mantieni il top)
            if len(winners) > k_cap:
                # ordina per score decrescente e preserva il top
                rest = [isa for isa in winners if isa != top_idx]
                rest.sort(key=lambda isa: scores[isa], reverse=True)
                winners = [top_idx] + rest[:max(0, k_cap-1)]

        # F) Fallback se ancora vuoto
        if not winners:
            if (not is_diffuse) and top_pass_strict and (z[top_idx] >= z_min):
                winners = [top_idx]
            else:
                winners = [unknown_id]

        # SPICY aversion check after TEST winners
        # is SPICY present in the ground truth or in the winners?
        # If so, trigger the aversion response
        is_spicy_present = (spicy_id in true_ids) or (spicy_id in winners)

        if is_spicy_present: # trigger spicy aversion
            happened, p_now = spicy_aversion_triggered( # returns bool, p_now
                taste_neurons, mod, spicy_id,
                p_base=p_aversion_base,
                slope=p_aversion_slope,
                cap=p_aversion_cap,
                trait=profile['spicy_aversion_trait'],
                k_hun=profile['k_hun_spice'],
                k_h2o=profile['k_h2o_spice']
            )
            # apply spicy aversion effects
            if happened:
                mod.HT[:]  += ht_pulse_aversion
                mod.DA_f[:] *= da_penalty_avers
                mod.DA_t[:] *= da_penalty_avers
                sl = taste_slice(spicy_id)
                taste_neurons.thr_spice[sl] = taste_neurons.thr_spice[sl] + thr_spice_kick
                if verbose_rewards:
                    print(f"[SPICY-AVERSION] p={p_now:.2f} → HT+{ht_pulse_aversion}, DA×{da_penalty_avers}, thr_spice+={thr_spice_kick:.3f}")

        # debug prints
        print(f"\n[DBG] decision: PMR={PMR:.3f} H={H:.3f} gap={gap:.3f} top={taste_map[top_idx]} "
            f"score_top={scores[top_idx]:.1f} thr_eff_top={thr_eff[top_idx]:.1f} z_top={z[top_idx]:.2f}")
        print(f"[DBG] thr_eff[:]= {[round(float(x),1) for x in thr_eff]}")
        print(f"[DBG] z[:]= {[round(float(x),2) for x in z]}")
        print(f"[DBG] is_diffuse={is_diffuse} is_mixture_like={is_mixture_like} k_active={k_active} k_cap={k_cap}")
        if winners == [unknown_id]:
            print(f"[REJECT] PMR={PMR:.3f}<{PMR_thr:.3f}? {PMR<PMR_thr} | H={H:.3f}>{H_thr:.3f}? {H>H_thr} "
            f"| gap={gap:.3f}<{gap_thr:.3f}? {gap<gap_thr} | top={taste_map[top_idx]} "
            f"score_top={scores[top_idx]:.1f} thr_eff_top={thr_eff[top_idx]:.1f} z_top={z[top_idx]:.2f}")

    order = np.argsort(scores)
    dbg = [(taste_map[idxs], int(scores[idxs])) for idxs in order[::-1]]
    print("\nTest scores:", dbg)

    # burst NE on ambiguities
    ambiguous = (second > 0 and top / (second + 1e-9) < 1.3) or (len(winners) > 2)
    if test_emotion_mode == "active" and (ambiguous or is_diffuse):
        mod.NE[:] += ne_pulse_amb
        mod.HI[:] += 0.5 * hi_pulse_nov
        inhibitory_S.inh_scale[:] = np.maximum(0.8, inhibitory_S.inh_scale[:] * 1.05) # if a class is weak, is not going to be suppressed in general cases except for ambiguous cases

    # Emotional burst in test phase
    if test_emotion_mode == "active":
        if set(winners) == set(true_ids):
            mod.DA_f[:] += da_pulse_reward    # phasic burst
            mod.DA_t[:] += da_tonic_tail      # tonic tail
        has_strong_fp = any(
            (idx not in true_ids) and (float(scores[idx]) >= fp_gate[idx])
            for idx in range(num_tastes-1)
        )
        if has_strong_fp:
            mod.HT[:] += ht_pulse_fp # prudence or fear increased

        if set(winners) != set(true_ids):  # miss
            mod.HI[:] += 0.5 * hi_pulse_miss
    
    if test_emotion_mode != "off":
        DA_disp = float(mod.DA_f[0] + k_tonic_DA * mod.DA_t[0])
        msg += (f" | DA={DA_disp:.2f} (f={float(mod.DA_f[0]):.2f},t={float(mod.DA_t[0]):.2f})"
            f" HT={float(mod.HT[0]):.2f} NE={float(mod.NE[0]):.2f} HI={float(mod.HI[0]):.2f}"
            f" ACH={float(mod.ACH[0]):.2f} GABA={float(mod.GABA[0]):.2f}")
    pbar_update(msg)

    # to make a confrontation: expected vs predicted values
    expected  = [taste_map[idxs] for idxs in true_ids]
    predicted = [taste_map[w] for w in winners]
    hit = set(winners) == set(true_ids)

    extra = ""
    if did_unsup_relabel:
        extra = f"[UNSUP@TEST accettato {unsup_labels} | {unsup_log}]"
    elif winners == [unknown_id] and (not is_diffuse):
        extra = "[UNSUP rifiutato]"

    # output visualization
    print(f"\n{label}\n  expected:   {expected}\n  predicted: {predicted}\n  exact:   {hit}")
    #results.append((label, expected, predicted, hit))
    note = unsup_log if (did_unsup_relabel and unsup_log is not None) else ""
    results.append((label, expected, predicted, hit, note))
    # 4) final valutation
    if set(winners) == set(true_ids):
        exact_hits += 1
    # 5) final refractory period after trial
    net.run(recovery_between_trials)

# unfreezing intrinsic homeostasis
taste_neurons.homeo_on = 1.0

# ending test debug
hot_fp = {taste_map[isa]: int(sum((lab.startswith("OOD") or lab.startswith("NULL")) and (taste_map[isa] in pred)
        for lab,_,pred,_,_ in results))
        for isa in range(unknown_id)}
print("[DBG] FP su OOD/NULL per classe:", hot_fp)

# clean the bar
pbar_done()
print(f"\nTEST phase done (elapsed: {fmt_mmss(time.perf_counter()-test_t0)})")

# Print summary
print("\n===== SUMMARY =====")
known_trials = [ra for ra in results if all(xs != "UNKNOWN" for xs in ra[1])]
unknown_trials = [ra for ra in results if ra[1] == ["UNKNOWN"]]
acc_overall = 100.0 * sum(1 for ra in results if ra[3]) / max(1, len(results))
acc_known   = 100.0 * sum(1 for ra in known_trials if ra[3]) / max(1, len(known_trials))
rej_rate    = 100.0 * len(unknown_trials) / max(1, len(results))
print(f"Total trials:           {len(results)}")
print(f"Known-only trials:      {len(known_trials)}")
print(f"NULL/OOD trials:        {len(unknown_trials)}")
print(f"Exact match (overall):  {acc_overall:.2f}%")
print(f"Exact match (known):    {acc_known:.2f}%")
print(f"UNKNOWN rejection rate: {rej_rate:.2f}%")
pred_flat = []
tgt_flat = []
for _, tgt, pred, _, _ in results:
    for ts in tgt:
        if ts != "UNKNOWN":
            tgt_flat.append(ts)
    for p in pred:
        pred_flat.append(p)
# somewhat predicted counters
print("\nPredicted counts:", dict(Counter(pred_flat)))
print("Target counts:   ", dict(Counter(tgt_flat)))
print("\n")

# salva pesi post-test e ripristina stato salvato
w_after_test = S.w[:].copy()
restore_state(snap)  # se vuoi riportarti allo snapshot pre-test

# Metrics classification report with Jaccard class and confusion matrix
# a. Test accuracy
ok = 0
for row in results:
    if len(row) == 5:
        label, exp, pred, hit, note = row
    else:
        label, exp, pred, hit = row
        note = ""
    status = "OK" if hit else "MISS"
    suffix = f" | {note}" if note else ""
    print(f"{label:26s} | expected={exp} | predicted={pred} | {status}{suffix}")
    ok += int(hit)
print(f"\nTest accuracy (exact-set match): {ok}/{len(results)} = {ok/len(results):.2%}")

# b. Jaccard class, recall, precision, f1-score
label_to_id = {lbl: idx for idx, lbl in taste_map.items()}
classes = [idx for idx in range(num_tastes) if idx != unknown_id]
# Class counters
stats = {idx: {'tp': 0, 'fp': 0, 'fn': 0} for idx in classes}
jaccard_per_case = []
for row in results:
    exp_labels = row[1]
    pred_labels = row[2]
    T = {label_to_id[lbl] for lbl in exp_labels if label_to_id[lbl] != unknown_id}
    P = {label_to_id[lbl] for lbl in pred_labels if label_to_id[lbl] != unknown_id}
    # expected vs predicted
    inter = T & P
    union = T | P
    jaccard_per_case.append(len(inter) / len(union) if len(union) > 0 else 1.0)

    for c in classes:
        if c in P and c in T:
            stats[c]['tp'] += 1
        elif c in P and c not in T:
            stats[c]['fp'] += 1
        elif c not in P and c in T:
            stats[c]['fn'] += 1

# Metric helpers
def prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1   = (2*prec*rec/(prec+rec)) if np.isfinite(prec) and np.isfinite(rec) and (prec+rec) > 0 else float('nan')
    iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')  # Jaccard per class
    return prec, rec, f1, iou

def fmt_pct(x):
    return "" if not np.isfinite(x) else f"{x*100:5.1f}%"

# Confusion matrix
print("\nMulti-label metrics for every taste:")
for c in classes:
    tp, fp, fn = stats[c]['tp'], stats[c]['fp'], stats[c]['fn']
    prec, rec, f1, iou = prf(tp, fp, fn)
    print(f"{taste_map[c]:>6s}: TP={tp:2d} FP={fp:2d} FN={fn:2d} | "
          f"P={fmt_pct(prec)} R={fmt_pct(rec)} F1={fmt_pct(f1)} IoU={fmt_pct(iou)}")

# Micro / Macro
sum_tp = sum(d['tp'] for d in stats.values())
sum_fp = sum(d['fp'] for d in stats.values())
sum_fn = sum(d['fn'] for d in stats.values())

micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else float('nan')
micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else float('nan')
micro_f1 = (2*micro_p*micro_r/(micro_p+micro_r)) if np.isfinite(micro_p) and np.isfinite(micro_r) and (micro_p+micro_r) > 0 else float('nan')

per_class_prec = [prf(stats[c]['tp'], stats[c]['fp'], stats[c]['fn'])[0] for c in classes]
per_class_rec  = [prf(stats[c]['tp'], stats[c]['fp'], stats[c]['fn'])[1] for c in classes]
per_class_f1   = [prf(stats[c]['tp'], stats[c]['fp'], stats[c]['fn'])[2] for c in classes]
per_class_iou  = [prf(stats[c]['tp'], stats[c]['fp'], stats[c]['fn'])[3] for c in classes]

macro_p = float(np.nanmean(per_class_prec)) if len(per_class_prec) else float('nan')
macro_r = float(np.nanmean(per_class_rec))  if len(per_class_rec)  else float('nan')
macro_f1 = float(np.nanmean(per_class_f1))  if len(per_class_f1)  else float('nan')
mean_iou = float(np.nanmean(per_class_iou)) if len(per_class_iou) else float('nan')

print("\n Micro/Macro ")
print(f"Micro  -> P={fmt_pct(micro_p)} R={fmt_pct(micro_r)} F1={fmt_pct(micro_f1)}")
print(f"Macro  -> P={fmt_pct(macro_p)} R={fmt_pct(macro_r)} F1={fmt_pct(macro_f1)}")
print(f"Mean IoU (per-class): {fmt_pct(mean_iou)}")

# Jaccard per test-case (expected vs predicted)
if jaccard_per_case:
    mean_jaccard_cases = float(np.mean(jaccard_per_case))
    print("\nJaccard per test-case:", [f"{j:.2f}" for j in jaccard_per_case])
    print(f"Average Jaccard (set vs set): {mean_jaccard_cases:.2f}")

# PR-/ROC-AUC management
if len(all_scores) > 0 and len(all_targets) > 0:
    S_scores = np.vstack(all_scores)   # shape: (N_trials, C)
    Y_scores = np.vstack(all_targets)  # shape: (N_trials, C)
    # class support pos/neg and AP baseline H pos/N
    print("\n- Supports and AP baseline for every class -")
    for c in range(Y_scores.shape[1]):
        n_pos = int(Y_scores[:, c].sum())
        n_neg = int(Y_scores.shape[0] - n_pos)
        base_ap = (n_pos / (n_pos + n_neg)) if (n_pos + n_neg) > 0 else float('nan')
        print(f"{taste_map[c]:>6s}: support +={n_pos}, -={n_neg}, baseline APH{base_ap:.3f}")

    def roc_points(y, s):
        # y  {0,1}, s = continual scores
        order = np.argsort(-s)
        y = y[order]
        P = int(y.sum()); N = len(y) - P
        tps = np.cumsum(y)
        fps = np.cumsum(1 - y)
        TPR = tps / (P if P > 0 else 1)
        FPR = fps / (N if N > 0 else 1)
        # key points
        TPR = np.concatenate(([0.0], TPR, [1.0]))
        FPR = np.concatenate(([0.0], FPR, [1.0]))
        auc = float(trapezoid(TPR, FPR))
        return FPR, TPR, auc

    def pr_auc(y, s):
        y = np.asarray(y, dtype=int)
        s = np.asarray(s, dtype=float)
        P = int(y.sum())
        N = len(y) - P
        if P == 0 or N == 0:
            return float('nan')
        order = np.argsort(-s)
        y = y[order]
        tp = np.cumsum(y)
        fp = np.cumsum(1 - y)
        prec = tp / np.maximum(tp + fp, 1)
        rec  = tp / P
        # envelope: precision doesn't grow when recall grows up
        prec = np.maximum.accumulate(prec[::-1])[::-1]
        # key points
        rec  = np.concatenate(([0.0], rec, [1.0]))
        prec = np.concatenate(([1.0], prec, [prec[-1]]))
        return float(trapezoid(prec, rec))

    def average_precision(y, s):
        y = np.asarray(y, dtype=int)
        s = np.asarray(s, dtype=float)
        P = int(y.sum())
        if P == 0:
            return float('nan')
        order = np.argsort(-s)
        y = y[order]
        tp = 0
        ap = 0.0
        for idx, yi in enumerate(y, start=1):
            if yi == 1:
                tp += 1
                ap += tp / idx
        return ap / P

    C_scores = S_scores.shape[1]
    roc_auc_per_class = []
    pr_auc_per_class  = []
    ap_per_class      = []

    for c in range(C_scores):
        y = Y_scores[:, c]; s = S_scores[:, c]

        # ROC-AUC at least one positive and one negative
        if np.unique(y).size < 2:
            roc_auc_per_class.append(np.nan)
        else:
            _, _, auc_roc = roc_points(y, s)
            roc_auc_per_class.append(auc_roc)

        # PR-AUC and AP at least one positive
        if y.sum() == 0:
            pr_auc_per_class.append(np.nan)
            ap_per_class.append(np.nan)
        else:
            pr_auc_per_class.append(pr_auc(y, s))
            ap_per_class.append(average_precision(y, s))

    # Macro: average without NaN
    macro_roc_auc = float(np.nanmean(roc_auc_per_class)) if len(roc_auc_per_class) else float('nan')
    macro_pr_auc  = float(np.nanmean(pr_auc_per_class))  if len(pr_auc_per_class)  else float('nan')
    macro_mAP     = float(np.nanmean(ap_per_class))      if len(ap_per_class)      else float('nan')

    # Micro-average: all the classes
    y_micro = Y_scores.ravel()
    s_micro = S_scores.ravel()
    _, _, micro_roc_auc = roc_points(y_micro, s_micro)
    micro_pr_auc        = pr_auc(y_micro, s_micro)
    micro_mAP           = average_precision(y_micro, s_micro)

    print("\n AUC scores ")
    print("Per-class ROC-AUC:", [f"{xa:.3f}" if np.isfinite(xa) else "" for xa in roc_auc_per_class])
    print("Per-class PR-AUC: ", [f"{xa:.3f}" if np.isfinite(xa) else "" for xa in pr_auc_per_class])
    print("Per-class AP:     ", [f"{xa:.3f}" if np.isfinite(xa) else "" for xa in ap_per_class])
    print(f"Macro ROC-AUC={macro_roc_auc:.3f} | Macro PR-AUC={macro_pr_auc:.3f} | Macro mAP={macro_mAP:.3f}")
    print(f"Micro ROC-AUC={micro_roc_auc:.3f} | Micro PR-AUC={micro_pr_auc:.3f} | Micro mAP={micro_mAP:.3f}")
else:
    print("\n[INFO] Skipping AUC metrics: no stored per-trial score arrays.")

# Rejection UNKNOWN metrics
unknown_trials = sum(1 for row in results if row[1] == ['UNKNOWN'])
unknown_ok      = sum(1 for row in results if row[1] == ['UNKNOWN'] and ('UNKNOWN' in row[2]))
unknown_strict_ok = sum(1 for row in results if row[1] == ['UNKNOWN'] and row[2] == ['UNKNOWN'])
if unknown_trials > 0:
    print(f"\nRejection accuracy (UNKNOWN on OOD/NULL): {unknown_ok}/{unknown_trials} = {unknown_ok/unknown_trials:.2%}")
else:
    print("\n[WARN] No UNKNOWN/OOD trials were included in the test set.")

if unknown_trials > 0:
    print(f"Rejection accuracy (STRICT): {unknown_strict_ok}/{unknown_trials} = {unknown_strict_ok/unknown_trials:.2%}")

# weight changes during test confrontation -> they don't change because STDP frozen during test phase
print("\nWeight changes during unsupervised test (population mean Δ):")
for p in range(unknown_id):
    idx = diag_indices_for_taste(p)  # p->pop(p)
    if idx.size == 0:
        print(f"  {taste_map[p]}→{taste_map[p]}: [no synapses]")
        continue
    dw = np.asarray(S.w[idx], float) - np.asarray(w_before_test[idx], float)
    print(f"  {taste_map[p]}→{taste_map[p]}: Δμ={float(np.mean(dw)):+.4f} "
          f"(min={float(np.min(dw)):+.4f}, max={float(np.max(dw)):+.4f})")
    
print("\nMean weight by class pair p→q (population-aware):")
for p in range(unknown_id):
    row = []
    for q in range(unknown_id):
        idx = diag_indices_for_taste(p, q)
        if idx.size:
            row.append(f"{np.mean(S.w[idx]):.3f}")
        else:
            row.append("---")
    print(f"{taste_map[p]:>6}: " + "  ".join(row))

# end test
print("\nEnded TEST phase successfully!")

# 13. Plots
# a) Spikes over time
plt.figure(figsize=(10,4))
plt.plot(spike_mon.t/b.ms, spike_mon.i, '.k')
plt.xlabel("Time (ms)")
plt.ylabel("Neuron index")
plt.title("Taste neurons spikes")
plt.show()

# b) Weight trajectories for diagonal synapses (i�i)
plt.figure(figsize=(14,6))
wm, syn = weight_monitors[0]  # monitor + associated synapse object
has_labels = False  # track if we actually added any visible label

for record_index, syn_index in enumerate(wm.record):
    pre = int(syn.i[syn_index])
    post = int(syn.j[syn_index])
    if pre == post and pre != unknown_id:
        plt.plot(wm.t/b.ms, wm.w[record_index], label=taste_map[pre])
        has_labels = True

plt.xlabel("Time (ms)")
plt.ylabel("Weight w")
plt.title("STDP + eligibility: diagonal synapses over time")

# Only show legend if we actually added any labeled line
if has_labels:
    plt.legend(loc='upper right')
else:
    print("[WARNING] No diagonal synapse (i->i) found among monitored synapses. Legend skipped.")

plt.tight_layout()
plt.show()

# c) Membrane potentials for all neurons
plt.figure(figsize=(14,6))
for idx in range(num_tastes):
    if idx == unknown_id:
        continue
    plt.plot(state_mon.t/b.ms, state_mon.v[idx], label=taste_map[idx])
plt.xlabel("Time (ms)")
plt.ylabel("v")
plt.title("Membrane potentials during always-on loop")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# d) Neuromodulators/WTA plot
# d1) Neuromodulators
plt.figure(figsize=(10,3))
for k in ['DA_f','DA_t','HT','NE','HI','ACH','GABA']:
    plt.plot(mod_mon.t/b.ms, getattr(mod_mon, k)[0], label=k)
plt.title('Neuromodulators through the time')
plt.legend(loc="upper right") 
plt.xlabel('ms') 
plt.tight_layout() 
plt.show()

# extimation of the average WTA movement
# d2) WTA / inh_scale
plt.figure(figsize=(8,3))
t = inh_mon.t / b.ms
# temporal average
y0 = np.asarray(inh_mon.inh_scale)[0]  # synapse 0
y_mean_over_syn = np.mean(np.asarray(inh_mon.inh_scale), axis=0)
plt.plot(t, y0, label='inh_scale: syn 0')
plt.plot(t, y_mean_over_syn, label='mean over synapses')

# synthetic indicator for the horizontal average
plt.axhline(y_mean_over_syn.mean(), linestyle='--', linewidth=1,
            label=f'time mean={y_mean_over_syn.mean():.2f}')

plt.title('WTA (inh\\_scale) through the time')
plt.xlabel('ms')
plt.ylabel('inh\\_scale')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# e) pos/neg EMA plot
'''for c in range(num_tastes-1):
    pos = np.array(pos_counts[c])
    neg = np.array(neg_counts[c])
    plt.figure(figsize=(5,3))
    plt.hist(neg, bins=20, alpha=0.6, label='neg')
    plt.hist(pos, bins=20, alpha=0.6, label='pos')
    plt.axvline(thr_per_class_test[c], ls='--')
    plt.title(f'{taste_map[c]}  | thr={int(thr_per_class_test[c])}')
    plt.xlabel('#spike per trial')
    plt.ylabel('freq')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()'''

# f) Cross-talk off-diagonal weights
W = np.zeros((num_tastes, num_tastes))
Si = np.array(syn.i[:])
Sj = np.array(syn.j[:])
Sw = np.array(syn.w[:])
for ii, jj, ww in zip(Si, Sj, Sw):
    W[int(ii), int(jj)] = float(ww)

plt.figure(figsize=(6,5))
plt.imshow(W[:unknown_id, :unknown_id], aspect='equal')
plt.xticks(range(unknown_id), [taste_map[k] for k in range(unknown_id)], rotation=45)
plt.yticks(range(unknown_id), [taste_map[k] for k in range(unknown_id)])
plt.colorbar(label='w')
plt.title('Weights matrix (taste->taste)')
plt.tight_layout()
plt.show()

# g) Plot dynamic SPICY
plt.figure(figsize=(10,4))
plt.plot(spice_mon.t/b.ms, spice_mon.spice_drive[0], label='drive')
plt.plot(spice_mon.t/b.ms, spice_mon.thr_spice[0],  label='thr')
plt.plot(spice_mon.t/b.ms, spice_mon.a_spice[0],    label='aversion')
plt.plot(spice_mon.t/b.ms, spice_mon.da_gate[0],    label='DA_gate')
plt.legend(loc="upper right")
plt.title('SPICY: drive vs thr (tolerance)')
plt.xlabel('ms')
plt.tight_layout()
plt.show()


# h) Plot dynamic taste i = 0..unknown_id-1
for idx in range(num_tastes-1):
    plt.figure(figsize=(10,3))
    plt.plot(hed_mon.t/b.ms, hed_mon.taste_drive[idx], label='drive')
    plt.plot(hed_mon.t/b.ms, hed_mon.thr_hi[idx], label='thr_hi')
    plt.plot(hed_mon.t/b.ms, hed_mon.thr_lo[idx], label='thr_lo')
    plt.legend(loc='upper right')
    plt.title(f'Hedonic window for {taste_map[idx]}')
    plt.xlabel('ms')
    plt.tight_layout()
    plt.show()
