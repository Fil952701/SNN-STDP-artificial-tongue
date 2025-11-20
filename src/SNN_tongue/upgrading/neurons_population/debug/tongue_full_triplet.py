# SNN with STDP + eligibility trace to simulate an artificial tongue # that continuously learns to recognize multiple tastes (always-on). 
# that continuously learns to recognize multiple tastes.

import brian2 as b
b.prefs.codegen.target = 'numpy'
#b.prefs.codegen.target = 'cpp_standalone'
#b.set_device('cpp_standalone', build_on_run=False)
import matplotlib.pyplot as plt
import sys
import random
import numpy as np
from collections import Counter, deque, OrderedDict
from numpy import trapezoid
import time
import math
import shutil

# GDI toggle for activation/deactivation
USE_GDI = True # True => use GDI; False => use rate normalization

# Imbalance/test controls (keep baseline clean = True)
# Rates management and sanity check on it
TASTE_NAMES = ["SWEET", "BITTER", "SALTY", "SOUR", "UMAMI", "FATTY", "SPICY", "UNKNOWN"]
#NUM_TASTES  = len(TASTE_NAMES)
num_tastes = 8  # SWEET, BITTER, SALTY, SOUR, UMAMI, FATTY, SPICY, UNKNOWN
UNKNOWN_ID  = TASTE_NAMES.index("UNKNOWN")
NUM_KNOWN   = UNKNOWN_ID
NORMALIZE_TEST_RATES = True # baseline=True; per test sbilanciati metti False
# global base rate per every class -> not 500 in total to split among all the classes but 500 for everyone
BASE_RATE_PER_CLASS = 500
BASE_RATE_KNOWN   = np.full(NUM_KNOWN, 420.0, dtype=float)
BASE_RATE_UNKNOWN = 1.0  # meglio di 0.0 per evitare divisioni per zero
BASE_RATE_PER_TASTE = np.concatenate([BASE_RATE_KNOWN, [BASE_RATE_UNKNOWN]])
#BASE_RATE_PER_TASTE = None

def check_shapes():
    assert len(BASE_RATE_PER_TASTE) == num_tastes, (
        f"BASE_RATE_PER_TASTE len={len(BASE_RATE_PER_TASTE)} "
        f"ma NUM_TASTES={num_tastes}. Allinea le dimensioni."
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
        # SPICY individual aversion
        spicy_aversion_trait = float(np.clip(rng.beta(2.0, 2.0), 0.0, 1.0)),  # 0..1
        k_hun_spice = float(rng.uniform(-0.20, -0.10)),  # fame riduce p (negativa)
        k_h2o_spice = float(rng.uniform(+0.15, +0.30))  # sete aumenta p (positiva)
    )

# state bias mapping for every taste (coeff dimensionless)
c_hun_hi = np.array([+0.06, 0.00, +0.01, 0.00, +0.03, +0.04, 0.00]) 
c_hun_lo = np.array([-0.05, 0.00, 0.00, 0.00, -0.02, -0.03, 0.00])
# slowing hungry tastes
c_hun_hi = np.asarray(c_hun_hi, dtype=float) * np.array([0.50, 1.00, 1.00, 1.00, 0.75, 0.90, 1.00])
c_hun_lo = np.asarray(c_hun_lo, dtype=float) * np.array([0.60, 1.00, 1.00, 1.00, 0.80, 0.95, 1.00])

c_sat_hi = np.array([-0.07, 0.00, 0.00, 0.00, -0.03, -0.05, 0.00])
c_sat_lo = np.array([+0.05, 0.00, 0.00, 0.00, +0.02, +0.03, 0.00])
c_h2o_hi = np.array([0.00, 0.00, -0.08, 0.00, 0.00, 0.00, -0.02])
c_h2o_lo = np.array([0.00, 0.00, +0.02, 0.00, 0.00, 0.00, 0.00])

# state bias mapping (dimensionless)
def apply_internal_state_bias(profile, mod, tn):
    H = float(mod.HUN[0])   # hungry
    S = float(mod.SAT[0])   # Satiety
    W = float(mod.H2O[0])   # Hydratation
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
    # anti-runaway
    max_rate_cap = 420.0
    r[:unknown_id] = np.minimum(r[:unknown_id], max_rate_cap)
    pg.rates = r * b.Hz

# Rates vector helper without normalization
def set_stimulus_vector(rate_vec, include_unknown=False):
    r = np.asarray(rate_vec, dtype=float).copy()
    if not include_unknown:
        r[unknown_id] = 0.0
    # anti-runaway
    max_rate_cap = 420.0
    r[:unknown_id] = np.minimum(r[:unknown_id], max_rate_cap)
    pg.rates = r * b.Hz

# Rates vector helper for test stimulus with zero UNKNOWN rate always and OVERSAMPLING for imbalanced data
# only for test phase
def set_test_stimulus(rate_vec):
    r = np.asarray(rate_vec, dtype=float).copy()

    # guardie utili
    assert r.ndim == 1, "rate_vec must be 1D"
    assert 0 <= unknown_id < r.size, f"unknown_id={unknown_id} out of range for size={r.size}"

    # azzera sempre UNKNOWN nel test
    r[unknown_id] = 0.0

    if NORMALIZE_TEST_RATES:
        # maschera solo sui gusti noti (0..unknown_id-1)
        act = r[:unknown_id] > 0
        if np.any(act):
            # slice con where (safe e leggibile)
            r[:unknown_id] = np.where(act, 160.0, r[:unknown_id])
    else:
        if BASE_RATE_PER_TASTE is not None:
            base = np.asarray(BASE_RATE_PER_TASTE, dtype=float)
            assert base.size >= unknown_id, (
                f"BASE_RATE_PER_TASTE measures {base.size}, but at least needs {unknown_id}"
            )
            # mantieni i gusti attivi, scala ai base-rate
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
    alpha_w = np.zeros(K, float)
    At = A.T
    # step suggerito: 1/L con L = ||A||_2^2 (stima spettrale cheap con norma Frobenius)
    if lr is None:
        L = np.linalg.norm(A, ord='fro')**2
        lr = 1.0/max(L, 1e-9)
    prev = alpha_w.copy()
    for _ in range(iters):
        grad = At @ (A @ alpha_w - b)
        alpha_w = _project_to_simplex(alpha_w - lr*grad, z=l1_cap)
        if np.max(np.abs(alpha_w - prev)) < tol:
            break
        prev = alpha_w.copy()
    return alpha_w
# 2.
def recon_error(A, b, alpha_w):
    r = b - A @ alpha_w
    return float(np.linalg.norm(r, 2))  # L2
# 3. UNSUP: conf score and micro-probe for logging and verification of unsupervised learning success
def clamp01(xs): 
    return float(np.clip(xs, 0.0, 1.0)) # clipping
# 4. confidence probability => higher means good confidence: correct unsupervised learning, smaller means bad confidence
def conf_unsup_score(err, top_alpha, good_coverage):
    # conf_unsup = (1 - err) * top_alpha * good_coverage
    return clamp01((1.0 - float(err)) * float(top_alpha) * (1.0 if good_coverage else 0.0))
# 5.
def probe_boost_candidates(cands, base_rates, K=3, dur_ms=120, boost_frac=0.12):
    """
    Micro-trial: re-inietta lo stimolo K volte aumentando i canali candidati di +10-15%.
    Ritorna (ok, pmr_series, gap_series, z_series_dict)
    - ok=True se PMR/gap e z dei candidati crescono "abbastanza" in maniera consistente.
    Nota: non tocca STDP (è già frozen nel TEST).
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
    scores, P_pos, P_cop,
    z_scores,                          # array z per-gusto (C,)
    frac_thr=0.08,                     # soglia frazionaria per i pesi -> between 0.10 and 0.15
    z_min_guard=0.015,                  # guardia minima su z per accettare un candidato
    allow_k4=True,                     # flag per quadruple
    allow_k5=True,                     # flag per quintuple
    gap=None, gap_thr=None,            # calcolati a monte da ood_calibration
    pmr=None, pmr_thr=None,            #
    abs_floor=None,                    # None, float o array per-gusto
    nnls_iters=250, nnls_lr=None, l1_cap=1.0 # iperparametri di NNLS proiettato sul simplesso
):
    # blend fisso
    alpha = 0.55 if is_mixture_like else 0.20 # più copioso se mix, mentre 0.20 se positivi puri
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
    
    # clamp l1_cap in base a k_est (più piccolo => più sparso)
    l1_cap_eff = min(l1_cap, 0.80 + 0.05*min(k_est, 5))  # 0.85..1.05 clamp a ~0.85-0.95
    ws = nnls_projected_grad(P_n, b_n, iters=nnls_iters, lr=nnls_lr, l1_cap=l1_cap_eff)

    # soglie frazionarie: 0.10 base, 0.08 se target quintuple
    ws_sum = float(ws.sum())
    thr_rel_base = frac_thr * ws_sum
    thr_rel_k5   = min(frac_thr, 0.08) * ws_sum

    # conta coefficienti “significativi”
    k_abs = int(np.sum(ws >= max(1e-12, thr_rel_base)))
    k_abs5 = int(np.sum(ws >= max(1e-12, thr_rel_k5)))

    # criteria acceptation
    basic_ok = (err <= 0.65 and cover >= 0.50 and k_abs in (1,2,3,4,5))

    k4_ok = False
    if allow_k4 and (k_abs == 4 or (k_est >= 4 and int(np.sum(ws >= thr_rel_base)) >= 4)):
        # più pulito e distribuzione compatibile
        clean_ok   = (err <= 0.28 and cover >= 0.78)
        shape_ok   = (k_est >= 5)
        pmr_gap_ok = True
        if (gap is not None and gap_thr is not None) or (pmr is not None and pmr_thr is not None):
            pmr_gap_ok = ((gap is not None and gap_thr is not None and gap >= 0.95*gap_thr) or
                          (pmr is not None and pmr_thr is not None and pmr >= 1.05*pmr_thr))
        k4_ok = clean_ok and shape_ok and pmr_gap_ok

    k5_ok = False
    if allow_k5 and (k_abs5 == 5 or (k_est >= 5 and int(np.sum(ws >= thr_rel_k5)) >= 5)):
        # condizioni ancora più severe
        clean_ok   = (err <= 0.30 and cover >= 0.76)
        shape_ok   = (k_est >= 5)  # lo spettro deve "dire" 5 per classificare correttamente 5-mix
        pmr_gap_ok = True
        if (gap is not None and gap_thr is not None) or (pmr is not None and pmr_thr is not None):
            pmr_gap_ok = ((gap is not None and gap_thr is not None and gap >= 1.05*gap_thr) or
                          (pmr is not None and pmr_thr is not None and pmr >= 1.15*pmr_thr))
        abs_floor = 2.0 if abs_floor is None else abs_floor   # almeno 2 spike-equivalenti
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
        # Se lo spettro è "piatto/confuso" → fallback UNKNOWN
        # Usa guardie debolmente conservative: se mancano gap/pmr passali come None.
        if (gap is not None and gap_thr is not None and gap < gap_thr) or \
        (pmr is not None and pmr_thr is not None and pmr < pmr_thr):
            return [unknown_id], dict(err=float(err), cover=cover, k_abs=0, k_est=k_est,
                                  thr_rel=0.0, ws=ws)
        # altrimenti scegli il top z tra i noti
        top1 = int(np.argmax(z_scores[:unknown_id]))
        return [top1], dict(err=float(err), cover=cover, k_abs=1, k_est=k_est,
                        thr_rel=0.0, ws=ws)

    # selezione candidati
    thr_rel_eff = float(thr_rel_k5 if k5_ok else thr_rel_base)
    cand = [ids for ids, wi in enumerate(ws) 
            if (wi >= thr_rel_eff) and (z_scores[ids] >= (0.0 if k5_ok else z_min_guard))]
    
    # una guardia addizionale se 4 o 5 candidati (es. min score assoluto)
    if (len(cand) >= 4) and isinstance(abs_floor, float):
        cand = [ids for ids in cand if scores[ids] >= abs_floor]
    elif (len(cand) >= 4) and isinstance(abs_floor, np.ndarray):
        cand = [ids for ids in cand if scores[ids] >= abs_floor[ids]]

    # se per rumore cand > 5, tieni i migliori 5 per peso ws -> si ordina per score di ognuno
    if len(cand) > 5:
        cand = sorted(cand, key=lambda ids: ws[ids], reverse=True)[:5]
    
    # cardinality priority
    if len(cand) > k_est:
        # tieni i top-k_est per ws, ma consenti +1 se quasi pari
        order = np.argsort(ws[cand])[::-1]
        keep = list(np.array(cand)[order[:k_est]])
        if len(cand) >= k_est+1:
            nxt = cand[order[k_est]]
            if ws[nxt] >= 0.95*ws[keep[-1]] and z_scores[nxt] >= 0.9*max(z_scores[keep[-1]], z_min_guard):
                keep.append(nxt)  # tolleranza +1
        cand = keep

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

        # floor
        if floor > 0:
            wcol = np.maximum(wcol, floor)

        # bias diagonale: se il neurone j appartiene alla pop di classe q → boosta i==q
        q = (jx // NEURONS_PER_TASTE)  # classe della pop di j
        wcol[(icol == q)] *= diag_bias_gamma

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
    conds = 0
    if pmr < PMR_thr: conds += 1
    if gap < gap_thr: conds += 1
    if H   > H_thr:  conds += 1

    # prima: >=3 → 0.6, altrimenti 0.0
    # dopo:  >=2 → 1.0 (forte), 1 → 0.4 (blando), 0 → 0.15 (baseline anti-falso-positivo)
    S_unk.gain_unk = 1.15 if conds >= 2 else (0.55 if conds == 1 else 0.35)
    
    # se c'è craving alto ma evidenza debole -> preferisci UNKNOWN, evita “wishful thinking”
    if (max(crave_f[:unknown_id]) > 0.80 or max(crave_s[:unknown_id]) > 0.80) and (pmr < PMR_thr or gap < gap_thr):
        S_unk.gain_unk = max(float(S_unk.gain_unk), 1.30)
    
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
    h = f"\n[Trial {step}] " if step is not None else ""
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
    
# UNKNOWN ID
unknown_id = num_tastes-1
# gusti senza UNKNONW ID
num_eff = num_tastes-1

#############################################################################

# THE FOLLOWING PART IS ALL ABOUT CRAVING AND DESIRE HEDONIC WINDOW STATE

#############################################################################

# CRAVING state
crave_f = np.zeros(num_tastes, dtype=float)    # fast trace per class
crave_s = np.zeros(num_tastes, dtype=float)    # slow trace per class
crave_hedonic = np.full(num_tastes, -1, dtype=int)  # hedonic window per class
mix_desire = OrderedDict()  # key: tuple di classi attive (ordinata); val: dict(f,s,deadline)
trial_idx = 0  # trial global counter

# DESIRE e CRAVING toggles & hyperparams
TRAINING_PHASE = True                  # set True nel loop train, False nel loop test
# toggle ON/OFF
ENABLE_DESIRE        = True
ENABLE_DESIRE_TEST   = False           # to try ablation put to False
# plasticity overall scaling
CRAVE_KF             = 0.40            # fast trace weight
CRAVE_KS             = 0.25            # slow trace weight
CRAVE_MAX            = 1.5             # saturation
# exponential decays
CR_TAU_F_TRIALS      = 40              # memory of minutes/hours
CR_TAU_S_TRIALS      = 550             # between 400-600 => long term consolidation
# hedonic window crawing satisfaction
CR_WIN_TRIALS        = 30
# finestra di soddisfazione del desiderio (in trial)
CR_WIN_TRIALS        = 30
# quanto aumenta la voglia se “piace” (scalato da confidenza)
CR_INIT_BOOST        = 0.25
# bonus dopaminergico addizionale se soddisfi il desiderio in finestra
CR_SAT_BONUS         = 0.35
# penalità se esperienza è avversiva (spicy cattivo, mix pessimo, ecc.)
CR_NEG_PENALTY       = 0.20
# soglia minima di confidenza per generare desiderio potente
CR_MIN_CONF          = 0.65
# memoria mix con LRU
CR_MIX_CACHE         = 128

# HELPERS FOR CRAVING
# 1. Clamp
def clamp(xc, lo=0.0, hi=CRAVE_MAX):
    return float(np.clip(xc, lo, hi))

# 2. Decay step
def decay_step(arr, tau_trials):
    if tau_trials <= 0:
        return
    decay = np.exp(-1.0 / float(tau_trials))
    arr *= decay

# 3. Mixture key => returns a key for the current mix (without UNKNOWN).
#                   active_tastes_list: lista di id classe 'attive' nello stimolo
def mixture_key(active_tastes):
    lst = [int(xs) for xs in active_tastes if xs != unknown_id]
    if not lst:
        return None
    return tuple(sorted(lst))

# 4. Crave scale => plasticity scaling factor for idx class
def crave_scale(idx: int) -> float:
    sc = 1.0 + CRAVE_KF*float(crave_f[idx]) + CRAVE_KS*float(crave_s[idx])
    return 1.0 if (not TRAINING_PHASE) else float(np.clip(sc, 1.0, 1.0 + CRAVE_KF + CRAVE_KS))

# 5. Mix cache set => easy LRU buffer to store all the desires
def mix_cache_set(kd, st):
    if kd in mix_desire:
        mix_desire.move_to_end(kd)
    mix_desire[kd] = st
    if len(mix_desire) > CR_MIX_CACHE:
        mix_desire.popitem(last=False)

# 6. Update desire => effective function to update taste desire after trial
def update_desire(
    winners,                 # lista di vincitori (id classe)
    top_idx,                 # miglior classe
    conf_score,              # conf_unsup_score(...) / altra conf
    liked_flag: bool,        # flag per "mix godurioso"
    active_tastes,           # stimolo presentato (classi attive)
    aversion_flag: bool      # p.es. spicy avversivo o mix pessimo
):
    
    global trial_idx         # trial global counter
    
    # 1) global decay at every trial
    decay_step(crave_f, CR_TAU_F_TRIALS)
    decay_step(crave_s, CR_TAU_S_TRIALS)
    
    # 2) desiderio "arde" quando il mix è particolarmente piaciuto e la rete è sicura
    liked = bool(liked_flag)
    if liked and (conf_score >= CR_MIN_CONF) and (top_idx != unknown_id):
        inc = CR_INIT_BOOST * float(conf_score)
        crave_f[top_idx] = clamp(crave_f[top_idx] + inc)
        crave_s[top_idx] = clamp(crave_s[top_idx] + 0.25*inc) # because it's slower than fast one
        crave_hedonic[top_idx] = trial_idx + CR_WIN_TRIALS # hedonic window if it is liked
        
        # what is that taste?
        # we recover all data about it
        mk = mixture_key(active_tastes)
        if mk:
            st = mix_desire.get(mk, {"fast": 0.0, "slow": 0.0, "hedonic": trial_idx + CR_WIN_TRIALS})
            st["fast"] = clamp(st["fast"] + inc)
            st["slow"] = clamp(st["slow"] + 0.25*inc)
            st["hedonic"] = trial_idx + CR_WIN_TRIALS
            mix_cache_set(mk, st)
    
    # 3) penalty where there is aversion
    if aversion_flag and (top_idx != unknown_id):
        dec = CR_NEG_PENALTY * max(0.25, float(conf_score))
        crave_f[top_idx] = clamp(crave_f[top_idx] - dec)
        crave_s[top_idx] = clamp(crave_s[top_idx] - 0.5*dec)
    
    # 4) soddisfazione del desiderio: se lo stimolo del trial
    # coincide con un desiderio "aperto", dai un bonus DA e chiudi la finestra
    if TRAINING_PHASE and (top_idx != unknown_id) and (crave_hedonic[top_idx] >= trial_idx):
        try:
            # fast DA
            gain = meta_scale(top_idx)  # 0.3..1.65
            mod.DA_f = float(np.clip(mod.DA_f + CR_SAT_BONUS * float(conf_score) * gain, 0.0, 1.0))
            try:
                mod.DA_f = float(max(0.35, min(1.0, mod.DA_f)))  # non sotto 0.35
            except Exception:
                pass
        except Exception:
            pass
        crave_hedonic[top_idx] = -1  # finestra consumata
    
    # 5) decay and cleanup mix: delete all the offload voices outside hedonic window
    to_del = []
    for mk, st in mix_desire.items():
        st["fast"] *= np.exp(-1.0/CR_TAU_F_TRIALS)
        st["slow"] *= np.exp(-1.0/CR_TAU_S_TRIALS)
        if (st["fast"] + st["slow"] < 1e-3) and (trial_idx > st["hedonic"] + CR_WIN_TRIALS):
            to_del.append(mk)
    for mk in to_del:
        mix_desire.pop(mk, None)

    # 6) avanzamento tempo trial-based
    trial_idx += 1

#############################################################################

# ENDING OF THE CRAVING AND DESIRE PART

#############################################################################



#############################################################################
# 3. Simulation global parameters
#############################################################################
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
g_step_exc           = 4.2 * b.nS       # excitation from inputs
g_step_bg            = 0.2 * b.nS       # tiny background excitation noise
g_step_inh_local     = 1.5 * b.nS       # lateral inhibition strength

# Global Divisive Inhibition (GDI) parameters -> we need to balance very well this improvement
# because it will break all the mixes otherwise
tau_gdi              = 80 * b.ms        # global pool temporal window
k_e2g_ff             = 0.0038           # feed-forward contribute (Poisson input spikes) -> GDI
k_e2g_fb             = 0.0016           # feedback contribute (output neuron spikes) -> GDI
gamma_gdi_0          = 0.14             # scaled reward for (dimensionless)
gdi_reset_each_trial = True             # managing carry-over thorugh trials

# Neuromodulators (DA Dopamine: reward, 5-HT Serotonine: aversion/fear, NE Noradrenaline: arousal/attention)
# Dopamine (DA) - reward gain
tau_DA               = 300 * b.ms       # fast decay: short gratification
tau_HT               = 2 * b.second     # slow decay: prudence, residual fear
da_gain              = 1.0              # how much DA expand the positive reinforcement
ht_gain              = 1.0              # how much 5-HT expand the punishment or how much it stops LTP
# Serotonine (5-HT) - aversive state on the entire circuit
k_theta_HT           = 2.0              # mV bias threshold per 5-HT unit
k_inh_HT             = 0.4              # WTA scaling per 5-HT unit
# aversive stochastic events
da_pulse_reward      = 1.0              # burst DA if classification is correct -> reward
ht_pulse_aversion    = 1.0              # burst 5-HT when imminent event is aversive
ht_pulse_fp          = 0.35              # burst 5-HT extra if there are a lot of strong FP
p_aversion_base      = 0.01             # probabilità base di evento avversivo quando SPICY è presente => mettere 0.02 per spegnere o quasi
p_aversion_slope     = 0.04             # quanto la probabilità cresce con l'intensità relativa => mettere 0.05 per spegnere o quasi
p_aversion_cap       = 0.60             # massimo assoluto (clamp)
da_penalty_avers     = 0.85              # frazione con cui attenuare il reward DA (0.5 = dimezza)
thr_spice_kick       = 0.02             # quanto alzare la soglia di tolleranza se l'evento accade (adattamento rapido)
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
ach_train_level      = 0.85             # high ACh during training
ach_test_level       = 0.30             # lower ACh during test
k_ex_ACH             = 0.20             # exictement gain with ACh
ach_plasticity_gain  = 0.40             # ACh -> reward effect on -> w
k_noise_ACH          = 0.25             # ACh noise environment reduction
# GABA -> global inhibition/stability
tau_GABA             = 800 * b.ms       # decay for GABA
k_inh_GABA           = 0.60             # WTA scaling with GABA
gaba_pulse_stabilize = 0.8              # burst when activity is too much
gaba_active_neurons  = 4                # if > k neurons are activated in the same time � stability
gaba_total_spikes    = 120              # if total spikes per trial overcome this threshold � stability

# Dopamine delay dynamics + tonic tail (phasic vs tonic)
tau_DA_phasic        = 300 * b.ms       # fast phasic peak decay
tau_DA_tonic         = 2 * b.second     # slow tonic tail decay
dopamine_latency     = 150 * b.ms       # little delay before weight update -> more biologically plausible
k_tonic_DA           = 0.35             # how much the tonic tail contributes during plasticity
da_tonic_tail        = 0.25             # when the reward is gained => how big is the tonic tail quote
# state -> tonic bias (hungry/thirsty increase DA_tonic baseline)
k_hun_tonic          = 0.20             # hungry state scaling
k_h2o_tonic          = 0.20             # thirsty state scaling

# Intrinsic homeostasis adapative threshold parameters
target_rate          = 50 * b.Hz        # reference firing per neuron (tu 40-80 Hz rule)
tau_rate             = 200 * b.ms       # extimate window for rating -> spikes LPF
tau_theta            = 1 * b.second     # threshold adaptive speed
theta_init           = 0.0 *b.mV        # starting theta threshold for homeostasis
rho_target = target_rate * tau_rate     # dimensionless (Hz*s)

# Decoder threshold parameters
k_sigma              = 0.9               # scaling factor for decoder threshold
q_neg                = 0.99              # negative quantile

# Multi-label RL + EMA decoder
ema_lambda            = 0.10             # EMA factor for decoder expectations
tp_gate_ratio         = 0.30             # threshold to reward winner classes
fp_gate_warmup_steps  = 200              # delay punitions to loser classes if EMA didn't stabilize them yet
decoder_adapt_on_test = False            # updating decoder EMA in test phase
ema_factor            = 0.40             # EMA factor to punish more easy samples
use_rel_gate_in_test  = True             # using relative gates for mixtures and not only absolute gates
rel_gate_ratio_test   = 0.13             # second > 45 % rel_gate
mixture_thr_relax     = 0.40             # e 50% of threshold per-class
z_rel_min             = 0.005            # z margin threshold to let enter taste in relative gate
z_min_base            = 0.06             # prima 0.20
z_min_mix             = 0.035            # prima 0.10
# dynamic absolute thresholds for spikes counting  
rel_cap_abs           = 10.0             # absolute value for spikes
dyn_abs_min_frac      = 0.22             # helper for weak co-tastes -> it needs at least 30% of positive expected
# boosting parameters to push more weak examples
norm_rel_ratio_test   = 0.022            # winners with z_i >= 15% normalized top
min_norm_abs_spikes   = 2                # at least one real spike -> avoiding numerical issues
# EMA historicals for decoder
eps_ema               = 1e-3             # epsilon for EMA decoder
mix_abs_pos_frac      = 0.07             # positive expected fraction
# metaplasticity -> STDP reward adapted to the historical of how often the reward is inside the hedonic window
meta_min              = 0.3              # lower range STDP scaling
meta_max              = 1.45             # higher range STDP scaling
meta_lambda           = 0.05             # EMA velocity
gwin_ema        = np.zeros(unknown_id)   # historical for every class

# Off-diag hyperparameters
beta                  = 0.03             # learning rate for negative reward
beta_offdiag          = 0.25 * beta      # off-diag parameter
use_offdiag_dopamine  = True             # quick toggle to activate/deactivate reward for off-diagonals
OFFDIAG_DECAY         = 0.85              # fattore di decadimento graduale
OFFDIAG_W_FLOOR       = 1.5e-3             # pavimento > 0 per non andare mai a 0 secco
beta_offdiag_map      = np.ones(unknown_id)
beta_offdiag_map[0]   = 2.40             # SWEET
beta_offdiag_map[1]   = 1.60             # BITTER 
beta_offdiag_map[2]   = 2.40             # SALTY
beta_offdiag_map[3]   = 1.60             # SOUR
beta_offdiag_map[4]   = 2.20             # UMAMI 
beta_offdiag_map[5]   = 2.20             # FATTY
beta_offdiag_map[6]   = 2.00             # SPICY

#beta_offdiag_map = {ts: 1.2 for ts in TASTE_NAMES} 

# Normalization per-column (synaptic scaling in input)
use_col_norm          = True             # on the normalization
col_norm_mode         = "l1"             # "l1" (sum=target) or "softmax" -> synaptic scaling that mantains input scale per post neuron to avoid unfair competition
col_norm_every        = 29               # execute norm every N trial
col_norm_temp         = 1.0              # temperature softmax (if mode="softmax")
col_norm_target       = None             # if None, calculating the target at the beginning of the trial
diag_bias_gamma       = 1.40             # or 1.45; >1.0 = light bias to the diagonal weight before normalization
col_floor             = 0.01             # floor (0 or light epsilon) before norm
col_allow_upscale     = False            # light up-scaling
col_upscale_slack     = 0.95             # if L1 < 90% target � boost
col_scale_max         = 1.01            # max factor per step
# col_norm conditions function
NORM_WARMUP           = 90
COLNORM_PHASE         = (NORM_WARMUP + 1) % col_norm_every
#COLNORM_PHASE         = None
SAFE_GAP              = 2                # salta col norm per 2 trial dopo l’homeostasi plastica
DECAY_GAP             = 1                # salta col norm per il trial del decay e per quello subito dopo
GLOBAL_MIN_GAP        = 2                # almeno 15 trial di distanza
SOFT_PULL_STRONG_THR  = 0.7
soft_pull_strength    = 0.0              # 0=nessun veto, 1=veto forte

# SPICY dynamic tolerance / aversion dynamics
spicy_id              = 6                # spicy taste is the sixth one
fatty_id              = 5                # fatty taste is the fifth one
thr0_spice_var        = 0.36             # baseline aversive threshold -> driven unit
tau_thr_spice         = 30 * b.second    # adapting threshold -> slow
tau_sd_spice          = 80 * b.ms        # spicy intensity integration
tau_a_spice           = 200 * b.ms       # dynamic aversion
k_spike_spice         = 0.012            # spike contribution pre SPICY->drive
k_a_spice             = 1.0              # aversion reward
#k_hab_spice          = 0.0015           # upgrade threshold with adapting to aversion
eta_da_spice          = 2.0              # multiplier DA for adapting
#k_sens_spice         = 0.001            # sensitization if above threshold but without reward -> just an adapting on the previous threshold: now higher
reinforce_dur         = 150 * b.ms       # short window to push DA gate on SPICY
SPICY_AVERSION_WARMUP_STEPS = 300        # o ~len(train_phase1)
SPICY_MIN_SEEN        = 5                # almeno 5 trial SPICY visti prima di punire
spicy_seen            = 0
# DYNAMIC SPICY AVERSION META-STATE (combo-based)
spicy_combo_bad_ema   = 0.0              # 0 = va tutto bene, 1 = SPICY+mix disastroso

SPICY_BAD_ALPHA       = 0.02             # velocità EMA (0.01–0.05)
SPICY_BAD_MIN_SEEN    = 8                # minimo numero di volte che ho visto una combo per fidarmi

SPICY_BAD_THR         = 0.30             # oltre questo errore medio, considero "problema serio"
SPICY_P_GAIN          = 0.6              # quanto può crescere p_aversion_base (max +60%)
SPICY_HT_GAIN         = 0.5              # quanto può crescere il colpo di HT
SPICY_THR_GAIN        = 0.4              # quanto può crescere thr_spice_kick

# Hedonic window for all the tastes (SWEET, SOUR ecc...) -> one taste is rewarding ONLY if his spikes fire during this period
tau_drive_win         = 50 * b.ms        # intensity/taste integration
tau_av_win            = 200 * b.ms       # aversion/sub-threshold integration
tau_thr_win           = 30 * b.second    # thresholds adapting
eta_da_win            = 2.0              # rewarding on the habit of the Hedonic window
k_spike_drive         = 0.015            # driving kick on each input spike

# Hedonic gating for DA state-dependente (fallback included) -> if a taste is recognized inside the hedonic window => full DA, otherwise if it is ricognized but it's not in the window => less DA
use_hedonic_da       = True             # toggle for hedonic DA gating ON/OFF
hed_fallback         = 0.65             # minimum reinforcement if prediction is not inside the hedonic window
hed_gate_k           = 1.5              # gating convergence: �k = more aggressive
hed_min              = 0.10             # lower fallback clamp
hed_max              = 0.95             # higher fallback clamp
k_hun_fb             = 0.15             # hungry -> � fallback for SWEET/UMAMI/FATTY
k_sat_fb             = 0.25             # satiety -> � fallback for SWEET/UMAMI/FATTY
k_h2o_fb             = 0.25             # thirsty -> � fallback for SALTY/SPICY
k_bitter_sat         = 0.10             # bitter -> � light fallback with satiety
# energy needs requirements mapping
hunger_idxs          = [0, 4, 5]        # SWEET, UMAMI, FATTY
water_idxs           = [2, 6]           # SALTY, SPICY

# Spike-Timing Dependent Plasticity STDP and environment parameters
tau                  = 30 * b.ms        # STDP time constant
Te                   = 50 * b.ms        # eligibility trace decay time constant
A_plus               = 0.006            # dimensionless
A_minus              = -0.0045          # dimensionless
alpha_lr             = 0.06             # learning rate for positive reward
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
top2_margin_ratio    = 1.12             # top/second >= 1.4 -> safe
verbose_rewards      = True             # dopamine reward logs
test_emotion_mode    = "off"            # to test with active neuromodulators

# Short-Term Plasticity STP (STF/STD) (Tsodyks-Markram) parameters
use_stp              = True             # toggle to set STP ON/OFF
# default parameters for STP applied to pg -> taste neurons (avoiding collapse)
stp_u0               = 0.11             # baseline u (utilization)
stp_uinc             = 0.02             # increasing per-spike (facilitation)
stp_tau_rec          = 180 * b.ms       # recovery (STD)
stp_tau_facil        = 450 * b.ms       # facilitation decay (STF)
stp_r_ref            = 140.0            # reference Hz to calibrate gain for STP
stp_warmup_trials    = 150              # important initial warmup to avoid DA suppression in the beginning
# Warmup steps calculation for STP variables stabilization during test phase
WARM_STP             = True             # toggle to set STP warmup ON/OFF during test phase
WARM_STP_MS          = 100*b.ms         # between 80–120 ms

# Turrigiano-like synaptic scaling Plastic Homeostasis -> meta-meta-plasticity
HOMEOSTASIS_ON           = True         # toggle generale
HOMEOSTASIS_TARGET_RATE = 30.0          # Hz per neurone di output (set-point desiderato)
HOMEOSTASIS_ALPHA       = 0.03          # EMA lenta sui firing rate (≈ costante di tempo ~100 trial)
HOMEOSTASIS_BETA        = 0.05          # morbidezza scaling (0 -> no scaling, 1 -> scaling pieno)
HOMEOSTASIS_EVERY       = 301           # applica lo scaling ogni N trial
HOMEOSTASIS_SCALE_MIN   = 0.97          # fattore minimo per step (evita cambiamenti violenti)
HOMEOSTASIS_SCALE_MAX   = 1.03          # fattore massimo per step
HOMEOSTASIS_PHASE       = 7             # avoiding other intervents coincidences
homeo_r_avg = np.zeros(TOTAL_OUT, dtype=float)
trial_T_sec = float(training_duration / b.second) # durata del trial in secondi (per passare da spike/trial a Hz)

# Structural epigenetics (very slow modulation of metaplastic range)
EPI_ON                  = True          # master toggle per l'epigenetica strutturale
EPI_ALPHA               = 0.005         # EMA molto lenta della performance
EPI_EVERY               = 1061           # ogni quanti trial di training aggiornare meta_min/meta_max
EPI_CONSOLIDATION       = False         # trigger for EPI
# range "di specie" entro cui può muoversi il range metaplastico
EPI_META_MIN_LO         = 0.20
EPI_META_MIN_HI         = 0.60
EPI_META_MAX_LO         = 1.30
EPI_META_MAX_HI         = 2.00
# Scheduling/guard per aggiornamento epigenetico
EPI_PHASE               = 5             # offset di fase per evitare coincidenze con step multipli
EPI_SAFE_GAP            = 2             # salta se troppo vicino a homeostasi/col-norm/decay
EPI_GLOBAL_MIN_GAP      = 15            # distanza minima da altri interventi strutturali
# performance lenta "di specie" (per epigenetica)
epi_perf_ema            = 0.0
# History per l’epigenetica (plateau detection)
epi_ema_hist            = deque(maxlen=512)  # history lunga di ema_perf_long (o simile)
n_epi_events            = 0             # quante volte abbiamo già fatto un "macro-evento"
EPI_FREQ                = EPI_EVERY     # alias esplicito: ogni quanti step provare l’evento        

# Dynamic OVERSAMPLING -> TRAIN ONLY
USE_DYNAMIC_OVERSAMPLING = False # switch to TRUE to apply it
CLASS_BOOST = np.ones(unknown_id, dtype=float)
CLASS_BOOST[6]       = 1.10             # SPICY leggermente sopra perché è già molto presente
BOOST_LAM            = 0.10             # EMA speed
BOOST_GAIN           = 0.40             # quanto “spinge” il need
BOOST_CAP            = (0.90, 1.15)     # clamp per gusto (min,max)
BOOST_APPLY_GUARD_STEPS = fp_gate_warmup_steps  # non applicare boost nei primissimi step (EMA ancora grezze)

# Biological per-class attention bias -> TRAIN ONLY (no oversampling)
USE_ATTENTIONAL_BIAS = True             # toggle to set attentional bias ON/OFF
ATTN_BIAS_GAIN_MV    = 0.42             # mV di bias max circa per unità "need_norm-1"
ATTN_BIAS_CAP_MV     = 6.0              # cap di sicurezza per gusto

# Bio-plausible early-stopping (DA gating + best snapshot)
ema_perf             = 0.0              # EMA della performance
perf_alpha           = 0.03             # smoothing dolce della performance
DA_GATE_JACC         = 0.40             # sotto questa Jaccard per trial: niente DA (no consolidamento)
VAL_EVERY            = 0                # 0 = solo proxy intra-trial; (>0 per mini-validation periodica)
# Early stopping setup
best_score           = -float("inf")    # best early-stop score
best_step            = -1               # best step for early-stop
best_state           = None             # best snapshot of SNN
patience             = 0                # patience counter to trigger early-stop
# Bio-plausible consolidation & early-stop params
USE_PLASTICITY_DECAY = False            # toggle ON/OFF plasticity decay
USE_SLOW_CONSOLIDATION = True           # toggle to set slow consolidation ON/OFF 
USE_EARLY_STOP       = False            # Test 1: niente early stopping
EARLY_STOP_MIN_FRAC  = 0.7              # usa early stop solo dopo il 70% dei trial
PATIENCE_LIMIT       = 200              # tuning it on the current setup base; with 1500 trials and soft-EMA: 100 is enough
ETA_CONSOL           = 0.05             # slow capture rate (0..1) toward current fast weights
DA_THR_CONSOL        = 0.35             # require enough DA to consolidate into w_slow
BETA_MIX_TEST        = 0.10             # how much fast to keep when mixing slow→test (0..1)
USE_SOFT_PULL        = True             # softly drift toward best-state when stagnating
RHO_PULL             = 0.25             # 0..1, per-trial pull strength
PLASTICITY_DECAY     = 0.95             # multiplicative decay on S.stdp_on when stagnating
MAX_PLASTICITY_DECAYS= 1                # maximum number of decays allowed
W_EMA                = 12               # window size for EMA calculation
# storico corto per stimare il rumore dell'EMA
ema_hist             = deque(maxlen=32) # storicizzo le ultime 32 performance
ema_perf_sd          = 0.0              # aggiornata ad ogni trial
decays_done          = 0                # quante volte ho fatto decay finora
# Reduce-on-plateau globals
PLAST_GLOBAL         = 1.0              # plasticity global scaling factor
PLAST_GLOBAL_FLOOR   = 0.30             # minimum plasticity scaling factor
REDUCE_FACTOR        = 0.90             # factor to reduce plasticity on plateau
COOLDOWN             = 8                # cooldown period after plasticity reduction
PLATEAU_WINDOW       = 20               # ogni N trial senza migliorie posso decidere un decay per il plateau
# EMA "lenta" per decidere i decays di plasticità
ema_perf_long        = 0.0
LAM_LONG             = 0.03             # più lento della ema_perf classica
best_ema_perf        = 0.0              # best sulla EMA lunga (no rumore)
best_ema_step        = 0
last_decay_step      = -10**9
EMA_POS_SD_COEFF     = 0.6   
TP_FLOOR_SCALE       = 0.75  
# Gate sui decays (su EMA lunga)
DECAY_WARMUP_STEPS   = 500              # prima di 500 step, mai decays
DECAY_COOLDOWN_STEPS = 100              # min distanza tra due decays
DECAY_PATIENCE_STEPS = 200              # finestra lunga senza miglioramenti
DECAY_EPS            = 0.08             # quanto deve ESSERE PEGGIO di best_ema_perf
STDP_ON_MIN          = 0.65             # clamp minimo per stdp_on / PLAST_GLOBAL
STDP_ON_DECAY_FACTOR = 0.9              # riduzione quando faccio un decay globale
JACC_THR             = 0.4              # minima soglia Jaccard per blind

# Connectivity switch: "diagonal" | "dense" (keep it "dense" for population neurons and more bio-plausibility)
connectivity_mode    = "dense"          # "dense" -> fully-connected | "diagonal" -> one to one

# Test stimulus manager to stop/start test stimulus
def stop_test_stimulus():
    set_test_stimulus(None, start=0*b.ms, stop=0*b.ms)

# Plasticity pull (in Tensorflow's ReduceLROnPlateau style)
def set_plasticity_scale(scale: float):
    prev = float(np.mean(S.stdp_on[:])) if hasattr(S, 'stdp_on') else float(globals().get('PLAST_GLOBAL', 1.0))
    floor = float(globals().get('PLAST_GLOBAL_FLOOR', 0.0))
    s = float(np.clip(scale, floor, 1.0))
    # Hysteresis/floor (evita flap su micro variazioni)
    EPS = 1e-3
    if abs(s - prev) < EPS:
        return prev, prev
    
    # states application
    prev_raw = float(globals().get('PLAST_GLOBAL', 1.0))
    if hasattr(S, 'stdp_on') and S.stdp_on[:].size:
        prev_raw = float(np.mean(np.asarray(S.stdp_on[:], dtype=float)))
    prev = 0.0 if not np.isfinite(prev_raw) else prev_raw
    global PLAST_GLOBAL
    PLAST_GLOBAL = s
    if hasattr(S, 'elig'):
        cool = 0.5 + 0.5*s
        if s < prev:            # stai riducendo (plateau/instabilità)
            cool *= 0.9         # raffredda un filo di più
        S.elig[:] *= cool
    
    # update log
    if globals().get('PLAST_LOG', False):
        print(f"[Plasticity Scale] {prev:.3f} → {s:.3f} (cool={cool:.3f})")
    return prev, s

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

    # neurons population management
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
        
# blind seen classes procedure function
def blind_spots(
    cls_seen, cls_errors,
    pair_seen, pair_errors,
    combo_seen, combo_errors,
    taste_map,
    min_seen_cls=10,
    min_seen_pair=5,
    min_seen_combo=5
):
    print("\n================ BLIND SPOTS ANALYSIS ================")
    
    # classi più problematiche
    # --- SINGLE CLASSES ---
    print("\n--- Blind single classes: ---")
    stats_cls = []
    for tid in range(len(cls_seen)):
        seen = cls_seen[tid]
        err  = cls_errors[tid]
        if seen < min_seen_cls:
            continue
        err_rate = err / seen if seen > 0 else 0.0
        stats_cls.append((err_rate, seen, err, tid))
    
    # ordina per error-rate decrescente => si parte vedendo la più problematica
    stats_cls.sort(reverse=True, key=lambda xa: xa[0])  
    
    if verbose_rewards:
        for err_rate, seen, err, tid in stats_cls[:10]:
            print(f"  - {taste_map[tid]:<8} | seen={seen:4d} | err={err:4d} | err_rate={err_rate:.3f}")

    # --- COUPLE CLASSES ---
    print("\n--- Blind couple classes: ---")
    num_eff = pair_seen.shape[0]
    stats_pairs = []
    for ia in range(num_eff):
        for ja in range(ia+1, num_eff):
            seen = pair_seen[ia, ja]
            err  = pair_errors[ia, ja]
            if seen < min_seen_pair:
                continue
            err_rate = err / seen if seen > 0 else 0.0
            stats_pairs.append((err_rate, seen, err, ia, ja))

    stats_pairs.sort(reverse=True, key=lambda xa: xa[0])
    
    if verbose_rewards:
        for err_rate, seen, err, ia, ja in stats_pairs[:15]:
            print(f"  - ({taste_map[ia]} + {taste_map[ja]}) | seen={seen:4d} | | err={err:4d} |  | err_rate={err_rate:.3f}")

    # --- COMBO CLASSES ---
    print("\n--- Blind (>= 3 tastes) combo classes:")
    stats_combo = []
    for key, seen, in combo_seen.items():
        if seen < min_seen_combo:
            continue
        if len(key) < 3:
            continue
        err  = combo_errors.get(key, 0)
        err_rate = err / seen if seen > 0 else 0.0
        stats_combo.append((err_rate, seen, err, key))
        
    stats_combo.sort(reverse=True, key=lambda xa: xa[0])
    
    if verbose_rewards:
        for err_rate, seen, err, key in stats_combo[:15]:
            names = "+".join(taste_map[ts] for ts in key)
            print(f"  - [{names}] | seen={seen:4d} | err={err:4d} | err_rate={err_rate:.3f}")

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
'''Riassunto "a cosa serve cosa":
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
    frac = rng.uniform(co_frac[0], co_frac[1]) # co-taste random fraction in range
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

# dopamine DA clamp function
DA_MIN = 0.2   # o 0.0
DA_MAX = 1.0
def clamp_DA_f(mod):
    val = float(mod.DA_f[0])
    val = max(DA_MIN, min(DA_MAX, val))
    mod.DA_f[:] = val

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

# 1. Variante senza pesi => ripristina tutto come nella sua simile, tranne S.w[:]
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

# col norm utilities management
# 1. 0..1: quanto 'forte' è il soft-pull di curriculum.
def arm_soft_pull(strength: float):
    
    global soft_pull_strength
    soft_pull_strength = float(np.clip(strength, 0.0, 1.0))

# 2. Fai decadere la forza se vuoi (rate=0 => step-function).
def soft_pull_decay(rate=0.0):
    global soft_pull_strength
    if rate > 0.0:
        soft_pull_strength *= (1.0 - rate)
    else:
        # una-tantum (equivalente a cooldown=1)
        soft_pull_strength = 0.0

# 3. pre calcolatore degli step di normalizzazione
def next_colnorm_after(s, every, phase):
    return s + ((phase - (s % every)) % every)


#  Alpha dinamico per l'EMA lunga della performance.
#    - all'inizio (step ~ 0) ≈ alpha_max (reagisce forte)
#    - dopo 'decay_steps' trial ≈ alpha_min (curva liscia/stabile)
def get_alpha_long(step, alpha_min=0.03, alpha_max=0.15, decay_steps=2000):
    t = min(step / float(decay_steps), 1.0)
    return alpha_max * (1.0 - t) + alpha_min * t

# 4. gate per la normalizzazione di colonna
def run_col_norm(step) -> bool:
    
    global freeze_until
    
    if not (use_col_norm and connectivity_mode == "dense"):
        return False
    
    if step <= freeze_until:
        if verbose_rewards:
            print(f"\n[COL-NORM] skip: freeze window (step={step} <= freeze_until={freeze_until})")
        return False
    
    if step <= NORM_WARMUP:
        if verbose_rewards: print(f"\n[COL-NORM] skip: warmup ({step} <= {NORM_WARMUP})")
        return False
    
    phase = (step % col_norm_every)
    phase_ok = (phase == COLNORM_PHASE) or (phase == (COLNORM_PHASE + 1) % col_norm_every)
    if not phase_ok:
        if verbose_rewards:
            print(f"\n[COL-NORM] skip: phase win (step%{col_norm_every}={phase} not in {{{COLNORM_PHASE},{(COLNORM_PHASE+1)%col_norm_every}}})")
        return False

    last_struct_step = max(
        last_homeo_step,
        last_colnorm_step,
        (last_decay_step if last_decay_step is not None else -10**9),
        last_epi_step
    )
    gap = step - last_struct_step
    if gap <= GLOBAL_MIN_GAP:
        if verbose_rewards: print(f"\n[COL-NORM] skip: GLOBAL_MIN_GAP (gap={gap} <= {GLOBAL_MIN_GAP})")
        return False
    if (step - last_homeo_step) <= SAFE_GAP:
        if verbose_rewards: print(f"\n[COL-NORM] skip: SAFE_GAP vs homeo")
        return False
    if last_decay_step is not None and (step - last_decay_step) <= DECAY_GAP:
        if verbose_rewards: print(f"\n[COL-NORM] skip: DECAY_GAP")
        return False
    if (step - last_epi_step) <= EPI_SAFE_GAP:
        if verbose_rewards: print(f"\n[COL-NORM] skip: EPI_SAFE_GAP")
        return False
    if soft_pull_strength > SOFT_PULL_STRONG_THR:
        if verbose_rewards: print(f"\n[COL-NORM] skip: just_soft_pulled") 
        return False
    return True

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
    #saved_stdp = float(S.stdp_on[0]) # saved current plasticity state
    #S.stdp_on[:] = 0.0

    saved_gamma = read_shared_scalar(S, 'gamma_gdi')
    def read_stdp_on():
        try:
            return float(S.stdp_on[0])
        except Exception:
            return float(S.stdp_on)  # scalar fallback
    saved_stdp = read_stdp_on()
    S.stdp_on[:] = 0.0
    saved_noise = (pg_noise.rates * 1.0).copy() # the same with noise
    pg_noise.rates = 0 * b.Hz

    # list of lists to collect all negative spikes for each class
    tmp_spikes = [[] for _ in range(num_tastes-1)] 
    pmr_list, h_list, gap_list = [], [], [] # lists to collect class metrics

    def _one_trial(vix):
        # no UNKNOWN
        set_stimulus_vector(vix, include_unknown=False)
        prev = spike_mon.count[:].copy() # saving previous trial spikes
        net.run(dur) # starting simulation
        diff = (spike_mon.count[:] - prev).astype(float) # misuring current spikes with substract previous to current ones
        # evita quantili su array vuoti / degenerate
        if not np.any(diff):
            pmr_list.append(0.0)
            h_list.append(np.log(num_tastes-1))
            gap_list.append(0.0)
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

    # safeguard for small samples
    def _q(x, q, default=0.0):
        return float(np.quantile(x, q)) if len(x) else default

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
                neg_q = 0.999 if idx not in (1, 4) else 0.995 # bitter and umami: avoid too high thresholds
                thr_vec[idx] = max(float(thr_vec[idx]), _q(tmp_spikes[idx], neg_q, 0.0))

    # open-set data-driven thresholds
    # (=> if during the test PMR/H/gap are inside the "negative typical part", refusing)
    PMR_thr_auto = float(np.quantile(pmr_list, 0.98))  # soglia più bassa -> meno trigger
    H_thr_auto   = float(np.quantile(h_list,  0.99))  # entropia deve essere davvero alta
    gap_thr_auto = float(np.quantile(gap_list, 0.98))  # gap molto basso per trigger

    # restore states after OOD
    if saved_gamma is not None:
        write_shared_scalar(S, 'gamma_gdi', saved_gamma)
    pg_noise.rates = saved_noise
    S.stdp_on[:] = saved_stdp

    # 0.997 per-class quantile on negatives OOD/NULL
    ood_q = np.array([
        (np.quantile(tmp_spikes[idx], 0.9995) if len(tmp_spikes[idx]) else 0.0)
        for idx in range(num_tastes-1)
    ], dtype=float)

    return PMR_thr_auto, H_thr_auto, gap_thr_auto, ood_q

# more often a class is inside hedonic window (high gwin_ema) -> less push (conservative)
def meta_scale(idx):
    return np.clip(meta_min + (meta_max - meta_min) * (1.0 - float(gwin_ema[idx])), meta_min, meta_max)

# Aggiorna meta_min/meta_max in base alla confidenza di specie conf_s (0..1),
# ma con intensità che diminuisce a ogni evento epigenetico.
def structural_epi_update(step, conf_s: float):
   
    global meta_min, meta_max, n_epi_events, last_epi_step

    # Calcolo del nuovo range metaplastico "target" in base alla performance lenta
    meta_min_new = float(EPI_META_MIN_LO + (EPI_META_MIN_HI - EPI_META_MIN_LO) * conf_s)
    meta_max_new = float(EPI_META_MAX_LO + (EPI_META_MAX_HI - EPI_META_MAX_LO) * conf_s)

    # Prima macro-mossa forte, poi sempre più delicate
    scale = 1.0 / (1.0 + n_epi_events)

    meta_min = (1.0 - scale) * meta_min + scale * meta_min_new
    meta_max = (1.0 - scale) * meta_max + scale * meta_max_new

    n_epi_events += 1
    last_epi_step = step

    if verbose_rewards:
        print(
            f"\n[STRUCTURAL EPIGENETICS] → event #{n_epi_events} @ step={step} "
            f"conf={conf_s:.3f} → meta_min={meta_min:.3f}, meta_max={meta_max:.3f}"
        )

# Evento raro e condizionato:
#   - non troppo presto nel training
#   - solo ogni EPI_FREQ step
#   - solo se la performance lunga è in plateau
#   - solo se si è lontani da altri interventi strutturali
def maybe_epi_update(step: int) -> bool:
    
    # if is off, no epigenetics
    if not EPI_ON:
        return False
    
    # no epigenetics too soon => dynamic scanning of the right period
    if step < 0.20 * TOTAL_TRAIN_STEPS:
        return False
    
    # non aligned temporal slot
    if step % EPI_FREQ != 0:
        return False
    
    # enough history achieved
    if len(epi_ema_hist) < 300:
        return False
    
    # sliding window for epigenetics history based
    window = list(epi_ema_hist)[-300:]
    delta = max(window) - min(window)
    # if long EMA is still changing => no macro-mutations
    if delta > 0.03:
        return False
    
    # distanza da altri eventi strutturali (homeo / col_norm / decay / epi)
    last_struct_step = max(
        last_homeo_step,
        last_colnorm_step,
        (last_decay_step if last_decay_step is not None else -10**9),
        last_epi_step,
    )
    if (step - last_struct_step) <= EPI_GLOBAL_MIN_GAP:
        return False
    
    # if guards are passed => then epigenetics
    conf_s = float(np.clip(epi_perf_ema, 0.0, 1.0))
    structural_epi_update(step, conf_s)
    
    return True

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
# NB: accumuliamo in 'elig' (che poi decresce con Te). Il segno finale lo dà il rinforzo r
#     (positivo -> LTP; negativo -> LTD)
S = b.Synapses(
    pg, taste_neurons,
    model='''
        w            : 1
        cr_gate      : 1                 # gate per-sinapsi (>=0) -> 1.0 = neutro, >1 potenzia, <1 smorza
        x_stp        : 1                 # risorsa disponibile (0..1)
        u            : 1                 # utilizzo corrente (0..1)
        u0           : 1                 # set-point di u
        uinc         : 1                 # incremento per spike
        tau_rec      : second            # STD recovery
        tau_facil    : second            # STF decay
        ex_scale_stp : 1                 # gain STP
        
        # versioni effettive (calcolate on-the-fly)
        Aplus   : 1
        Aminus  : 1
        
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
        ge_post += (w * u * x_stp * cr_gate * g_step_exc * ex_scale * ex_scale_stp) / (1.0 + gamma_gdi * gdi_eff_post)        
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
        'A2p':  0.0065,   # pair LTP => prima 0.0065, poi 0.0070
        'A3p':  0.0070,   # triplet LTP => prima 0.0070, poi 0.0075
        'A2m': -0.0070,   # pair LTD => prima -0.0070, poi -0.0075
        'A3m': -0.0023,   # triplet LTD
    }
)

# STP continual recovery between spikes: x grow up, u go back to U0
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
taste_neurons.gdi_half   = 1.60 

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
                   on_pre='ge_post += gain_unk * (0.30*nS)',
                   namespace={'npt': NEURONS_PER_TASTE, 'unk': unknown_id}
) # small but only on real UNKNOWN tastes
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
# IF UNKNOWN IS PLASTIC LIKE NOW => it's necessary to connect it with all of the rest synapses
# TO AVOID INTERFERENCE DURING OPEN-SET TESTING
#inhibitory_S.connect('i//npt != j//npt')

# UNKNOWN is not connected to the inhibition network to avoid interference during open-set
#inhibitory_S.connect('i//npt != j//npt and i//npt != unk and j//npt != unk')

# For dense connectivity mode with multiple neurons per taste, intra-pop inhibition is disabled
# we want cooperation inside the same taste population
# GDI modulation of WTA inhibition strength
USE_INTRA_WTA = False
inhibitory_S.connect('i != j and i//npt != unk and j//npt != unk') # inter-pop and intra-pop compete
inhibitory_S.inh_scale = 0.60 # with GDI installed less WTA inhibition
if not USE_INTRA_WTA:
    inhibitory_S.inh_scale['i//npt == j//npt'] = 0.0   # niente competizione intra-pop
else:
    inhibitory_S.inh_scale['i//npt == j//npt'] = 0.3   # versione “sparse” intra-pop
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
    # Drive debole verso la popolazione UNKNOWN (solo j nello slice di UNKNOWN)
    S_unk.connect('j//npt == unk')
    # verso UNKNOWN: j in popolazione UNKNOWN
    '''slu = taste_slice(unknown_id)
    for ts in range(num_tastes):
        if ts == unknown_id:
            continue
        S_unk.connect(i=ts, j=np.arange(slu.start, slu.stop))'''

# GDI Synapses initialization
S.gamma_gdi = gamma_gdi_0
S_noise.gamma_gdi = gamma_gdi_0
# UNKNOWN demand gain initialization
S_unk.gain_unk = 0.0
# craving initialization
S.cr_gate[:] = 1.0

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
PURE_WARMUP_EPOCHS = 10
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

# Static Oversampling for hard-mixes (Train-only phase 3)
HARD_MIX_DAF = 4 # how many variants per-combo
SPICY_MIXES = [
    (0,5,6),  # SWEET+FATTY+SPICY
    (3,5,6),  # SOUR+FATTY+SPICY
    (0,4,6),  # SWEET+UMAMI+SPICY
    (1,3,6),  # BITTER+SOUR+SPICY
    (2,5,6),  # SALTY+FATTY+SPICY
    (3,4,6)   # SOUR+UMAMI+SPICY
]
for _ in range(HARD_MIX_DAF * n_repeats):
    for (ia, ja, ka) in SPICY_MIXES:
        va, ids, lab = make_asymmetric_triple(ia, ja, ka,
                                              amp_hi=rng.integers(260, 321),
                                              rng=rng)
        va = jitter_active(va, frac=0.12, rng=rng)
        va = global_gain(va, lo=0.9, hi=1.15, rng=rng)
        mixture_train.append((va, ids, lab + " (SPICY-MIXES)"))

# utilities per costruire dataset in maniera intelligente (prima gusti singoli, poi coppie ecc.)
def count_unique(stimulus):
    _, ids, _ = stimulus
    return (len(set(ids)))

# riconoscitori per coppie, triple ecc.
def is_pair(stimulus):
    return count_unique(stimulus) == 2

def is_single(stimulus):
    return count_unique(stimulus) == 1

def is_hard(stimulus):
    return count_unique(stimulus) >= 3

# costruzione liste in maniera incrementale
# 1. single
train_1 = pure_warmup + pure_train
# 2. easy pairs and single
easy_pairs = [stimls for stimls in mixture_train
              if is_pair(stimls) and not is_hard(stimls)]
train_2 = train_1 + easy_pairs
# 3. hard mixes
hard_mix = [stimls for stimls in mixture_train
            if is_hard(stimls)]
train_3 = train_2 + hard_mix

# shuffle interno per ogni fase
rng.shuffle(train_1)
rng.shuffle(train_2)
rng.shuffle(train_3)

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
def get_training_list_for_phase():
    if not CURRICULUM_MODE:
        return train_3  # cioè tutto
    if curr_phase == 1:
        return train_1
    elif curr_phase == 2:
        return train_2
    else:
        return train_3

# 1. Train
# DATA-DRIVEN EPOCHS PER LE FASI
PHASE1_EPOCHS = 1.0   # quante volte voglio passare su train_1
PHASE2_EPOCHS = 1.0   # quante volte su train_2 prima di sbloccare la 3
PHASE3_EPOCHS = 2.0   # quante volte su train_3 (poi eventualmente early-stopping)

phase1_max_steps = int(PHASE1_EPOCHS * len(train_1))
phase2_max_steps = int(PHASE2_EPOCHS * len(train_2))
phase3_max_steps = int(PHASE3_EPOCHS * len(train_3))

# Train sets composition logs
print("\nTRAINING SET:")
print(f"Phase1 trials (singles): {phase1_max_steps}")
print(f"Phase2 trials (+couples): {phase2_max_steps}")
print(f"Phase3 trials (full):    {phase3_max_steps}")

# steps totali (prima dell'early stopping)
TOTAL_TRAIN_STEPS = phase1_max_steps + phase2_max_steps + phase3_max_steps
print(f"\n[CURRICULUM] → TOTAL_TRAIN_STEPS = {TOTAL_TRAIN_STEPS}")
   
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
S.x[:] = 0; S.xbar[:] = 0
S.y[:] = 0; S.ybar[:] = 0
S.stdp_on[:] = 1.0
S.elig[:]  = 0
ema_cop_m1 = np.zeros(num_tastes-1)  # E[x] quando la classe � presente in mix (co-presenza)
ema_cop_m2 = np.zeros(num_tastes-1)  # E[x^2]
n_noti = unknown_id   # max available tastes
# reset GDI
taste_neurons.v[:] = EL
sim_t0 = time.perf_counter()
step = 0

# little procedure that helps to calculate SNN's blind state on classes
# seen classes
cls_seen = np.zeros(num_tastes, dtype=int)
cls_errors = np.zeros(num_tastes, dtype=int)

# seen pairs
pair_seen = np.zeros((num_tastes, num_tastes), dtype=int)
pair_errors = np.zeros((num_tastes, num_tastes), dtype=int)

# seen combos
combo_seen   = Counter()
combo_errors = Counter()

#  col_norm_target
if use_col_norm and NEURONS_PER_TASTE == 1 and connectivity_mode == "dense" and col_norm_target is None:
    # expected fan-in: all the pre-synaptics except for UNKNOWN
    fanin = (num_tastes - 1)
    init_mean = float(np.mean(S.w[:])) if len(S.w[:]) > 0 else 0.5
    col_norm_target = init_mean * fanin
    # target clamp
    col_norm_target = float(np.clip(col_norm_target, 0.5*fanin*0.2, 1.5*fanin*0.8))
    if verbose_rewards:
        print(f"\ncol_norm_target → auto={col_norm_target:.3f} (fanin={fanin}, init_mean={init_mean:.3f})")

################ TRIAL CYCLE ##################

# ablation conditions
# set_ach(False)   # ablation ACh
# set_gaba(False)  # ablation GABA
# utility flags
last_homeo_step   = -10
last_epi_step     = -10
last_colnorm_step = -10
freeze_until = -10**9  # nessun freeze attivo all'inizio
freeze_steps = 0
THR_RATIO_BASE = threshold_ratio
last_decay_step   = None
just_soft_pulled  = False
complex_mix_counter = 0
elig_cooldown     = 0

# CURRICULUM DATASET CONTROLLER
CURRICULUM_MODE = True
curr_phase = 1
# contatori per fase (per indicizzare le liste)
step_phase1 = 0
step_phase2 = 0
step_phase3 = 0
# HARD MIXES DINAMICI (solo FASE 3)
HARD_FREQ = 6  # 1 trial su 6 in fase 3 sarà un hard-mix on-the-fly

# dynamic on-the-fly hard SPICY mix
def sample_hard_mix(rng):
    ia, ja, ka = SPICY_MIXES[rng.integers(0, len(SPICY_MIXES))]
    va, ids, lab = make_asymmetric_triple(
        ia, ja, ka,
        amp_hi=rng.integers(260, 321),
        rng=rng
    )
    va = jitter_active(va, frac=0.12, rng=rng)
    va = global_gain(va, lo=0.9, hi=1.15, rng=rng)
    return va, ids, lab + " (HARD-ON-THE-FLY)"

# TRAINING MAIN LOOP
for step in range(1, TOTAL_TRAIN_STEPS + 1):
    
    # avanza di fase in modo DATA-DRIVEN (in base ai passi per fase)
    if CURRICULUM_MODE:
        if curr_phase == 1 and step_phase1 >= phase1_max_steps:
            curr_phase = 2
            print("\n[CURRICULUM] → FASE 2 (singoli + coppie)")
        if curr_phase == 2 and step_phase2 >= phase2_max_steps:
            curr_phase = 3
            print("\n[CURRICULUM] → FASE 3 (mix heavy + HARD_MIXES)")

    # scegli la lista di training e l'indice locale
    if curr_phase == 1:
        train_list = train_1
        idx_local  = step_phase1 % len(train_1)
        step_phase1 += 1
    elif curr_phase == 2:
        train_list = train_2
        idx_local  = step_phase2 % len(train_2)
        step_phase2 += 1
    else:
        train_list = train_3
        idx_local  = step_phase3 % len(train_3)
        step_phase3 += 1

    # sample base dalla lista di fase
    input_rates, true_ids, label = train_list[idx_local]
    # se vediamo spicy incrementiamo il contatore
    if spicy_id in true_ids:
        spicy_seen += 1

    # OVERSAMPLING DINAMICO HARD_MIXES: solo in fase 3, 1 trial su HARD_FREQ
    if (curr_phase == 3) and (step_phase3 % HARD_FREQ == 0):
        input_rates, true_ids, label = sample_hard_mix(rng)

    soft_pull_decay(rate=0.0)
    # early soft clamp of GDI for first 300 trials to avoid excessive inhibition at start
    # because weights are still low and GDI can grow up too much
    # this is important especially if the initial weights are high
    # after 300 trials the GDI is free to grow up as needed
    if step <= 300: 
        S.gamma_gdi = min(S.gamma_gdi, 0.14)
        S_noise.gamma_gdi = S.gamma_gdi
    
    # initialize updating flags
    decay_requested   = False
    decay_cache       = None
    colnorm_requested = False
    homeo_tick        = False
    homeo_cache       = None
    can_epi           = False
    epi_cache         = None
    epi_tick          = False
    
    # ----------------- CURRICULUM SEMPLICE SU MIXTURE -----------------
    '''
    - When the network is "healthy" (ema_perf close to the best), the 4–5 mixes are run normally.
    - When it is in crisis (ema_perf drops), the 4–5 mixes are downgraded:
        a) either in single warmups,
        b) or in triples/pairs.'''
     # Ritmo 1–2 mix complessi -> 1 warmup sulla classe peggiore
    if complex_mix_counter >= 2:
        # stima "classe peggiore" dalla EMA dei positivi (più bassa = più scarsa)
        # se non hai altra metrica per classe, va benissimo così
        with np.errstate(divide='ignore', invalid='ignore'):
            # proxy semplicissimo: -ema_pos_m1 => argmax = più scarsa
            score_vec = ema_pos_m1.copy()
        worst_id = int(np.argmin(score_vec[:unknown_id]))

        # costruisci uno stimolo puro sulla classe peggiore
        new_rates = np.zeros_like(input_rates)
        amp_warm = max(240.0, float(np.max(input_rates[:unknown_id]) if np.any(input_rates[:unknown_id] > 0) else 260.0))
        new_rates[worst_id] = amp_warm

        input_rates = new_rates
        true_ids    = [worst_id]
        label      += f" [curriculum: warmup {taste_map[worst_id]}]"
        arm_soft_pull(0.9) # forte: stai cambiando drasticamente lo stimolo
        complex_mix_counter = 0

        # aggiorna k_tastes per coerenza con il resto del codice
        k_tastes = 1
        heavy_mix = False
        is_complex_mixture = False
        
    # numero di gusti nel target
    k_tastes = len(true_ids)
    heavy_mix = (k_tastes >= 4)  # 4-5 gusti => difficult combination

    # Contatore mix complessi (>=3 gusti)
    is_complex_mixture = (k_tastes >= 3)
    if is_complex_mixture:
        complex_mix_counter += 1
    else:
        complex_mix_counter = max(0, complex_mix_counter - 1)
        
    # "rete in difficoltà" se l'EMA corrente è molto sotto il best
    net_struggling = (
        (step > DECAY_WARMUP_STEPS) and
        (best_ema_perf > 0.0) and
        (ema_perf < best_ema_perf - 0.05)
    )
    # 50% dei casi -> faccio warmup singolo sulla prima classe del mix
    if heavy_mix and net_struggling:
        if rng.random() < 0.5:
            main = true_ids[0]
            new_rates = np.zeros_like(input_rates)
            new_rates[main] = max(240.0, float(input_rates[main]))
            input_rates = new_rates
            true_ids = [main]
            label += " [curriculum: single-warmup]"
            arm_soft_pull(0.8)  # strong
        else:
            # 50% -> riduco a mix 2–3 gusti (più leggero)
            max_k = 3
            k_new = min(max_k, k_tastes - 1)
            k_new = max(2, k_new)  # almeno 2 gusti
            subset = list(rng.choice(true_ids, size=k_new, replace=False))

            new_rates = np.zeros_like(input_rates)
            for tid in subset:
                new_rates[tid] = input_rates[tid]
            input_rates = new_rates
            true_ids = subset
            label += " [curriculum: light-mix]"
            arm_soft_pull(0.5)
    
    # progress bar + chrono + ETA
    frac   = step / TOTAL_TRAIN_STEPS
    filled = int(frac * progress_bar_len)
    bar    = '�'*filled + '�'*(progress_bar_len - filled)

    elapsed = time.perf_counter() - sim_t0
    eta = (elapsed/frac - elapsed) if frac > 0 else 0.0

    if len(true_ids) == 1:
       reaction = taste_reactions[true_ids[0]]
       msg = (f"[{bar}] {int(frac*100)}% | Step {step}/{TOTAL_TRAIN_STEPS} | {label} | {reaction}"
           f" | t={fmt_mmss(elapsed)} | ETA={fmt_mmss(eta)}")
    else:
       msg = (f"[{bar}] {int(frac*100)}% | Step {step}/{TOTAL_TRAIN_STEPS} | {label} (mixture)"
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
                wk = np.clip(wk, 0.92, 1.10)  # pinna l'effetto per stabilità
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
            increment = ATTN_BIAS_GAIN_MV * debito[ta]
            increment = min(increment, ATTN_BIAS_CAP_MV)
            bias_ta = base_bias_mv - increment
            taste_neurons.theta_bias[sl] = bias_ta * b.mV
    else:
        taste_neurons.theta_bias[:] = base_bias_mv * b.mV

    # craving abilitation during training
    if ENABLE_DESIRE:
        # soft gain clamp
        S.cr_gate[:] = np.clip(S.cr_gate[:], 0.5, CRAVE_MAX)  # 0.5–1.5 nel tuo setup
        # fattore per gusto (solo classi note)
        # stesso mix della tua formula per gli score (0.5 KF + 0.5 KS)
        boost = 1.0 + 0.5*CRAVE_KF*crave_f[:unknown_id] + 0.5*CRAVE_KS*crave_s[:unknown_id]
        # scrivi cr_gate per sinapsi con pre-sinaptico = gusto t (puoi limitarlo ai j della sua popolazione, ma non è obbligatorio)
        for ts in range(unknown_id):
            sl = taste_slice(ts)
            S.cr_gate[f"(i == {ts}) and (j >= {sl.start}) and (j < {sl.stop})"] = float(np.clip(boost[ts], 0.25, CRAVE_MAX))

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
        gamma_val = min(gamma_val, 0.09)
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
    
    # Meta-meta plasticity => Plastic Homeostasis Turrigiano-like (firing-rate based)
    # instant firing per-neuron
    r_inst = diff_counts.astype(float) / trial_T_sec
    # firing rate slow EMA
    homeo_r_avg = (1.0 - HOMEOSTASIS_ALPHA) * homeo_r_avg + HOMEOSTASIS_ALPHA * r_inst
    # plastic homeostasis trigger conditions
    homeo_tick = (
        HOMEOSTASIS_ON and HOMEOSTASIS_EVERY > 0 and step > 300 and
        ((step % HOMEOSTASIS_EVERY) == HOMEOSTASIS_PHASE)
    )
    
    can_prepare_homeo = homeo_tick and (step > freeze_until)
        
    # Prepara materiali per l'eventuale esecuzione (ma NON eseguire qui)
    if can_prepare_homeo:
        w_all = np.asarray(S.w[:], dtype=float)
        j_all = np.asarray(S.j[:], dtype=int)
        eps   = 1e-6
        ratio = HOMEOSTASIS_TARGET_RATE / (homeo_r_avg + eps)
        ratio = np.clip(ratio, 1e-3, 1e3)          # bound duro anti-outlier
        scale_vec = np.power(ratio, HOMEOSTASIS_BETA)
        scale_vec = np.clip(scale_vec, HOMEOSTASIS_SCALE_MIN, HOMEOSTASIS_SCALE_MAX)
        scale_vec[~np.isfinite(scale_vec)] = 1.0   # fallback neutro
        epi_lock = (np.asarray(S.epi_lock[:], dtype=bool)
                if hasattr(S, 'epi_lock') else np.zeros_like(w_all, dtype=bool))
        scale_per_syn = scale_vec[j_all]
        homeo_cache = (w_all, scale_per_syn, epi_lock) 
    else:
        homeo_cache = None
    
    # population aggregation
    dc_pop = population_scores_from_counts(diff_counts)
    # OOD/NULL hard-negative mining on-the-fly
    #E = float(np.sum(diff_counts[:unknown_id]))
    #pmr = (float(diff_counts[:unknown_id].max()) / (E + 1e-9)) if E>0 else 0.0
    E_pop = float(np.sum(dc_pop[:unknown_id]))
    pmr_pop = (float(dc_pop[:unknown_id].max()) / (E_pop + 1e-9)) if E_pop>0 else 0.0

    ######### REWARDING PHASE #########
    # trial diffuso/ambiguo => usa gate negativo più severo
    is_diffuse_train = (pmr_pop < 0.45)
    if is_diffuse_train:
        j_all = np.asarray(S.j[:], int)
        for q in range(unknown_id):
            if q in true_ids: # non punisco mai le classi vere
                continue
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
                    S.w[idx] = np.clip(S.w[idx] + r_off * S.elig[idx] * (S.cr_gate[idx] if ENABLE_DESIRE else 1.0), 0.0, 1.0)
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
    if step % 10 == 0 and NEURONS_PER_TASTE == 1 and connectivity_mode == "dense" and step > freeze_until:
        w_all = np.asarray(S.w[:], float)
        i_all = np.asarray(S.i[:], int)
        j_all = np.asarray(S.j[:], int)
        mask_off = (i_all != j_all) & (j_all != unknown_id) & (i_all != unknown_id)
        # off-diagonal pruning SOFT
        gamma_decay_off = 0.8   # 0.7–0.8 ok
        w_floor_off     = 0.005 # pavimento per non andare a 0

        if np.any(mask_off):
            w_off = w_all[mask_off].astype(float)

            # shrink moltiplicativo
            w_off *= gamma_decay_off

            # pavimento: mai esattamente 0 sugli off-diagonal
            small = np.abs(w_off) < w_floor_off
            w_off[small] = np.sign(w_off[small]) * w_floor_off

            w_all[mask_off] = w_off

        # qui il clip non ti ammazza più nulla perché hai il floor sugli off-diag
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
    
    # applica craving desiderio solo alle classi note
    if ENABLE_DESIRE:
        scores[:unknown_id] *= (
            1.0 + 0.5*CRAVE_KF*crave_f[:unknown_id] + 0.5*CRAVE_KS*crave_s[:unknown_id]
        )
    
    scores[unknown_id] = -1e9
    mx = scores.max()
    
    # mix-like metrics analysis
    E_scores = float(np.sum(scores[:unknown_id]))
    top = float(np.max(scores[:unknown_id])) if E_scores > 0 else 0.0
    PMR = top / (E_scores + 1e-9) if E_scores > 0 else 0.0
    p = scores[:unknown_id] / (E_scores + 1e-9) if E_scores > 0 else np.zeros(unknown_id)
    p = np.clip(p, 1e-12, 1.0); p /= p.sum()
    H = float(-(p*np.log(p)).sum())

    # Dopo aver calcolato tp_gate/fp_gate, gestisco mix_like
    is_mix_like = (0.35 <= PMR <= 0.60) and (0.8 <= H <= 1.4)
    thr_ratio_cur = THR_RATIO_BASE
    if is_mix_like:
        thr_ratio_cur *= 0.85   # decoder più permissivo nei mix
    rel = thr_ratio_cur * mx
    
    # define TP and FP gates
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
        tp_gate_i = max(TP_FLOOR_SCALE * min_spikes_for_known * ema_factor,
                ema_pos_m1[idx] - EMA_POS_SD_COEFF * ema_factor * pos_sd_i)
        fp_gate_i = max(thr_ema_i, rel)
        # absolute max per classi "calde"
        if is_mix_like:
            fp_gate_i = max(fp_gate_i, 0.18 * mx + 3)
        else:
            fp_gate_i = max(fp_gate_i, 0.20 * mx + 4)
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
                if not np.any(mask_col):
                    continue

                # SOLO OFF-DIAGONAL dentro la colonna q
                mask_off_col = mask_col & (i_all != j_all)
                if not np.any(mask_off_col):
                    continue

                # soft pruning off-diag: shrink moltiplicativo + floor
                gamma_decay_off = 0.8   # più prudente (0.7 se vuoi più aggressivo)
                w_floor_off     = 0.005

                w_off = w_all[mask_off_col].astype(float)
                w_off *= gamma_decay_off

                small = np.abs(w_off) < w_floor_off
                w_off[small] = np.sign(w_off[small]) * w_floor_off

                w_all[mask_off_col] = w_off

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
        thr = thr_ratio_cur * mx
        winners = [idx for idx,c in enumerate(scores) if c >= thr]
        
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
        
    # fallback totale se non si trova altri vincitori
    if not winners:
        top_idx = int(np.argmax(scores[:unknown_id]))
        if scores[top_idx] >= max(0.35 * mx, 0.6 * min_spikes_for_known):
            winners = [top_idx]
        else:
            winners = []  # vero UNKNOWN

    # SPICY aversion management (if SPICY is present among true or winner ids)
    is_spicy_present = (spicy_id in true_ids) or (spicy_id in winners)

    # nuovo gate: niente avversione troppo presto / senza esperienza su SPICY
    can_spicy_aversion = (
        is_spicy_present and
        #is_mix_trial and   # attivare se si vuole dare avversione SPICY solo per mix => meno biologico ma più stabile
        (step > SPICY_AVERSION_WARMUP_STEPS) and
        (spicy_seen >= SPICY_MIN_SEEN) and
        (curr_phase >= 2)               # niente avversione in Fase 1 per stabilizzare
    )

    # MODULAZIONE DINAMICA in base ai mix SPICY
    # spicy_combo_bad_ema ∈[0,1]; se > SPICY_BAD_THR vuol dire che
    # storicamente i mix con SPICY hanno errori alti
    if spicy_combo_bad_ema > SPICY_BAD_THR:
        bad_factor = (spicy_combo_bad_ema - SPICY_BAD_THR) / (1.0 - SPICY_BAD_THR)
        bad_factor = float(np.clip(bad_factor, 0.0, 1.0))
    else:
        bad_factor = 0.0
        
    # p_aversion, HT e thr_spice_kick diventano più forti quando SPICY fa schifo nei mix
    p_av_base_dyn  = p_aversion_base * (1.0 + SPICY_P_GAIN  * bad_factor)
    ht_pulse_dyn   = ht_pulse_aversion * (1.0 + SPICY_HT_GAIN  * bad_factor)
    thr_spice_dyn  = thr_spice_kick    * (1.0 + SPICY_THR_GAIN * bad_factor)

    if can_spicy_aversion:
        happened, p_now = spicy_aversion_triggered(
            taste_neurons, mod, spicy_id,
            p_base=p_av_base_dyn,
            slope=p_aversion_slope,
            cap=p_aversion_cap,
            trait=profile['spicy_aversion_trait'],
            k_hun=profile['k_hun_spice'],
            k_h2o=profile['k_h2o_spice']
        )

        if happened:
            mod.HT[:]  += ht_pulse_dyn
            mod.DA_f[:] *= da_penalty_avers
            clamp_DA_f(mod)
            mod.DA_t[:] *= da_penalty_avers
            sl = taste_slice(spicy_id)
            taste_neurons.thr_spice[sl] = taste_neurons.thr_spice[sl] + thr_spice_dyn
            if verbose_rewards:
                print(
                    f"\n[SPICY-AVERSION] → p={p_now:.2f} (bad_ema={spicy_combo_bad_ema:.2f}, "
                    f"factor={bad_factor:.2f}) → HT+{ht_pulse_dyn:.2f}, "
                    f"DA×{da_penalty_avers}, thr_spice+={thr_spice_dyn:.3f}"
                )
                
    # ANALYSIS of the scores distribution to detect mix-like patterns
    # prima di applicare i rinforzi, analizza il pattern di punteggi
    # stai per decidere i winner e dare rinforzi. Allentare WTA/GDI prima del rinforzo permette a più classi vere di sparare insieme nei mix (e poi ricevere reward).
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

    # CREATE BLIND-SEEN PROCEDURE FOR TRAINING TASTES
    # 1) filtra solo i gusti “veri” (escludo UNKNOWN)
    true_idx = [tid for tid in true_ids if tid < num_eff]
    # 2) definizione di trial corretto / errato
    correct = bool(jacc_proxy >= JACC_THR)
    # 3) aggiornamento STATISTICHE solo per esempi veri di train
    if "NULL" not in label and "OOD" not in label:
        # a) per-CLASSE: ogni gusto presente nel mix
        for ids in true_idx:
            if ids >= num_eff:
                continue
            cls_seen[ids] += 1
            if not correct:
                cls_errors[ids] += 1
        # b) per-COPPIE: tutte le coppie (i<j) dentro al mix
        for ax in range(len(true_idx)):
            for bx in range(ax+1, len(true_idx)):
                ids, jds = true_idx[ax], true_idx[bx]
                if ids >= num_eff or jds >= num_eff:
                    continue
                pair_seen[ids, jds] += 1
                if not correct:
                    pair_errors[ids, jds] += 1
        # c) per-COMBO (coppie, triple, quad, quint...) → un’unica chiave
        key = tuple(sorted(ids for ids in true_idx if ids < num_eff))
        if len(key) >= 2:   # ignora i singoli
            combo_seen[key] += 1
            if not correct:
                combo_errors[key] += 1
    
    # UPDATE EMA "quanto fanno schifo i mix con SPICY"
    # usiamo SOLO le combo che contengono SPICY e hanno almeno 2 gusti
    worst_err = 0.0
    any_combo = False

    for key, seen in combo_seen.items():
        # key è una tupla di id gusto, es: (0, 5, 6)
        if spicy_id not in key:
            continue
        if len(key) < 2:  # niente singoli
            continue
        if seen < SPICY_BAD_MIN_SEEN:
            continue

        err_rate = combo_errors[key] / max(1, seen)  # ∈[0,1]
        if err_rate > worst_err:
            worst_err = float(err_rate)
        any_combo = True

    if any_combo:
        spicy_combo_bad_ema = (
            (1.0 - SPICY_BAD_ALPHA) * spicy_combo_bad_ema
            + SPICY_BAD_ALPHA * worst_err
        )

    # EMA della performance per "best snapshot" and EARLY STOPPING
    # 1) Performance istantanea del trial (usa jacc_proxy come metrica 0..1)
    inst_perf = float(jacc_proxy)
    ema_perf = (1.0 - perf_alpha) * ema_perf + perf_alpha * inst_perf
    # dynamic alpha learning rate update during training
    if step < 50:
        alpha_long = 0.15
    else:
        alpha_long = get_alpha_long(
            step - 50, 
            alpha_min=0.03, 
            alpha_max=0.15, 
            decay_steps=2000
        )
        if verbose_rewards:
            print(f"\n[DynamicAlphaUpdate] → alpha_long={alpha_long:.4f} at trial {step}")
    # EMA lunga per decidere i decays
    ema_perf_long = (1.0 - alpha_long) * ema_perf_long + alpha_long * inst_perf
    # aggiorna best_ema_perf / best_ema_step (sull'EMA lunga, non sulla breve)
    if ema_perf_long > best_ema_perf + 1e-6:
        best_ema_perf = ema_perf_long
        best_ema_step = step
    if verbose_rewards:
        print(
            f"\n[Trial {step}] PERF per-class → \n"
            f"ema_inst={inst_perf:.4f} | ema_short={ema_perf:.4f} | ema_long={ema_perf_long:.4f}"
        )
    
    # Structural Epigenetics: precompute tick & can-run (no execution here)
    if EPI_ON:
        # EMA lentissima della performance “di specie”
        epi_perf_ema = (1.0 - EPI_ALPHA) * epi_perf_ema + EPI_ALPHA * jacc_proxy
        # history per plateau detection (uso la EMA lunga globale)
        epi_ema_hist.append(float(ema_perf_long))
    
    # questo gate moltiplica ogni rinforzo r (vedi A4) ⇒ se gate=0, nessun consolidamento
    if DA_GATE_JACC <= 0:
        perf_gate = 1.0
    else:
        perf_gate = float(np.clip(jacc_proxy / DA_GATE_JACC, 0.0, 1.0))
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
    MIN_STEPS_BEFORE_STOP = max(200, int(0.15 * TOTAL_TRAIN_STEPS)) # step dinamici da considerare ad ogni trial

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
                r = (alpha_lr * (1.0 + da_gain * DA_now) * (1.0 + ach_plasticity_gain * ACH_now)) / (1.0 + ht_gain * ht_eff)
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
                    r = 0.33 * alpha_lr * (1.0 + da_gain * DA_now) / (1.0 + ht_gain * HT_now)
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

        # applicazione unica del rinforzo con gating + craving and hedonic desire
        # perf_gate ∈ {0,1} o [0,1] (dipende da come lo calcoli); PLAST_GLOBAL ∈ [0,1]
        cr_scale = 1.0
        gain_crav = (S.cr_gate[idx_list] if (ENABLE_DESIRE and r > 0) else 1.0)
        if ENABLE_DESIRE and (idx != unknown_id):
            cr_scale = crave_scale(idx)
        r_final = perf_gate * PLAST_GLOBAL * cr_scale * r
        if r_final != 0.0 and abs(r_final) * np.max(S.elig[idx_list]) >= 1e-6:
            # vettorizzato su tutta la popolazione di "idx"
            # S.elig è già il 3° fattore (eligibility trace)
            delta_vec = r_final * (S.elig[idx_list] * gain_crav)
            # applica e clampa
            S.w[idx_list] = np.clip(S.w[idx_list] + delta_vec, 0.0, 1.0)
            # svuota le tracce consumate
            S.elig[idx_list] = 0.0

    # A5: OFF-DIAGONAL: punish p->q when q is big FP with starting warmup
    OFFDIAG_WARMUP = 250
    if use_offdiag_dopamine and step >= OFFDIAG_WARMUP and step > freeze_until:
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
        clamp_DA_f(mod)
        mod.DA_t[:] += da_tonic_tail * jacc
    elif jacc > 0.0:
        mod.DA_f[:] += 0.4 * da_pulse_reward * jacc # partial reward
        clamp_DA_f(mod)
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
        best_state = snapshot_state()    # snapshot completo (pesate + stati)
        if w_slow is not None:
            w_slow_best = w_slow.copy()  # se usi consolidamento lento

        patience = 0
        decays_done = 0   # nuovo ciclo “explore”: azzero i decay conteggiati

        # re-opening plasticity in a slower way
        if 'stdp_on' in S.variables:
            curs = float(np.mean(S.stdp_on[:]))
            target_s = 1.0
            set_plasticity_scale(0.50 * curs + 0.50 * target_s)
        if verbose_rewards:
            print(f"NEW LONG BEST ema_perf → {best_score:.4f} at trial {best_step} (+>{IMPROVE_EPS:.4f} increment)")
    else:
        # no best improvement
        patience += 1
        
        # 3a) soft-pull verso il best durante plateau prolungato (bio consolidamento)
        if USE_SOFT_PULL and patience >= max(W_EMA, PLATEAU_WINDOW) and best_state is not None:
            soft_pull_toward_best(best_state, rho=RHO_PULL)
            elig_cooldown = max(elig_cooldown, 2)
            arm_soft_pull(1.0)
        else:
            pass
            
        # 3b) Reduce-on-plateau biologico con gate su EMA lunga
        #    (decay solo se: warmup finito, cooldown rispettato,
        #     EMA lunga scesa rispetto al massimo, e non ho esaurito i decays)
        do_decay = (USE_PLASTICITY_DECAY
            and TRAINING_PHASE
            and step >= DECAY_WARMUP_STEPS
            and (last_decay_step is None or (step - last_decay_step) >= DECAY_COOLDOWN_STEPS)
            and (step - best_ema_step) >= DECAY_PATIENCE_STEPS   # es. 150 step, o == PLATEAU_WINDOW
            and ema_perf_long < (best_ema_perf - DECAY_EPS)
            and decays_done < MAX_PLASTICITY_DECAYS)

        # DECAY: precompute request (no execution here)
        if do_decay:
            old_scale = float(PLAST_GLOBAL)
            proposed  = old_scale * STDP_ON_DECAY_FACTOR
            new_scale = max(STDP_ON_MIN, proposed)
            decay_cache = (old_scale, new_scale)
            decay_requested = True 

        # 3c) Early Stopping finale
        # early stopping solo dopo aver esaurito i decays (o quasi)
        if  USE_EARLY_STOP and patience >= PATIENCE_LIMIT:
            if step >= EARLY_STOP_MIN_FRAC * TOTAL_TRAIN_STEPS:
                if decays_done >= MAX_PLASTICITY_DECAYS or PLAST_GLOBAL <= STDP_ON_MIN + 1e-3:
                    print(f"\n[EARLY STOPPING] → No synaptic plasticity best improvement for "
                        f"{PATIENCE_LIMIT} consecutive trials (best={best_score:.4f} @ trial {best_step}) "
                        f"— stopping training.")
                    # restore best snapshot come già fai ora
                    restore_state(best_state)
                    break
                else:
                    # se non hai ancora usato tutti i decays, non fermare il training:
                    # al massimo lasci al gate di cui sopra decidere un nuovo decay
                    patience = 0  # resetta la pazienza e continua
            else:
                # prima del 70% dei passi, anche se hai plateau, NON fermare:
                patience = 0
                
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
    
    # 3) COL-NORM: valuta UNA SOLA VOLTA la condizione (non richiamare due volte)
    if verbose_rewards and use_col_norm and connectivity_mode == "dense":
        if step > NORM_WARMUP and ((step % col_norm_every) == COLNORM_PHASE):
            nxt = next_colnorm_after(step, col_norm_every, COLNORM_PHASE)
            print(f"[COL-NORM] slot ready @trial={step} (next={nxt}) | soft_pull={soft_pull_strength:.2f}")
    colnorm_requested = bool(run_col_norm(step))
    
    ######################################################################
    # STRUCTURAL MUTEX ORCHESTRATOR for EPIGENETICS, PLASTIC HOMEOSTASIS, PLASTICITY DECAY and COL_NORM L1 (one action per step)
    # near-* per homeo
    near_colnorm_homeo = (last_colnorm_step is not None and ((step - last_colnorm_step) <= GLOBAL_MIN_GAP))
    near_decay_homeo   = (last_decay_step   is not None and ((step - last_decay_step)   <= DECAY_GAP))
    near_epi_homeo     = (last_epi_step     is not None and ((step - last_epi_step)     <= EPI_SAFE_GAP))
    epi_busy           = ('EPI_CONSOLIDATION' in globals()) and bool(EPI_CONSOLIDATION)
    
    # can_homeo basato sul tick + guardie
    can_homeo = (homeo_tick
             and (not near_colnorm_homeo)
             and (not near_decay_homeo)
             and (not near_epi_homeo)
             and (not epi_busy)
             and (homeo_cache is not None))
    
    # Column normalization (incoming synaptic scaling) with population management
    can_col_norm = bool(colnorm_requested)
    # can_decay già impostato con decay_requested
    can_decay = bool(decay_requested)
    
    # PRIORITY: decay > col_norm > homeostasis > epigenetics
        # 1. Stops global escalation (decay),
        # 2. Cleans locally the column if necessary (col_norm),
        # 3. Slowly realigns activity toward the set point (homeostasis),
        # 4. Updates meta ranges very rarely (epigenetics) => probably never used
    # Why decay before col_norm?
        # Decay is global, col_norm is local.
        # Decay (lowering PLAST_GLOBAL/stdp_on) changes the learning gain of the entire system; 
        # col_norm redefines the geometry of the columns (L1/L2). 
        # First adjust the gain, then—in the following steps—do local housekeeping. 
        # It's like adjusting the amplifier's volume before equalizing the individual channels.
    did_action = False
    
    # 1. plasticity decay
    if can_decay:
        old_scale, new_scale = decay_cache
        set_plasticity_scale(new_scale)
        PLAST_GLOBAL   = new_scale
        decays_done   += 1
        freeze_until = step + freeze_steps
        last_decay_step = step
        did_action = True
        if verbose_rewards:
            print(
                f"\n[ReducePlasticityOnPlateau] → ema_long={ema_perf_long:.3f} "
                f"best_ema={best_ema_perf:.3f} → scale={old_scale:.3f}->{new_scale:.3f} "
                f"(decays={decays_done}/{MAX_PLASTICITY_DECAYS})"
            )
    
    # 2. L1 col_norm
    elif can_col_norm:
        # stats BEFORE
        w_before = np.asarray(S.w[:], float)
        j_all    = np.asarray(S.j[:], int)
        # L1 per-colonna (solo classi note)
        L1_before = []
        for q in range(unknown_id):
            L1_before.append(float(np.sum(w_before[j_all == q])))
        
        # L1 col_norm
        col_norm_pop(target=col_norm_target, mode=col_norm_mode, temperature=col_norm_temp,
                 diag_bias_gamma=diag_bias_gamma, floor=col_floor,
                 allow_upscale=col_allow_upscale, scale_max=col_scale_max)
        last_colnorm_step = step
        did_action = True
        
        # to visualize how weights change between before and after
        if verbose_rewards:
            # come cambiano i pesi dopo la normalizzazione L1
            w_after = np.asarray(S.w[:], float)
            L1_after = []
            for q in range(unknown_id):
                L1_after.append(float(np.sum(w_after[j_all == q])))
            
            # deviazione media assoluta dal target (se definito), altrimenti media L1
            if col_norm_target is not None:
                mad_before = float(np.mean([abs(xa - col_norm_target) for xa in L1_before]))
                mad_after  = float(np.mean([abs(xa - col_norm_target) for xa in L1_after]))
                print(f"\n[L1 COL NORM] → "
                    f"mad_before={mad_before:.4f} → mad_after={mad_after:.4f}  "
                    f"(target={col_norm_target})")
            else:
                m_before = float(np.mean(L1_before)); m_after = float(np.mean(L1_after))
                print(f"\n[L1 COL NORM] → "
                    f"meanL1_before={m_before:.4f} → meanL1_after={m_after:.4f}")
            
            # prime 3 colonne più corrette (delta maggiore)
            deltas = [L1_after[qs] - L1_before[qs] for qs in range(unknown_id)]
            topfix = np.argsort(np.abs(deltas))[::-1][:3]
            info   = ", ".join([f"{qs}:{deltas[qs]:+.3f}" for qs in topfix])
            print(f"\n[L1 COL NORM] → top fixes (ΔL1): {info}")
    
    # 3. plastic homeostasis
    elif can_homeo:
        # esecuzione plastic homeostasi
        w_all, scale_per_syn, epi_lock = homeo_cache
        free_mask = ~epi_lock
        w_all[free_mask] *= scale_per_syn[free_mask]
        S.w[:] = np.clip(w_all, 0.0, 1.0)
        last_homeo_step = step
        did_action = True
        if verbose_rewards:
            print("\n[PLASTIC HOMEOSTASIS] → mean r_avg:", np.mean(homeo_r_avg[:unknown_id]), "Hz")
    
    # 4. structural epigenetics
    elif not did_action and EPI_ON:
        # evento raro e condizionato: plateau + distanza dagli altri
        if maybe_epi_update(step):
            did_action = True
    
    #########################################################################################

    # idempotency DA clamp to avoid hidden bugs inside conditional branches
    clamp_DA_f(mod)
    # To monitor the effect of oversampling (dynamic or static), log for taste
    log_population_stats(diff_counts, step=step, label="POST-REWARD") 
    print("\n") 
    # 5) eligibility trace decay among trials
    net.run(pause_duration)
    S.elig[:] = 0
    # flags reset
    elig_cooldown = max(0, elig_cooldown-1)
    just_soft_pulled = (elig_cooldown > 0)

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

# log problematic classes and blind cases
blind_spots(
    cls_seen, cls_errors,
    pair_seen, pair_errors,
    combo_seen, combo_errors,
    taste_map=taste_map,   # es: {0:"SWEET", 1:"BITTER", ...}
    min_seen_cls=10,
    min_seen_pair=5,
    min_seen_combo=5
)

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
PMR_thr, H_thr, gap_thr, ood_q = ood_calibration(n_null=96, n_ood=192, dur=test_duration, gap=20*b.ms, thr_vec=thr_per_class_test)
# clamp minimo delle soglie OOD
PMR_thr = max(PMR_thr, 0.16)   # prima 0.20
H_thr   = max(H_thr,   0.95)   # prima 0.95
gap_thr = max(gap_thr, 0.15)   # prima 0.22
# clipping sul gate di UNKNOWN
S_unk.gain_unk = float(np.clip(0.06 + 0.40*np.tanh(np.mean(ood_q)/12.0), 0.04, 0.12))
# Overshoot OOD rispetto alla soglia corrente del TEST window
overshoot = np.maximum(0.0, ood_q - thr_per_class_test[:unknown_id])
# smorza overshoot con radice e riduci i gain
overshoot = np.sqrt(overshoot)
heat_gain = np.array([0.01, 0.03, 0.00, 0.03, 0.03, 0.00, 0.00]) # per-class heat gain vector
margin = heat_gain * overshoot
# cap to boost
pos_cap = 0.85 * np.maximum(ema_pos_m1 * dur_scale, 1.0)
#thr_per_class_test[:unknown_id] = np.minimum(thr_per_class_test[:unknown_id], pos_cap)
thr_per_class_test[:unknown_id] += margin # add some heat only where needed
# extra margin to SALTY/SOUR because they are more "leaky"
'''thr_per_class_test[2] = max(1.0, thr_per_class_test[2] - 1.5)  # SALTY
thr_per_class_test[3] = max(1.0, thr_per_class_test[3] - 1.0)  # SOUR
thr_per_class_test[5] = max(1.0, thr_per_class_test[5] - 0.5)  # FATTY
thr_per_class_test[6] = max(1.0, thr_per_class_test[6] - 1.0)  # SPICY
'''
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
use_test_noise = True
USE_ATTENTIONAL_BIAS = False
test_baseline_hz = baseline_hz if use_test_noise else 0.0
# Neuromodulator parameters in test
k_inh_HI_test   = -0.08
k_inh_HT_test = 0.4
k_theta_HI_test = -0.5
k_ex_HI_test    = 0.15
k_noise_HI_test = 0.15
p_aversion_base  = 0.0 # freezing SPICY aversion
p_aversion_slope = 0.0
p_aversion_cap   = 0.0
ht_pulse_aversion_test = 0.5
taste_neurons.v[:] = EL
taste_neurons.ge[:] = 0 * b.nS
taste_neurons.gi[:] = 0 * b.nS
taste_neurons.s[:]  = 0
taste_neurons.wfast[:] = 0 * b.mV
# Intrinsic homeostasis frozen
taste_neurons.homeo_on = 0.0 # disable homeostasis
# STDP OFF / clamp theta con unità coerenti (mV)
# Porta tutto in millivolt come float
low_mv   = float(theta_min / b.mV)
high_mv  = float(theta_max / b.mV)
# theta corrente -> float [mV]
th_mv = np.asarray(taste_neurons.theta[:] / b.mV, dtype=float)
# clamp numerico in mV
th_mv = np.clip(th_mv, low_mv, high_mv)
# offset morbido: median in mV
delta_mv = float(np.median(th_mv))
# bias ha unità di V: sottraggo delta come mV (corretto)
taste_neurons.theta_bias[:] -= delta_mv * b.mV
# clamp prudente del bias (sempre Quantity)
taste_neurons.theta_bias[:] = np.clip(taste_neurons.theta_bias[:], -3.0*b.mV, +3.0*b.mV)
# write-back di theta con unità (mV) + cappello di sicurezza vs theta_init+0.5 mV
new_theta = th_mv * b.mV
taste_neurons.theta[:] = np.minimum(new_theta, theta_init + 0.5*b.mV)
# to manage Hedonic window
taste_neurons.taste_drive[:] = 0.0
taste_neurons.av_over[:]  = 0.0
taste_neurons.av_under[:] = 0.0
S.x[:] = 0
S.xbar[:] = 0
S.y[:] = 0
S.ybar[:] = 0
S.elig[:] = 0
# NO NORMALIZATION during test
col_allow_upscale    = False
col_scale_max        = 1.10
col_norm_every       = 999999
inhibitory_S.g_step_inh = g_step_inh_local
inhibitory_S.delay = 0.5*b.ms
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
TRAINING_PHASE = False 
# no depression during TEST for STP
S.tau_rec[:]   = 400*b.ms
S.u0[:]        = 0.06

# STP warmup-start
# Disabilito ogni forma di plasticità prima del warm-up
S.stdp_on[:] = 0.0
taste_neurons.homeo_on = 0.0

# Flag per warm-start STP
if WARM_STP:
    # 1) Canale "attivo" dal PRIMO trial, senza consumarlo
    first_rates_vec, first_true_ids, _first_label = test_stimuli[0] # primo trial

    # ID target in modo generico (true_ids o argmax dei rates)
    if np.isscalar(first_true_ids):
        active_id = int(first_true_ids)
    else:
        # fallback robusto
        active_id = int(np.argmax(first_rates_vec[:unknown_id]))

    # 2) Costruzione di un vettore di stimolo UNA-TANTUM sul canale attivo
    warm_vec = np.zeros_like(first_rates_vec, dtype=float)
    # Ampiezza del primo trial per NON hard-codare frequenze
    warm_vec[active_id] = float(first_rates_vec[active_id])

    # 3) Applicazione della pre-stimolazione breve per il warm-up STP
    set_test_stimulus(warm_vec)         # imposta pg.rates coerentemente
    net.run(WARM_STP_MS)                # es. 80–120 ms
    set_test_stimulus(0.0 * warm_vec)   # azzera tutte le rate
    net.run(20*b.ms)                    # piccola pausa refrattaria di recovery

    # 4) Azzeramento contatori/accumulatori per NON contare il warm-up nel decoder
    prev_counts = spike_mon.count[:].copy()
# end STP warm-start

# reset GDI
gdi_pool.x[:] = 0.0  # svuota il pool GDI (altrimenti accumula)
S.gamma_gdi = 0.04
S_noise.gamma_gdi = 0.04
inhibitory_S.inh_scale = 0.28       # WTA ragionevole (non eccessiva)
pg_noise.rates = 0.2 * np.ones(num_tastes) * b.Hz  # rumore basso, non nullo
taste_neurons.theta_bias[:] = 0*b.mV  # niente bias attentivo ereditato dal train
#S.ex_scale = 1.0
#S.ex_scale_stp[:] = 1.0 
results = []
# low ACh in test phase
ach_test_level = 0.45
k_ex_ACH = 0.25   # leggermente meno del train, ma non troppo basso
mod.ACH[:] = ach_test_level
test_t0 = time.perf_counter()  # start stopwatch TEST

# pos e cap expectations
pos_expect_test = np.maximum(ema_pos_m1 * float(test_duration / training_duration), 1e-6)
cop_expect_test = np.maximum(ema_cop_m1 * float(test_duration / training_duration), 1.0)

min_spikes_for_known_base = 4 # soglia minima assoluta per gusti noti nella fase di test
# usa anche l'aspettativa positiva stimata:
min_spikes_for_known_test = max(2, int(np.floor(0.03 * np.max(pos_expect_test[:UNKNOWN_ID]))))
'''min_spikes_for_known_test = max(
    5,
    int(np.ceil(min_spikes_for_known * dur_scale * 1.2))  # p.es. 3 * 1.0 * 1.2 = 4 → ceil → 5
)'''
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
                                             float(min_spikes_for_known_test))

exact_hits = 0
total_test = len(test_stimuli)
n_known = unknown_id                     
H_unif = np.log(n_known) # uniform entropy
all_scores = []
all_targets = []
# hyperparameters
sep_min    = max(0.12, 0.25 / np.sqrt(n_known))  # minimum separation PMR
abs_margin_test = 0.0 # to avoid margin during test
# test loop
# -- inizio trial --
for step, (rates_vec, true_ids, label) in enumerate(test_stimuli, start=1):
    # GDI reset each trial
    if gdi_reset_each_trial:
        gdi_pool.x[:] = 0.0 # reset GDI pool at the beginning of test phase
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
        S.ex_scale = 1.10
        pg_noise.rates = test_baseline_hz * np.ones(num_tastes) * b.Hz
        taste_neurons.theta_bias[:] = 0 * b.mV

    # craving abilitation during test
    if ENABLE_DESIRE_TEST:
        # fattore per gusto (solo classi note)
        # stesso mix della tua formula per gli score (0.5 KF + 0.5 KS)
        boost = 1.0 + 0.5*CRAVE_KF*crave_f[:unknown_id] + 0.5*CRAVE_KS*crave_s[:unknown_id]
        # scrivi cr_gate per sinapsi con pre-sinaptico = gusto t (puoi limitarlo ai j della sua popolazione, ma non è obbligatorio)
        for ts in range(unknown_id):
            sl = taste_slice(ts)
            S.cr_gate[f"(i == {ts}) and (j >= {sl.start}) and (j < {sl.stop})"] = float(np.clip(boost[ts], 0.25, CRAVE_MAX))

    if len(true_ids) == 1 and true_ids[0] == unknown_id:
        # OOD/NULL � no normalization
        set_test_stimulus(rates_vec)
    else:
        if USE_GDI:
            set_test_stimulus(rates_vec)
        else:
            set_stimulus_vect_norm(rates_vec, total_rate=BASE_RATE_PER_CLASS * len(true_ids), include_unknown=False)

    # initializing the rewarding for GDI
    # divisive gain test-time proporzionale all'energia di input (proxy: somma rates noti)
    input_energy = float(np.sum(rates_vec[:unknown_id]))
    ref_rate = float(BASE_RATE_PER_CLASS)

    inp = np.asarray(rates_vec[:unknown_id], dtype=float)
    pmr_in = (inp.max() / (inp.sum() + 1e-9)) if inp.sum() > 0 else 0.0

    cap_base   = 0.45
    cap_boost  = float(np.interp(pmr_in, [0.25, 0.45], [0.95, 0.60]))     # più diffuso -> più cap
    cap_energy = float(np.interp(input_energy / (ref_rate + 1e-9), [0.5, 2.0], [cap_base, 0.90]))
    cap        = min(cap_boost, cap_energy)

    gamma_val  = float(np.clip(gamma_gdi_0, 0.20, cap))
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
    # base: soglia di rifiuto per-classe
    fp_gate_test = thr_per_class_test[:unknown_id].astype(float)  

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
    # TOP/SECOND robusti ai pari (niente argsort)
    top_idx = int(np.argmax(scores[:unknown_id]))
    top     = float(scores[top_idx])
    
    # trova l'indice del secondo senza bias e senza riordinare tutto
    if unknown_id >= 2:
        two_idx = np.argpartition(scores[:unknown_id], -2)[-2:]  # indici dei due massimi (non ordinati)
        # ordina quei due per valore e prendi il secondo
        two_idx = two_idx[np.argsort(scores[:unknown_id][two_idx])][::-1]
        second_idx = int(two_idx[1]) if int(two_idx[0]) == top_idx else int(two_idx[0])
        second = float(scores[:unknown_id][second_idx])
    else:
        second_idx = None
        second = 0.0

    sep = (top - second) / (top + 1e-9)  # relative separation

    # Entropia e metriche locali
    E_local = float(np.sum(scores[:unknown_id]))
    p_local = scores[:unknown_id].astype(float)
    if E_local > 0:
        p_local /= E_local
    else:
        p_local[:] = 0.0
    H   = float(-(p_local * np.log(p_local + 1e-12)).sum())
    PMR = top / (E_local + 1e-9)
    gap = sep
    z   = scores[:unknown_id] / np.maximum(pos_expect_test, 1.0)

    # EARLY REJECT: niente attività nota → UNKNOWN e chiudi il trial subito
    global_min_E = max(3.0, 0.6 * float(min_spikes_for_known_test))   # 3 o ~60% del minimo per classe
    if E_local < global_min_E:
        winners = [unknown_id]
        # debug minimale coerente coi tuoi log
        print(f"[DBG] early-reject: E={E_local:.1f}, top={top:.1f} < min_known={min_spikes_for_known_test}")
        # finalizzazione identica al resto del file
        expected  = [taste_map[idxs] for idxs in true_ids]
        predicted = [taste_map[w] for w in winners]
        hit = set(winners) == set(true_ids)
        print(f"\n{label}\n  expected:   {expected}\n  predicted: {predicted}\n  exact:   {hit}")
        results.append((label, expected, predicted, hit, ""))
        net.run(recovery_between_trials)
        continue

    # Piccolo stabilizzatore GABA
    if (E_local >= 3.0*min_spikes_for_known_test) and (gap < 0.14):
        mod.GABA[:] += 0.3 * gaba_pulse_stabilize

    #  Mix-like base window
    mix_pmr_lo, mix_pmr_hi = 0.26, 0.72
    mix_H_lo,   mix_H_hi   = 0.50, 1.85
    is_mixture_like = (mix_pmr_lo <= PMR <= mix_pmr_hi) and (mix_H_lo <= H <= mix_H_hi)
    did_extend = False

    # Estensione finestra ON-DEMAND solo se mix-like e co-gusti “borderline”
    if is_mixture_like:
        soft_abs = np.maximum(
            mix_abs_pos_frac * np.maximum(pos_expect_test, 1.0),
            float(min_spikes_for_known_test)
        )
        borderline = []
        # controlla solo i top-3 ma senza affidarti a argsort globale
        k = min(3, unknown_id)
        top_k_idx = np.argpartition(scores[:unknown_id], -k)[-k:]
        # ordina quei k per valore decrescente per il controllo
        top_k_idx = top_k_idx[np.argsort(scores[:unknown_id][top_k_idx])][::-1]

        for idxs in top_k_idx:
            if (scores[idxs] >= 0.8*soft_abs[idxs]) and (scores[idxs] < soft_abs[idxs]):
                borderline.append(int(idxs))

        if borderline:
            prev = spike_mon.count[:].copy()
            extra_b = 80 * b.ms  # +80 ms
            net.run(extra_b)
            diff_extra = (spike_mon.count[:] - prev).astype(float)
            scores = add_extra_scores(scores, diff_extra, unknown_id, reduce="sum")
            did_extend = True

            # ricalcola metriche con finestra estesa
            top_idx = int(np.argmax(scores[:unknown_id]))
            top     = float(scores[top_idx])
            if unknown_id >= 2:
                two_idx = np.argpartition(scores[:unknown_id], -2)[-2:]
                two_idx = two_idx[np.argsort(scores[:unknown_id][two_idx])][::-1]
                second_idx = int(two_idx[1]) if int(two_idx[0]) == top_idx else int(two_idx[0])
                second = float(scores[:unknown_id][second_idx])
            else:
                second_idx = None
                second = 0.0

        sep = (top - second) / (top + 1e-9)
        E   = float(np.sum(scores[:unknown_id]))
        p   = scores[:unknown_id].astype(float)
        if E > 0:
            p /= E
        else:
            p[:] = 0.0
        H   = float(-(p*np.log(p + 1e-12)).sum())
        PMR = top / (E + 1e-9)
        gap = sep
        E_local = E  # mantieni coerenza per il resto del flusso

    # Stima attivi e diffusione
    k_active = int(np.sum(scores[:unknown_id] >= min_spikes_for_known_test))
    n_strong = k_active
    is_mixture_like = is_mixture_like and (n_strong >= 2)

    gap_dyn = gap_thr
    if k_active in (2, 3):
        gap_dyn = max(0.10, 0.75 * gap_thr)

    flags = [
        PMR < PMR_thr,
        H   > H_thr,
        gap < gap_dyn,
    ]
    is_diffuse = sum(bool(x) for x in flags) >= 2

    # Politica unica inibizione per trial
    inh = float(inhibitory_S.inh_scale[0]) if hasattr(inhibitory_S, 'inh_scale') else 0.45  
    if k_active >= 3:
        inh *= min(1.40, 1.0 + 0.12*(k_active - 2))
    if is_diffuse:
        inh *= 1.12
    inhibitory_S.inh_scale[:] = float(np.clip(inh, 0.30, 1.40))

    # Gating UNKNOWN forte con soglie data-driven: spegni UNKNOWN se top è nettamente “buono”
    set_unknown_gate(
        pmr=PMR, gap=gap, H=H,
        PMR_thr=PMR_thr,      
        gap_thr=gap_thr,        
        H_thr=H_thr            
    )
    # z_min_guards verrà definito più sotto in base a mix/diffuse; qui solo soft gate:
    if z[top_idx] >= 0.90:
        S_unk.gain_unk = 0.0

    # Blend dinamico P_pos/P_cop
    lam_blend = float(np.clip(np.interp(PMR, [0.30, 0.55], [0.0, 1.0]) *
                          np.interp(H,   [0.7,  1.5 ], [0.0, 1.0]), 0.0, 1.0))
    P_blend = (1.0 - lam_blend) * P_pos + lam_blend * P_cop

    # Corsia soft per coppie (senza usare sorted_idx)
    if is_diffuse and is_mixture_like and k_active == 2 and second_idx is not None:
        # seconda classe candidata = second_idx
        top2 = int(second_idx)
        abs_soft = max(min_spikes_for_known_test,
                   0.30 * (pos_expect_test[top_idx] + pos_expect_test[top2]) / 2.0)
        if (scores[top_idx] >= abs_soft) and (scores[top2] >= 0.88*abs_soft):
            winners = sorted([top_idx, top2], key=lambda ids: scores[ids], reverse=True)
            is_diffuse = False
    else:
        winners = []

    # Soglie efficaci per classe
    mix_min_abs = (max(4, int(0.6*min_spikes_for_known_test))
               if is_mixture_like and (not is_diffuse) else
               min_spikes_for_known_test)

    clean_pmr = np.clip((PMR - PMR_thr) / max(1e-9, 0.85 - PMR_thr), 0.0, 1.0)
    clean_gap = np.clip((gap - gap_thr) / max(1e-9, 0.50 - gap_thr), 0.0, 1.0)
    clean     = 0.5*clean_pmr + 0.5*clean_gap

    mix_blend = np.interp(clean, [0.0, 1.0], [0.55, 0.95])
    frac_pos  = np.interp(clean, [0.0, 1.0], [0.68, 0.28])
    pos_exp_blend = (1.0 - mix_blend) * pos_expect_test + mix_blend * cop_expect_test

    if is_mixture_like and (not is_diffuse):
        frac_pos = max(0.38, frac_pos - 0.08)
        if (k_active in (2, 3)) and (mix_pmr_lo <= PMR <= mix_pmr_hi):
            frac_pos = max(0.34, frac_pos - 0.06)

    if not is_diffuse:
        thr_eff = np.minimum(thr_per_class_test[:unknown_id], frac_pos * pos_exp_blend)
        thr_eff = np.maximum(
            thr_eff,
            np.maximum(float(mix_min_abs), dyn_abs_min_frac * np.maximum(pos_expect_test, 1.0))
        )
    else:
        thr_eff = np.maximum(thr_per_class_test[:unknown_id], float(mix_min_abs))
    
    # Guardie z in funzione dello stato (base/mix/diffuse)
    z_min_guards = (z_min_mix if (is_mixture_like and not is_diffuse) else z_min_base) if (not is_diffuse) else 0.20

    top_pass_strict = (
        (scores[top_idx] >= 0.95 * thr_per_class_test[top_idx]) and
        (z[top_idx]     >= z_min_guards) and
        (gap            >= 0.95 * gap_thr)
    )   
    
    # ABSOLUTE-PASS HARD GUARD
    # z-guard aggiuntiva: per promuovere una classe singola, vogliamo anche z_top ≥ 1.0
    z_top_min = 1.0
    top_pass_strict = top_pass_strict and (z[top_idx] >= z_top_min)

    # Se il top NON passa il controllo assoluto, NON promuovere nulla.
    # Se NON è "mixture-like", chiudi subito il trial in UNKNOWN.
    if not top_pass_strict:
        if not is_mixture_like:
            winners = [unknown_id]
            expected  = [taste_map[idxs] for idxs in true_ids]
            predicted = [taste_map[wa] for wa in winners]
            hit = set(winners) == set(true_ids)
            print(f"\n{label}\n  expected:   {expected}\n  predicted: {predicted}\n  exact:   {hit}")
            results.append((label, expected, predicted, hit, ""))
            net.run(recovery_between_trials)
            continue
        else:
            # Caso mix-like: NON accettiamo ancora alcuna classe,
            # ma lasciamo la possibilità al blocco NNLS (se attivo) di proporre una scomposizione.
            # Se NNLS non trova candidati accettabili, ricadrà in UNKNOWN più avanti.
            pass

    # Penalità classi calde SOLO in energia diffusa
    if is_diffuse:
        neg_sd = np.array([ema_sd(ema_neg_m1[isd], ema_neg_m2[isd]) for isd in range(unknown_id)])
        heat = np.clip((overshoot / (neg_sd + 1e-6)), 0.0, 4.0)
        thr_eff[:unknown_id] *= (1.0 + 0.45 * heat)
        if top_idx == fatty_id and (z[fatty_id] < 1.45 or PMR < 0.66):
            winners = [unknown_id]

    # k_cap dinamico
    HHI   = float((p_local**2).sum()) if E_local > 0 else 1.0
    k_est = int(np.clip(round(1.0 / max(HHI, 1e-9)), 1, unknown_id))
    '''HHI = float((p**2).sum())
    k_est = int(np.clip(round(1.0 / max(HHI, 1e-9) - 0.3), 1, scores.size))'''  # -0.3 anti-sovrastima
    k_cap_base = int(np.clip(k_est, 2, n_noti))
    bonus = 0 if k_active < 2 else (2 if clean >= 0.70 else (1 if clean >= 0.60 else 0))
    k_cap = min(max(k_cap_base + bonus, 3), k_active, 9)

    print(f"\n[DBG] top_idx={top_idx} top={top:.1f} thr={thr_eff[top_idx]:.1f} "
        f"min_known={min_spikes_for_known_test} E={E_local:.1f} "
        f"PMR={PMR:.3f}/{PMR_thr:.3f} H={H:.3f}/{H_thr:.3f} gap={gap:.3f}/{gap_thr:.3f} "
        f"is_diffuse={is_diffuse} top_pass_strict={top_pass_strict}")
    print(f"\n[DBG] scores[:unknown_id]={scores[:unknown_id]}")

    # WINNERS SELECTION
    # A) vincitori “stretti”
    strict_winners = [
        idx for idx in range(unknown_id)
        if (scores[idx] >= (thr_eff[idx] + abs_margin_test)) and (z[idx] >= z_min_guards)
    ]
    winners = list(strict_winners)

    # B) mixture shortcut (non diffuso)
    top_known = (scores[top_idx] >= thr_eff[top_idx]) and (sep >= gap_thr)
    if top_known and (not is_diffuse) and is_mixture_like:
        # riusa second_idx se utile
        mixture_shortcut = []
        soft_abs = np.maximum(mix_abs_pos_frac * np.maximum(pos_expect_test, 1.0),
                          float(min_spikes_for_known_test))
        rel_guard = norm_rel_ratio_test * top
        for idx in range(unknown_id):
            if idx == top_idx:
                continue
            if (scores[idx] >= max(soft_abs[idx], rel_guard)) and (z[idx] >= max(0.18, z_rel_min)):
                mixture_shortcut.append(idx)

        if mixture_shortcut:
            add = [c for c in mixture_shortcut if c != top_idx and c not in winners]
            add.sort(key=lambda ia: scores[ia], reverse=True)
            add = add[:max(0, k_cap - 1)]
            if add and scores[add].sum() >= 0.20 * E_local:
                winners.extend(add)

    # C) Negative rel gate (come prima, invariato)
    if use_rel_gate_in_test and top_known and (not is_diffuse):
        rel_thr = rel_gate_ratio_test * top
        co_abs_soft = np.maximum(0.32 * thr_eff, 0.30 * cop_expect_test)
        co_abs_cap = 1.05 * rel_thr
        co_abs = np.minimum(co_abs_soft, co_abs_cap)

        E_abs = E_local
        co_abs_energy_min = np.clip(0.05 * E_abs, 2.0, 10.0)
        nrel = scores[:unknown_id] / (top + 1e-9)

        cand = [idx for idx in range(unknown_id)
            if (idx != top_idx) and (z[idx] >= z_rel_min) and
            ((nrel[idx] >= norm_rel_ratio_test and scores[idx] >= min_norm_abs_spikes) or
                (scores[idx] >= max(rel_thr, co_abs[idx]))) and
            (scores[idx] >= co_abs_energy_min)]
        
        cand.sort(key=lambda idx: scores[idx], reverse=True)

        add_max = max(0, k_cap - 1)
        # Filtro su winners già presenti e aggiunta di winners univoci
        cand = [c for c in cand if c not in winners]
        add = cand[:min(add_max, len(cand))]
        if add and scores[add].sum() >= 0.22 * E_local:
            winners.extend(add)
            
        # weak co-taste rescue (invariato)
        if top_pass_strict:
            rescue = []
            abs_co_min = 0.25 * min_spikes_for_known_test
            rel_min    = 0.05
            z_min_resc = 0.20
            for idx in range(unknown_id):
                if idx == top_idx or (idx in winners):
                    continue
                if (z[idx] >= z_min_resc and scores[idx] >= abs_co_min and
                (scores[idx] / (top + 1e-9) >= rel_min)):
                    rescue.append(idx)
            rescue.sort(key=lambda js: scores[js], reverse=True)
            if rescue and scores[rescue[0]] >= max(0.18 * E_local, 2.0):
                winners.append(rescue[0])

        if (k_active >= 3) and (clean < 0.35):
            winners = [top_idx] if top_pass_strict else [unknown_id]

    # D) NNLS solo se ancora vuoto/UNKNOWN e non diffuso e con energia sufficiente
    did_unsup_relabel = False
    unsup_labels = []
    unsup_log = None
    may_allow_k5 = (is_mixture_like and ((gap >= 1.00*gap_thr) or (PMR >= 1.10*PMR_thr)))
    abs_floor_vec = np.maximum(0.22*pos_expect_test, 0.18*cop_expect_test)
    #abs_floor_vec = np.maximum(ood_q, 0.0)  # per class floor from OOD quantiles
    
    # log monitor for NNLS
    def why_skipped():
        reasons = []
        if not ((winners_list == []) or (winners_list == [unknown_id])): reasons.append("already known winners")
        if is_diffuse: reasons.append("diffuse")
        if E_local < min_spikes_for_known_test: reasons.append("insufficient energy")
        if not is_mixture_like: reasons.append("non mixture-like")
        return ", ".join(reasons) or "condition for NNLS not satisfied"

    winners_list = list(winners) if not isinstance(winners, list) else winners
    should_try_base = (((winners_list == []) or (winners_list == [unknown_id])) and
                    (not is_diffuse) and
                    (E_local >= min_spikes_for_known_test))

    should_try_nnls = should_try_base
    # modifica guardia z minima per NNLS in base a mix/diffuse
    if should_try_nnls:
        if is_mixture_like and (not is_diffuse): # mix-like non diffuso → guardia z più alta
            cand_nnls, info_nnls = decode_by_nnls(
                scores=scores[:unknown_id],
                P_pos=P_pos, P_cop=P_cop,
                z_scores=z[:unknown_id],
                frac_thr=0.12,
                z_min_guard=max(0.05, z_min_mix), # z_min_mix ~ 0.05–0.12
                allow_k4=True,
                allow_k5=(True and may_allow_k5),
                gap=gap, gap_thr=gap_thr,
                pmr=PMR, pmr_thr=PMR_thr,
                abs_floor=ood_q,
                nnls_iters=250, nnls_lr=None, l1_cap=1.1
            )
        else:
            cand_nnls, info_nnls = decode_by_nnls(
                scores=scores[:unknown_id],
                P_pos=P_pos, P_cop=P_cop,
                z_scores=z[:unknown_id],
                frac_thr=0.07,
                z_min_guard=max(0.03, z_min_base), # z_min_base ~ 0.01–0.03
                allow_k4=True,
                allow_k5=(True and may_allow_k5),
                gap=gap, gap_thr=gap_thr,
                pmr=PMR, pmr_thr=PMR_thr,
                abs_floor=ood_q,
                nnls_iters=250, nnls_lr=None, l1_cap=1.0
        )  
        if cand_nnls:
            w_nnls = info_nnls.get("w", None)
            if w_nnls is not None:
                cand_nnls = sorted(cand_nnls, key=lambda isa: (scores[isa], w_nnls[isa]), reverse=True)
            else:
                cand_nnls = sorted(cand_nnls, key=lambda isa: scores[isa], reverse=True)
            k_cap_eff = min(k_cap, 5 if may_allow_k5 else 4)
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
            # stampa compatta “EFFECTIVE”
            print(f"\nNNLS: EFFECTIVE (k_abs={info_nnls['k_abs']}, "
                f"k_est={info_nnls['k_est']}, err={info_nnls['err']:.4f}, "
                f"cover={info_nnls['cover']:.3f}, thr_rel={info_nnls['thr_rel']:.3f}, "
                f"allow_k5={may_allow_k5})")
        else:
            # nessuna candidatura trovata
            print("\nNNLS: NO-EFFECT (no candidates)")
    else:
        # saltato perché le condizioni non sono soddisfatte
        print(f"\nNNLS: SKIPPED ({why_skipped()})")

    # E) Reject early per scene povere
    if (len(winners) == 0) and is_diffuse and \
    (scores[top_idx] < max(min_spikes_for_known_test, 0.5*thr_eff[top_idx])) and \
    (E_local < 0.35 * np.maximum(1.0, pos_expect_test.sum())):
        winners = [unknown_id]

    # F) Fallback finale (niente argsort, niente tie-bias)
    if len(winners) == 0:
        winners = [top_idx] if (scores[top_idx] >= min_spikes_for_known_test) else [unknown_id]

    # G) Cap finale su k (mantieni il top)
    if len(winners) > k_cap:
        rest = [isa for isa in winners if isa != top_idx]
        rest.sort(key=lambda ia: scores[ia], reverse=True)
        winners = [top_idx] + rest[:max(0, k_cap-1)]

    # H) Veto rapido su tanti attivi diffusi
    nnls_k = len(winners) if (did_unsup_relabel and winners and winners[0] != unknown_id) else 0
    too_many_active_diffuse = (
        (k_active > 5) and
        ((PMR < 1.03*PMR_thr) or (H > 0.95*H_thr)) and
        (not may_allow_k5) and
        (nnls_k < 5)
    )
    if too_many_active_diffuse:
        winners = [unknown_id]
        expected  = [taste_map[idxs] for idxs in true_ids]
        predicted = [taste_map[wx] for wx in winners]
        hit = set(winners) == set(true_ids)
        print(f"\n{label}\n  expected:   {expected}\n  predicted: {predicted}\n  exact:   {hit}")
        results.append((label, expected, predicted, hit, ""))
        net.run(recovery_between_trials)
        continue

    # SPICY during test
    # solo stato, niente plastica
    is_spicy_present = (spicy_id in true_ids) or (spicy_id in winners)
    if is_spicy_present and drv_now > thr_now:
        mod.HT[:] += 0.1
    
    # I) Veto finale rifiuto UNKNOWN
    if float(np.max(scores[:unknown_id])) < float(min_spikes_for_known_test):
        winners = [unknown_id]

    # debug prints
    print(f"[DBG] decision: PMR={PMR:.3f} H={H:.3f} gap={gap:.3f} top={taste_map[top_idx]} "
        f"score_top={scores[top_idx]:.1f} thr_eff_top={thr_eff[top_idx]:.1f} z_top={z[top_idx]:.2f}")
    print(f"[DBG] thr_eff[:]= {[round(float(x),1) for x in thr_eff]}")
    print(f"[DBG] z[:]= {[round(float(x),2) for x in z]}")
    print(f"\n[DBG] is_diffuse={is_diffuse} is_mixture_like={is_mixture_like} k_active={k_active} k_cap={k_cap}")
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
            (idx not in true_ids) and (float(scores[idx]) >= fp_gate_test[idx])
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

    # clamp finale: winners unici preservando l'ordine
    seen = set()
    winners = [wa for wa in winners if (wa not in seen and not seen.add(wa))]
    
    # to make a confrontation: expected vs predicted values
    expected  = [taste_map[idxs] for idxs in true_ids]
    predicted = [taste_map[wa] for wa in winners]
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
    
    # baseline per il prossimo trial successivo dopo aver eseguito trial attuale durante il test
    prev_counts = spike_mon.count[:].copy()
    
    # --- fine trial ---

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
'''plt.figure(figsize=(10,4))
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
for c in range(num_tastes-1):
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
    plt.show()

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
    plt.show()'''
