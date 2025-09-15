# SNN with STDP + eligibility trace to simulate an artificial tongue # that continuously learns to recognize multiple tastes (always-on). 
# that continuously learns to recognize multiple tastes (always-on).

import brian2 as b
b.prefs.codegen.target = 'numpy'
import matplotlib.pyplot as plt
import sys
import random
import numpy as np
import time
import shutil

# GDI toggle for activation/deactivation
USE_GDI = True # True => use GDI; False => use rate normalization

# global base rate per every class -> not 500 in total to split among all the classes but 500 for everyone
BASE_RATE_PER_CLASS = 500

# Individual profiles (thresholds/rewards) -> every new trial => new different individual
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
    return dict(thr0_hi=thr0_hi, thr0_lo=thr0_lo,
                k_hab_hi=k_hab_hi, k_sens_hi=k_sens_hi,
                k_hab_lo=k_hab_lo, k_sens_lo=k_sens_lo)

# state→bias mapping for every taste (coeff dimensionless)
c_hun_hi = np.array([+0.06, 0.00, +0.01, 0.00, +0.03, +0.04, 0.00])
c_hun_lo = np.array([-0.05, 0.00, 0.00, 0.00, -0.02, -0.03, 0.00])
c_sat_hi = np.array([-0.07, 0.00, 0.00, 0.00, -0.03, -0.05, 0.00])
c_sat_lo = np.array([+0.05, 0.00, 0.00, 0.00, +0.02, +0.03, 0.00])
c_h2o_hi = np.array([0.00, 0.00, -0.08, 0.00, 0.00, 0.00, -0.02])
c_h2o_lo = np.array([0.00, 0.00, +0.02, 0.00, 0.00, 0.00, 0.00])

def apply_internal_state_bias(profile, mod, tn):
    H = float(mod.HUN[0]); S = float(mod.SAT[0]); W = float(mod.H2O[0])
    thr0_hi = np.clip(profile['thr0_hi'] + c_hun_hi*H + c_sat_hi*S + c_h2o_hi*W, 0.05, 0.95)
    thr0_lo = np.clip(profile['thr0_lo'] + c_hun_lo*H + c_sat_lo*S + c_h2o_lo*W, 0.00, 0.70)
    # all neurons except for UNKNOWN
    tn.thr0_hi[:unknown_id] = thr0_hi
    tn.thr0_lo[:unknown_id] = thr0_lo

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

# EMA decoder helpers
def ema_update(m1, m2, x, lam):
    # scaling single updating
    m1_new = (1.0 - lam) * m1 + lam * x
    m2_new = (1.0 - lam) * m2 + lam * (x * x)
    return m1_new, m2_new

def ema_sd(m1, m2):
    var = max(1e-9, m2 - m1*m1)
    return np.sqrt(var)

# 1. Initialize the simulation
b.start_scope()
b.defaultclock.dt = 0.1 * b.ms  # high temporal precision
print("\n- ARTIFICIAL TONGUE's SNN with STDP and conductance-base LIF neurons, ELIGIBILITY TRACE, INTRINSIC HOMEOSTASIS and LATERAL INHIBITION: WTA (Winner-Take-All) -")

# 2. Define tastes
num_tastes = 8  # SWEET, BITTER, SALTY, SOUR, UMAMI, FATTY, SPICY, UNKNOWN

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
g_step_bg            = 0.3 * b.nS       # tiny background excitation noise
g_step_inh_local     = 1.2 * b.nS       # lateral inhibition strength

# Global Divisive Inhibition (GDI) parameters
tau_gdi              = 40 * b.ms        # global pool temporal window
k_e2g_ff             = 0.03             # feed-forward contribute (Poisson input spikes) -> GDI
k_e2g_fb             = 0.012            # feedback contribute (output neuron spikes) -> GDI
gamma_gdi_0          = 0.2              # scaled reward for (dimensionless)
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
p_aversion           = {6: 0.25}        # for example -> 25% of times SPICY is not good for the tongue
da_pulse_reward      = 1.0              # burst DA if classification is correct -> reward
ht_pulse_aversion    = 1.0              # burst 5-HT when imminent event is aversive
ht_pulse_fp          = 0.5              # burst 5-HT extra if there are a lot of strong FP
# Noradrenaline (NE) — arousal/attention
tau_NE               = 500 * b.ms       # fast decay
k_ex_NE              = 0.5              # reward scaling
k_inh_NE             = 0.2              # shrinks WTA for SNA
k_noise_NE           = 0.5              # low environment noise
ne_gain_r            = 0.3              # scaling reinforcement r entity
ne_pulse_amb         = 0.8              # burst on FP
# Histamine (HI) — novelty/exploration (slower than NE)
tau_HI               = 1500 * b.ms      # slow decay
k_ex_HI              = 0.30             # gain on synapses
k_inh_HI             = -0.15            # HI decrease WTA
k_theta_HI           = -1.0             # mV threshold bias per HI unit
k_noise_HI           = 0.30             # more HI ⇒ more environment noise
hi_gain_r            = 0.15             # reinforcement scaling with HI
hi_pulse_nov         = 0.8              # burst on ambiguity
hi_pulse_miss        = 1.0              # burst on error
# Acetylcholine (ACh) — contextual plasticity / different attention on train and test
tau_ACH              = 700 * b.ms       # decay for ACh
ach_train_level      = 0.8              # high ACh during training
ach_test_level       = 0.10             # low ACh during test
k_ex_ACH             = 0.25             # exictement gain with ACh
ach_plasticity_gain  = 0.40             # ACh ↑ reward effect on Δw
k_noise_ACH          = 0.25             # ACh noise environment reduction
# GABA — global inhibition/stability
tau_GABA             = 800 * b.ms       # decay for GABA
k_inh_GABA           = 0.60             # WTA scaling with GABA
gaba_pulse_stabilize = 0.8              # burst when activity is too much
gaba_active_neurons  = 4                # if > k neurons are activated in the same time → stability
gaba_total_spikes    = 120              # if total spikes per trial overcome this threshold → stability

# Intrinsic homeostasis adapative threshold parameters
target_rate          = 50 * b.Hz        # reference firing per neuron (tu 40-80 Hz rule)
tau_rate             = 200 * b.ms       # extimate window for rating -> spikes LPF
tau_theta            = 1 * b.second     # threshold adaptive speed
theta_init           = 0.0 *b.mV        # starting theta threshold for homeostasis
rho_target = target_rate * tau_rate     # dimensionless (Hz*s)

# Decoder threshold parameters
k_sigma              = 1.3              # ↑ if it is too weak
q_neg                = 0.99             # negative quantile

# Multi-label RL + EMA decoder
ema_lambda            = 0.05            # 0 < λ ≤ 1
tp_gate_ratio         = 0.30            # threshold to reward winner classes
fp_gate_warmup_steps  = 50              # delay punitions to loser classes if EMA didn't stabilize them yet
decoder_adapt_on_test = False           # updating decoder EMA in test phase
ema_factor            = 0.5             # EMA factor to punish more easy samples
use_rel_gate_in_test  = True            # using relative gates for mixtures and not only absolute gates
rel_gate_ratio_test   = 0.50            # second > 50 % rel_gate
mixture_thr_relax     = 0.35            # ≥ 35% of threshold per-class
z_rel_min             = 0.20            # z margin threshold to let enter taste in relative gate  
rel_cap_abs           = 10.0            # absolute value for spikes
dyn_abs_min_frac      = 0.30            # helper for weak co-tastes -> it needs at least 30% of positive expected
# boosting parameters to push more weak examples
norm_rel_ratio_test   = 0.15            # winners with z_i >= 15% normalized top
min_norm_abs_spikes   = 1               # at least one real spike
eps_ema               = 1e-3            # epsilon for EMA decoder

# Off-diag hyperparameters
beta                 = 0.03             # learning rate for negative reward
beta_offdiag         = 0.5 * beta       # off-diag parameter
use_offdiag_dopamine = True             # quick toggle to activate/deactivate reward for off-diagonals

# Normalization per-column (synaptic scaling in input)
use_col_norm         = True             # on the normalization
col_norm_mode        = "l1"             # "l1" (sum=target) or "softmax" -> synaptic scaling that mantains input scale per post neuron to avoid unfair competition
col_norm_every       = 3                # execute norm every N trial
col_norm_temp        = 1.0              # temperature softmax (if mode="softmax")
col_norm_target      = None             # if None, calculating the target at the beginning of the trial
diag_bias_gamma      = 1.30             # >1.0 = light bias to the diagonal weight before normalization
col_floor            = 0.0              # floor (0 or light epsilon) before norm
col_allow_upscale    = True             # light up-scaling
col_upscale_slack    = 0.90             # if L1 < 90% target → boost
col_scale_max        = 1.2              # max factor per step

# SPICY dynamic tolerance / aversion dynamics
spicy_id             = 6                # spicy taste is the sixth one
thr0_spice_var       = 0.40             # baseline aversive threshold -> driven unit
tau_thr_spice        = 30 * b.second    # adapting threshold -> slow
tau_sd_spice         = 50 * b.ms        # spicy intensity integration
tau_a_spice          = 200 * b.ms       # dynamic aversion
k_spike_spice        = 0.015            # spike contribution pre SPICY->drive
k_a_spice            = 1.0              # aversion reward
k_hab_spice          = 0.002            # upgrade threshold ↑ with adapting to aversion
eta_da_spice         = 2.0              # multiplier DA for adapting
k_sens_spice         = 0.001            # sensitization if above threshold but without reward -> just an adapting on the previous threshold: now higher
reinforce_dur        = 150 * b.ms       # short window to push DA gate on SPICY

# Hedonic window for all the tastes (SWEET, SOUR ecc...) -> one taste is rewarding ONLY if his spikes fire during this period
tau_drive_win        = 50 * b.ms        # intensity/taste integration
tau_av_win           = 200 * b.ms       # aversion/sub-threshold integration
tau_thr_win          = 30 * b.second    # thresholds adapting
eta_da_win           = 2.0              # rewarding on the habit of the Hedonic window
k_spike_drive        = 0.015            # driving kick on each input spike

# STDP and environment parameters
tau                  = 30 * b.ms        # STDP time constant
Te                   = 50 * b.ms        # eligibility trace decay time constant
A_plus               = 0.01             # dimensionless
A_minus              = -0.012           # dimensionless
alpha                = 0.1              # learning rate for positive reward
noise_mu             = 5                # noise mu constant
noise_sigma          = 0.8              # noise sigma constant
training_duration    = 1000 * b.ms      # stimulus duration
test_duration        = 500 * b.ms       # test verification duration
pause_duration       = 100 * b.ms       # pause for eligibility decay
n_repeats            = 10               # repetitions per taste
progress_bar_len     = 30               # characters
weight_monitors      = []               # list for weights to monitor
threshold_ratio      = 0.5              # threshold for winner spiking neurons
min_spikes_for_known = 10               # minimum number of spikes for neuron, otherwise UNKNOWN
top2_margin_ratio    = 1.4              # top/second >= 1.4 -> safe
weight_decay         = 1e-4             # weight decay for trial
verbose_rewards      = False            # dopamine reward logs
test_emotion_mode    = "active"         # to test with active neuromodulators

# Connectivity switch: "diagonal" | "dense"
connectivity_mode = "dense"  # "dense" -> fully-connected | "diagonal" -> one to one

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

# --- test generators OOD/NULL (expected = UNKNOWN) ---
def make_null(low=5, high=20):
    vix = np.random.randint(low, high+1, size=num_tastes).astype(float)
    vix[unknown_id] = 0.0
    return vix, [unknown_id], "NULL (only low background)"

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

# 4. LIF conductance-based OUTPUT taste neurons with intrinsic homeostatis and dynamic SPICY aversion
taste_neurons = b.NeuronGroup(
    num_tastes,
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
taste_neurons.theta_bias[:] = 0 * b.mV

# 5. Monitors
spike_mon = b.SpikeMonitor(taste_neurons) # monitoring spikes and time
state_mon = b.StateMonitor(taste_neurons, 'v', record=True) # monitoring membrane potential

# 6. Poisson INPUT neurons and STDP + eligibility trace synapses
# 1) Labelled stimulus (yes plasticity) -> stimulus Poisson (labelled)
pg = b.PoissonGroup(num_tastes, rates=np.zeros(num_tastes)*b.Hz)

# 2) Neutral noise (no plasticity) -> background Poisson (sorrounding ambient)
baseline_hz = 0.5  # 0.5–1 Hz
pg_noise = b.PoissonGroup(num_tastes, rates=baseline_hz*np.ones(num_tastes)*b.Hz)

# Global Division Inhibition GDI neuron integrator => only one 
gdi_pool = b.NeuronGroup(1, 'dx/dt = -x/tau_gdi : 1',
                         method='euler',
                         namespace={'tau_gdi': tau_gdi})
gdi_pool.x = 0.0

# linking GDI value to the output neurons
taste_neurons.gdi = b.linked_var(gdi_pool, 'x')

# GDI Synapses
# Feedforward: input Poisson -> to GDI
S_ff_gdi = b.Synapses(pg, gdi_pool, on_pre='x_post += k_e2g_ff', namespace={'k_e2g_ff': k_e2g_ff})
S_ff_gdi.connect('i != unknown_id')

# Feedback: output neurons -> to GDI
S_fb_gdi = b.Synapses(taste_neurons, gdi_pool, on_pre='x_post += k_e2g_fb', namespace={'k_e2g_fb': k_e2g_fb})
S_fb_gdi.connect('i != unknown_id')

# Monitoring GDI
gdi_mon = b.StateMonitor(gdi_pool, 'x', record=True)

# STDP Synapses with eligibility trace, Noradrenaline NE, lateral inhibition WTA, Global Division Inhibition GDI and EMA
S = b.Synapses(
    pg, taste_neurons,
    model='''
        w             : 1
        dApre/dt      = -Apre/tau   : 1 (event-driven)
        dApost/dt     = -Apost/tau  : 1 (event-driven)
        delig/dt      = -elig/Te    : 1 (clock-driven)
        ex_scale      : 1
        stdp_on       : 1
        gamma_gdi     : 1 (shared)      # global divisive rewarding
    ''',
    on_pre='''
        ge_post += (w * g_step_exc * ex_scale) / (1.0 + gamma_gdi * gdi_eff_post)
        Apre    += stdp_on * A_plus
        elig    += stdp_on * Apost
    ''',
    on_post='''
        Apost   += stdp_on * A_minus
        elig    += stdp_on * Apre
    ''',
    method='exact',
    namespace={
        'tau': tau, 'Te': Te,
        'A_plus': A_plus, 'A_minus': A_minus,
        'g_step_exc': g_step_exc
    }
)

# states initialization
taste_neurons.v[:] = EL
taste_neurons.s[:] = 0
taste_neurons.theta[:] = theta_init
taste_neurons.homeo_on = 1.0 # ON during training
taste_neurons.theta_bias[:] = 0 * b.mV

# dynamic SPICY states initialization
taste_neurons.is_spice[:]   = 0
taste_neurons.is_spice[spicy_id] = 1
taste_neurons.thr0_spice    = thr0_spice_var
taste_neurons.thr_spice[:]  = 0.0
taste_neurons.spice_drive[:] = 0.0
taste_neurons.a_spice[:]     = 0.0
taste_neurons.da_gate[:]     = 0.0

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

# initializing GDI
taste_neurons.gdi_center = 0.1
taste_neurons.gdi_half   = 0.50 

# Diagonal or dense connection mode except for UNKNOWN
if connectivity_mode == "diagonal":
    S.connect('i == j and i != unknown_id') # diagonal-connected
else:  # dense
    S.connect('i != unknown_id and j != unknown_id') # fully-connected without UNKNOWN

# init weights
if connectivity_mode == "dense":
    # initial advantage for true connections, minimal cross-talk
    S.w['i==j'] = '0.35 + 0.25*rand()'  # 0.30–0.50 value
    S.w['i!=j'] = '0.01 + 0.03*rand()'  # 0.02–0.06 value
else:
    S.w = '0.2 + 0.8*rand()'

# scaling factor for noradrenaline
S.ex_scale = 1.0
# Diagonal synapses index
ij_to_si = {}
Si = np.array(S.i[:], dtype=int)
Sj = np.array(S.j[:], dtype=int)
for k in range(len(Si)):
    ij_to_si[(int(Si[k]), int(Sj[k]))] = int(k) # synapse index

# Available diagonal index in 'diagonal' and 'dense'
diag_idx = {k: ij_to_si[(k, k)] for k in range(num_tastes-1) if (k, k) in ij_to_si} # dictionary to map the synapse couples i->j

# Background synapses (ambient excitation) with GDI
S_noise = b.Synapses(
    pg_noise, taste_neurons,
    model='gamma_gdi : 1 (shared)',
    on_pre='ge_post += (g_step_bg) / (1.0 + gamma_gdi * gdi_eff_post)',
    namespace=dict(g_step_bg=g_step_bg)
)
S_noise.connect('i == j and i != unknown_id')

# GDI Synapses initialization
S.gamma_gdi = gamma_gdi_0
S_noise.gamma_gdi = gamma_gdi_0

# Lateral inhibition for WTA (Winner-Take-All) with 5-HT modulation
inhibitory_S = b.Synapses(taste_neurons,
                    taste_neurons,
                    model='''
                        g_step_inh : siemens (shared)
                        inh_scale : 1
                    ''',
                    on_pre='gi_post += g_step_inh * inh_scale',
                    delay=0.2*b.ms,
                    #namespace=dict(g_step_inh=g_step_inh_local))
                    )
inhibitory_S.connect('i != j')
inhibitory_S.inh_scale = 0.6 # with GDI installed less WTA inhibition
inhibitory_S.g_step_inh = g_step_inh_local

# every spike from SPICY gate increases SPICY neuron drive
S_spice_sensor = b.Synapses(
    pg, taste_neurons,
    on_pre='spice_drive_post += k_spike_spice',
    namespace={'k_spike_spice': k_spike_spice}
)
S_spice_sensor.connect('i == spicy_id and j == spicy_id')

w_mon = b.StateMonitor(S, 'w', record=True)
weight_monitors.append((w_mon, S))

# Sensorial synapses for every taste neuron to module Hedonic window
S_drive = b.Synapses(pg, taste_neurons,
                     on_pre='taste_drive_post += k_spike_drive',
                     namespace={'k_spike_drive': k_spike_drive})
S_drive.connect('i == j and i != unknown_id')

# DA, 5-HT, NE, HI, ACh, GABA neuromodulators that decay over time
mod = b.NeuronGroup(
    1,
    model='''
        dDA/dt = -DA/tau_DA : 1
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
    namespace={'tau_DA': tau_DA, 'tau_HT': tau_HT, 'tau_NE': tau_NE, 
               'tau_HI' : tau_HI, 'tau_ACH' : tau_ACH, 'tau_GABA' : tau_GABA,
               'tau_HUN': 60*b.second, 'tau_SAT': 120*b.second, 'tau_H2O': 90*b.second}
)
mod.DA = 0.0
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
inh_mon = b.StateMonitor(inhibitory_S, 'inh_scale', record=True)
net.add(inh_mon)
theta_mon = b.StateMonitor(taste_neurons, 'theta', record=True)
s_mon = b.StateMonitor(taste_neurons, 's', record=True)
net.add(theta_mon, s_mon)
mod_mon = b.StateMonitor(mod, ['DA', 'HT', 'NE', 'HI', 'ACH', 'GABA'], record=True)
net.add(mod_mon)
# Hedonic window for SPICY nociceptive
spice_mon = b.StateMonitor(taste_neurons, ['spice_drive','thr_spice','a_spice','da_gate'],
                           record=[spicy_id])
net.add(spice_mon)
# Hedonic window monitor
hed_mon = b.StateMonitor(taste_neurons, ['taste_drive','thr_hi','thr_lo','av_over','av_under','da_gate'], record=True)
net.add(hed_mon)
# Monitoring all the GDI states
ge_mon = b.StateMonitor(taste_neurons, ['ge','gdi_eff'], record=[0])
net.add(ge_mon)
net.add(gdi_mon)

# 9. Prepare stimuli list
# 9A: pure‐taste training
pure_train = []
for taste_id in range(num_tastes-1):  # 0..6
    for _ in range(n_repeats):
        noise = np.clip(np.random.normal(noise_mu, noise_sigma, num_tastes), 0, None)
        noise[taste_id] = 100
        pure_train.append((noise, [taste_id],
                           f"TASTE: {taste_id} - '{taste_map[taste_id]}'"))

# 9B: mixture training
mixture_train = []
for _ in range(n_repeats): # N repeats for every training taste 
    mixture_train.append(noisy_mix([0, 3]))    # SWEET + SOUR
    mixture_train.append(noisy_mix([0, 2]))    # SWEET + SALTY
    mixture_train.append(noisy_mix([2, 4]))    # SALTY + UMAMI
    mixture_train.append(noisy_mix([0, 6]))    # SWEET + SPICY
    mixture_train.append(noisy_mix([1, 4, 6])) # BITTER + UMAMI + SPICY

# new random couples of tastes and mixtures to stress more the net
extra_mixes = [
    [1,4], [1,5], [3,4], [0,6],
    [2,3,4], [3,5,6], [1,3,4,6]
]
for _ in range(n_repeats):
    for mix in extra_mixes:
        mixture_train.append(noisy_mix(mix, amp=np.random.randint(180, 321)))

# 9C: test set -> couples and mixtures never seen during training
test_stimuli = [
    make_mix([0,4]),     # SWEET + UMAMI
    make_mix([1,2]),     # BITTER + SALTY
    make_mix([3,5]),     # SOUR + FATTY
    make_mix([0,2,4,6]), # 4-way
    make_mix([2,6]),     # SALTY + SPICY
]
# OOD + NULL
for _ in range(5):
    test_stimuli.append(make_null())
    test_stimuli.append(make_ood_diffuse())
    test_stimuli.append(make_ood_many(k=np.random.randint(3,6)))
# total stimuli
training_stimuli = pure_train + mixture_train
random.shuffle(training_stimuli) # continually randomize the stimuli without adapting patterns

# decoder parameters initialization
pos_counts = {idx: [] for idx in range(num_tastes-1)}
neg_counts = {idx: [] for idx in range(num_tastes-1)}
ema_neg_m1 = np.zeros(num_tastes-1)  # E[x] neg
ema_neg_m2 = np.zeros(num_tastes-1)  # E[x^2] neg
ema_pos_m1 = np.zeros(num_tastes-1)  # E[x] pos
ema_pos_m2 = np.zeros(num_tastes-1)  # E[x^2] pos

# SPICY initialization
taste_neurons.is_spice[:] = 0
taste_neurons.is_spice[spicy_id] = 1
taste_neurons.thr0_spice = thr0_spice_var
taste_neurons.thr_spice[:] = 0.0
taste_neurons.spice_drive[:] = 0.0
taste_neurons.a_spice[:] = 0.0
taste_neurons.da_gate[:] = 0.0

# Individual initialization before TRAINING loop
INDIV_ID = 42  # seed for representing different people
profile = sample_individual(seed=INDIV_ID)

# profile set in the group
taste_neurons.k_hab_hi[:unknown_id]  = profile['k_hab_hi']
taste_neurons.k_sens_hi[:unknown_id] = profile['k_sens_hi']
taste_neurons.k_hab_lo[:unknown_id]  = profile['k_hab_lo']
taste_neurons.k_sens_lo[:unknown_id] = profile['k_sens_lo']

apply_internal_state_bias(profile, mod, taste_neurons)
# in the beginning thr = thr0
taste_neurons.thr_hi[:unknown_id] = taste_neurons.thr0_hi[:unknown_id]
taste_neurons.thr_lo[:unknown_id] = taste_neurons.thr0_lo[:unknown_id]

# 10. Main "always-on" loop
print("\nStarting TRAINING phase...")
S.stdp_on[:] = 1.0
S.Apre[:]  = 0
S.Apost[:] = 0
S.elig[:]  = 0
# reset GDI
taste_neurons.v[:] = EL
sim_t0 = time.perf_counter()
step = 0
total_steps = len(training_stimuli) # pure + mixture
#  col_norm_target
if use_col_norm and connectivity_mode == "dense" and col_norm_target is None:
    # expected fan-in: all the pre-synaptics except for UNKNOWN
    fanin = (num_tastes - 1)
    init_mean = float(np.mean(S.w[:])) if len(S.w[:]) > 0 else 0.5
    col_norm_target = init_mean * fanin
    # target clamp
    col_norm_target = float(np.clip(col_norm_target, 0.5*fanin*0.2, 1.5*fanin*0.8))
    if verbose_rewards:
        print(f"col_norm_target auto={col_norm_target:.3f} (fanin={fanin}, init_mean={init_mean:.3f})")

# ablation conditions
# set_ach(False)   # ablation ACh
# set_gaba(False)  # ablation GABA
for input_rates, true_ids, label in training_stimuli:
    step += 1 
    # progress bar + chrono + ETA
    frac   = step / total_steps
    filled = int(frac * progress_bar_len)
    bar    = '█'*filled + '░'*(progress_bar_len - filled)

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
    apply_internal_state_bias(profile, mod, taste_neurons)
    taste_neurons.taste_drive[:] = 0.0
    taste_neurons.av_over[:]  = 0.0
    taste_neurons.av_under[:] = 0.0
    # GDI reset per-trial
    if gdi_reset_each_trial:
        gdi_pool.x[:] = 0.0

    # ACh must to be high during in training for efficient plasticity
    mod.ACH[:] = ach_train_level

    # after the increasing, neuromodulators must decay
    DA_now = float(mod.DA[0])
    HT_now = float(mod.HT[0])
    NE_now = float(mod.NE[0])
    HI_now = float(mod.HI[0])
    ACH_now = float(mod.ACH[0])
    GABA_now = float(mod.GABA[0])

    # reward gain and WTA addicted to NE/HI/ACh
    S.ex_scale = (1.0 + k_ex_NE * NE_now) * (1.0 + k_ex_HI * HI_now) * (1.0 + k_ex_ACH * ACH_now)
    # initializing the rewarding for GDI
    gamma_val = gamma_gdi_0 * (1.0 + 0.5*NE_now) * (1.0 - 0.3*HI_now)
    gamma_val = max(0.0, min(gamma_val, 0.5))  # clamp to max-limit gamma
    S.gamma_gdi = gamma_val
    S_noise.gamma_gdi = gamma_val
    # inhibition with 5-HT/GABA/HI -> whereas WTA more aggressive when 5-HT is higher, because aversion and fear must to influence the behiaviour during the train over and over
    _inh = (1.0 + k_inh_HT * HT_now + k_inh_NE * NE_now + k_inh_HI * HI_now + k_inh_GABA * GABA_now)
    inhibitory_S.inh_scale = max(0.3, _inh) # clamp to avoid errors

    # environment noise reduction with ACh, NE and HI, HT (clamp ≥0.05 Hz)
    ne_noise_scale = max(0.05, 1.0 - k_noise_NE * NE_now)
    hi_noise_scale = (1.0 + k_noise_HI * HI_now)
    ach_noise_scale = max(0.05, 1.0 - k_noise_ACH * ACH_now)
    pg_noise.rates = baseline_hz * ne_noise_scale * hi_noise_scale * ach_noise_scale * np.ones(num_tastes) * b.Hz

    # state gating guided by 5-HT because threshold has to be bigger if HT is higher -> behaviour of caution
    # 5-HT increase bias | HI decrease bias
    taste_neurons.theta_bias[:] = (k_theta_HT * HT_now + k_theta_HI * HI_now) * b.mV

    # 1) training stimulus with masking on no-target neurons
    masked = np.zeros_like(input_rates)
    masked[true_ids] = input_rates[true_ids]
    if USE_GDI:
        # no rates normalization with GDI
        set_stimulus_vector(masked, include_unknown=False)
    else:
        set_stimulus_vect_norm(masked, total_rate=BASE_RATE_PER_CLASS * len(true_ids), include_unknown=False)

    # GDI print debug
    print("\nGDI init:", float(gdi_pool.x[0]), "| gamma_gdi:", f"{gamma_val:.3f}")
    eff = float(taste_neurons.gdi_eff[0])
    div_eff = 1.0 + gamma_val * eff
    print(f"GDI eff-divisor≈{div_eff:.2f}  (gdi_eff={eff:.3f})")
    
    # 2) spikes counting during trial
    prev_counts = spike_mon.count[:].copy()
    net.run(training_duration)
    diff_counts = spike_mon.count[:] - prev_counts

    print(f"GDI end: x={float(gdi_pool.x[0]):.3f}, eff={float(taste_neurons.gdi_eff[0]):.3f}")

    # fear/aversion only if the generic taste stimulous overcomes the threshold
    drv = np.array(taste_neurons.taste_drive[:unknown_id])
    thr = np.array(taste_neurons.thr_hi[:unknown_id])
    if (drv > thr).any():
       mod.HT[:] += 0.2

    # fear/aversion only if the SPICY stimulous overcomes the threshold
    drv_now = float(taste_neurons.spice_drive[spicy_id])
    thr_now = float(taste_neurons.thr_spice[spicy_id])
    if drv_now > thr_now:
        mod.HT[:] += 0.3   # manage the aversion

    # to manage GABA during trial if there are too many spikes -> stabilizing the net
    total_spikes  = float(np.sum(diff_counts[:unknown_id]))
    active_neurs  = int(np.sum(diff_counts[:unknown_id] > 0))
    if (active_neurs > gaba_active_neurons) or (total_spikes > gaba_total_spikes):
        mod.GABA[:] += gaba_pulse_stabilize

    # collect all positive and negative counts
    for idx in range(num_tastes-1):
        if idx in true_ids:
            pos_counts[idx].append(int(diff_counts[idx]))
        else:
            neg_counts[idx].append(int(diff_counts[idx]))
    # updating EMA decoder parameters during online training
    for idx in range(num_tastes-1):
        if idx in true_ids:
            ema_pos_m1[idx], ema_pos_m2[idx] = ema_update(ema_pos_m1[idx], ema_pos_m2[idx],
                                                      float(diff_counts[idx]), ema_lambda)
        else:
            ema_neg_m1[idx], ema_neg_m2[idx] = ema_update(ema_neg_m1[idx], ema_neg_m2[idx],
                                                      float(diff_counts[idx]), ema_lambda)

    if diff_counts.max() <= 0:
        print("\nThere's no computed spike, skipping rewarding phase...")
        S.elig[:] = 0
        net.run(pause_duration)
        continue

    # A3: TP/FP threshold for each class
    scores = diff_counts.astype(float)
    scores[unknown_id] = -1e9
    mx = scores.max()
    rel = threshold_ratio * mx

    tp_gate = np.zeros(num_tastes-1, dtype=float)
    fp_gate = np.zeros(num_tastes-1, dtype=float)

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

       # safety clamp for infinite values
       if not np.isfinite(tp_gate_i): tp_gate_i = 0.0
       if not np.isfinite(fp_gate_i): fp_gate_i = 0.0

       tp_gate[idx] = tp_gate_i
       fp_gate[idx] = fp_gate_i

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
        inhibitory_S.inh_scale[:] = np.maximum(0.3, inhibitory_S.inh_scale[:] * 0.85)

    # total scores printing
    order = np.argsort(scores)
    dbg = [(taste_map[idx], int(scores[idx])) for idx in order[::-1]]

    # 4) 3-factors training reinforcement multi-label learning dopamine rewards for the winner neurons
    # A4: DIAGONAL: reward TP, punish big FP
    for idx in range(num_tastes-1):
       if idx not in diag_idx:
          continue
       si = diag_idx[idx]
       spikes_i = float(diff_counts[idx])

       r = 0.0
       if idx in true_ids:
          # big true positive
          if spikes_i >= tp_gate[idx]:
            # reward amplified by DA dopamine ACh acetylcholine and inhibited by 5-HT serotonine
            ht_eff = min(HT_now, 0.5)   # max 0.5 serotonine unit as penalty
            r = (alpha * (1.0 + da_gain * DA_now) * (1.0 + ach_plasticity_gain * ACH_now)) / (1.0 + ht_gain * ht_eff)
            r *= (1.0 + ne_gain_r * NE_now) * (1.0 + hi_gain_r * HI_now)
            conf = np.clip((top - second) / (top + 1e-9), 0.0, 1.0)
            r *= 0.5 + 0.5 * conf   # 0.5–1.0
       else:
         # big FP (after warm-up EMA)
          if step > fp_gate_warmup_steps and spikes_i >= fp_gate[idx]:
            # punition amplified by 5-HT -> aversive state verified
            r = - beta * (1.0 + ht_gain * HT_now)
            # same for FP
            r *= (1.0 + ne_gain_r * NE_now)

       if r != 0.0:
          delta = r * float(S.elig[si])
          if delta != 0.0:
            S.w[si] = float(np.clip(S.w[si] + delta, 0, 1))
          S.elig[si] = 0.0

    # A5: OFF-DIAGONAL: punish p→q when q is big FP
    if use_offdiag_dopamine:
        for p in true_ids:
            for q in range(num_tastes-1):
                if q == p:
                   continue
                # punish only if: hot EMA and q break the big FP threshold -> big FP case
                if step > fp_gate_warmup_steps and float(diff_counts[q]) >= fp_gate[q]:
                   si = ij_to_si.get((p, q), None)
                   if si is None:
                      continue  # if that synapse doesn't exist
                   old_w = float(S.w[si])
                   delta = - beta_offdiag * (1.0 + ht_gain * HT_now) * (1.0 + ne_gain_r * NE_now) * float(S.elig[si])
                   if delta != 0.0:
                      S.w[si] = float(np.clip(S.w[si] + delta, 0, 1))
                      if verbose_rewards and step % 10 == 0:
                          print(f"  offdiag - {taste_map[p]}→{taste_map[q]} | "
                              f"spk_q={float(diff_counts[q]):.1f} fp_q={fp_gate[q]:.1f} "
                              f"Δw={delta:+.4f}  w:{old_w:.3f}→{float(S.w[si]):.3f}")
                   S.elig[si] = 0.0
    
    # burst neuromodulators DA and 5-HT for the next trial as in a human-inspired biology brain
    # quality = Jaccard(T, P)
    T = set(true_ids); P = set(winners)
    jacc = len(T & P) / len(T | P) if (T | P) else 1.0

    if jacc >= 0.67:
        mod.DA[:] += da_pulse_reward * jacc # scaled reward
    elif jacc > 0.0:
        mod.DA[:] += 0.4 * da_pulse_reward * jacc # partial reward
        mod.HI[:] += 0.5 * hi_pulse_miss
    else:
        mod.HI[:] += hi_pulse_miss

    # if the prevision is good => reward to every taste in the trial
    if jacc >= 0.67:
        for tid in true_ids:
            if tid != unknown_id:
                taste_neurons.da_gate[tid] = 1.0
        net.run(reinforce_dur)
        for tid in true_ids:
            if tid != unknown_id:
                taste_neurons.da_gate[tid] = 0.0
    # clamp among thresholds for stability
    eps_thr = 0.02
    hi = np.array(taste_neurons.thr_hi[:]); lo = np.array(taste_neurons.thr_lo[:])
    hi = np.maximum(hi, lo + eps_thr)
    taste_neurons.thr_hi[:] = hi


    # with many strong FP, increase 5-HT -> future caution in the next trial
    has_strong_fp = any((i not in true_ids) and (float(diff_counts[i]) >= fp_gate[i])
                    for i in range(num_tastes-1))
    if has_strong_fp:
        mod.HT[:] += ht_pulse_fp
        mod.NE[:] += 0.5 * ne_pulse_amb # arousal on strong FP
        mod.HI[:] += 0.3 * hi_pulse_nov
        mod.GABA[:] += 0.3 * gaba_pulse_stabilize

    # safe clip on theta for homeostasis
    theta_min, theta_max = -12*b.mV, 12*b.mV
    taste_neurons.theta[:] = np.clip((taste_neurons.theta/b.mV), float(theta_min/b.mV), float(theta_max/b.mV)) * b.mV

    # Column normalization (incoming synaptic scaling)
    if use_col_norm and connectivity_mode == "dense" and (step % col_norm_every == 0):
        w_all = np.asarray(S.w[:], dtype=float)
        i_all = np.asarray(S.i[:], dtype=int)
        j_all = np.asarray(S.j[:], dtype=int)

        for jo in range(num_tastes - 1):  # no UNKNOWN
            idx = np.where(j_all == jo)[0]
            if idx.size == 0:
                continue

            col = w_all[idx]

            if col_floor > 0.0:
                col = np.maximum(col, col_floor)

            # light bias to the diagonal before normalization
            if diag_bias_gamma != 1.0:
                dloc = np.where(i_all[idx] == jo)[0]
                if dloc.size:
                    col[dloc[0]] *= float(diag_bias_gamma)

            L1 = float(np.sum(col))
            target = col_norm_target if col_norm_target is not None else L1

            if L1 > target and L1 > 1e-12:
                # downscale
                scale = target / L1
                col = np.clip(col * scale, 0.0, 1.0)
                w_all[idx] = col
            elif col_allow_upscale and (L1 < col_upscale_slack * target) and (L1 > 1e-12):
                # little up-scale to the target
                scale = min(col_scale_max, (target / L1))
                col = np.clip(col * scale, 0.0, 1.0)
                w_all[idx] = col

        S.w[:] = w_all

    # 5) eligibility trace decay among trials
    net.run(pause_duration)
    S.elig[:] = 0

# clean the bar
pbar_done()
print(f"\nEnded TRAINING phase! (elapsed: {fmt_mmss(time.perf_counter()-sim_t0)})")

# computing per-class thresholds
thr_per_class = np.zeros(num_tastes)
for idx in range(num_tastes-1):
    neg = np.asarray(neg_counts[idx], dtype=float)
    if neg.size == 0:
        neg = np.array([0.0])

    # neg_counts from batches
    mu_n  = float(np.mean(neg))
    sd_n  = float(np.std(neg))
    thr_gauss = mu_n + k_sigma * sd_n
    thr_quant = float(np.quantile(neg, q_neg)) if np.isfinite(neg).any() else 0.0

    # negative EMA during online training
    sd_ema = ema_sd(ema_neg_m1[idx], ema_neg_m2[idx])
    thr_ema = ema_neg_m1[idx] + k_sigma * sd_ema

    # hybrid: use max -> conservative against FP
    thr_i = max(float(min_spikes_for_known), thr_gauss, thr_quant, thr_ema)
    thr_per_class[idx] = thr_i

# safety clamp to cap negative threshold with positive example in TP case
for idx in range(num_tastes-1):
    pos_mu = float(ema_pos_m1[idx])
    pos_sd = float(ema_sd(ema_pos_m1[idx], ema_pos_m2[idx]))
    # safety if numbers are not finite
    if not np.isfinite(pos_mu): pos_mu = float(min_spikes_for_known)
    if not np.isfinite(pos_sd): pos_sd = 0.0
    thr_cap = max(float(min_spikes_for_known), pos_mu - 0.5 * pos_sd)
    thr_per_class[idx] = min(thr_per_class[idx], thr_cap)

print("Per-class thresholds (hybrid μ+kσ, quantile, EMA):",
      {taste_map[idx]: int(thr_per_class[idx]) for idx in range(num_tastes-1)})

# OOD/NULL calibration: increase threshold on OOD queues
def ood_calibration(n_null=10, n_ood=20, dur=200*b.ms, gap=0*b.ms):
    saved_stdp = float(S.stdp_on[0])
    S.stdp_on[:] = 0.0
    saved_noise = pg_noise.rates
    pg_noise.rates = 0 * b.Hz
    tmp = [[] for _ in range(num_tastes-1)]

    # NULL
    for _ in range(n_null):
        vix, _, _ = make_null()
        set_stimulus_vector(vix, include_unknown=False)
        prev = spike_mon.count[:].copy()
        net.run(dur)
        diff = spike_mon.count[:] - prev
        for idx in range(num_tastes-1):
            tmp[idx].append(int(diff[idx]))
        if gap > 0* b.ms: net.run(gap)

    # OOD
    for _ in range(n_ood):
        vix, _, _ = make_ood_diffuse()
        set_stimulus_vector(vix, include_unknown=False)
        prev = spike_mon.count[:].copy()
        net.run(dur)
        diff = spike_mon.count[:] - prev
        for idx in range(num_tastes-1):
            tmp[idx].append(int(diff[idx]))
        if gap > 0* b.ms: net.run(gap)

    # at least -> 99.5° percentile
    for idx in range(num_tastes-1):
        if tmp[idx]:
            thr_per_class[idx] = max(thr_per_class[idx], float(np.quantile(tmp[idx], 0.990)))

    pg_noise.rates = saved_noise
    S.stdp_on[:] = saved_stdp

# call that function
ood_calibration(n_null=8, n_ood=16, dur=200*b.ms, gap=0*b.ms)

# printing scaled weights after training
print(f"Target weights after training:")
for k in range(num_tastes-1):
    if k in diag_idx:
        si = diag_idx[k]
        print(f"  {taste_map[k]}→{taste_map[k]} = {float(S.w[si]):.3f}")
    else:
        print(f"  {taste_map[k]}→{taste_map[k]} = [no diag synapse]")

# weights copying before TEST
print("\n— Unsupervised TEST phase with STDP frozen —")
w_before_test = S.w[:].copy()
test_w_mon = b.StateMonitor(S, 'w', record=True)
net.add(test_w_mon)

# 11. Freezing STDP, homeostatis and input conductance
print("Freezing STDP for TEST phase…")
# to manage baseline_hz noise during test phase
use_test_noise = False
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
S.Apre[:] = 0
S.Apost[:] = 0
S.elig[:] = 0
# Intrinsic homeostasis frozen
taste_neurons.homeo_on = 0.0
th = taste_neurons.theta[:]
th = th - np.mean(th) # centered
theta_min, theta_max = -10*b.mV, 10*b.mV
taste_neurons.theta[:] = np.clip((th/b.mV), float(theta_min/b.mV), float(theta_max/b.mV)) * b.mV
# deactivate state effect for DA and 5-HT for clean test phase
taste_neurons.theta_bias[:] = 0 * b.mV
inhibitory_S.inh_scale = 0.8
mod.DA[:] = 0.0
mod.HT[:] = 0.0
# to manage Hedonic window
taste_neurons.taste_drive[:] = 0.0
taste_neurons.av_over[:]  = 0.0
taste_neurons.av_under[:] = 0.0
apply_internal_state_bias(profile, mod, taste_neurons)


# "emotive state" in test phase
if test_emotion_mode == "off":
    # neutral test
    mod.DA[:] = 0.0
    mod.HT[:] = 0.0
    mod.NE[:] = 0.0
    mod.HI[:] = 0.0
    taste_neurons.theta_bias[:] = 0 * b.mV
    inhibitory_S.inh_scale = 1.0
    S.ex_scale = 1.0  # gain reset
    pg_noise.rates = test_baseline_hz * np.ones(num_tastes) * b.Hz
else:
    # test with neuromodulators
    HT_now = float(mod.HT[0])
    NE_now = float(mod.NE[0])
    HI_now = float(mod.HI[0])
    # thresholds (HT ↑ threshold, HI ↓ threshold)
    taste_neurons.theta_bias[:] = (k_theta_HT * HT_now + k_theta_HI_test * HI_now) * b.mV
    # synaptic gain
    S.ex_scale = (1.0 + k_ex_NE * NE_now) * (1.0 + k_ex_HI_test * HI_now)
    # initializing the rewarding for GDI
    gamma_val = 0.1
    S.gamma_gdi = gamma_val
    S_noise.gamma_gdi = gamma_val
    # WTA (HI push the model to explore better ⇒ decrease WTA a bit)
    _inh = 1.0 + k_inh_HT * HT_now + k_inh_NE * NE_now + k_inh_HI_test * HI_now
    inhibitory_S.inh_scale = max(0.3, _inh)
    # noise (NE decrease, HI increase)
    ne_noise_scale = max(0.05, 1.0 - k_noise_NE * NE_now)
    pg_noise.rates = test_baseline_hz * ne_noise_scale * (1.0 + k_noise_HI_test * HI_now) * np.ones(num_tastes) * b.Hz

# to compute more mixtures
inhibitory_S.g_step_inh = 0.5 * g_step_inh_local
inhibitory_S.delay = 0.5*b.ms

# 12. TEST PHASE
print("\nStarting TEST phase...")
S.stdp_on[:] = 0.0
results = []
# low ACh in test phase
mod.ACH[:] = ach_test_level
pg_noise.rates = 0 * b.Hz
test_t0 = time.perf_counter()  # start stopwatch TEST
# Scale decoder thresholds to the test window
dur_scale = float(test_duration / training_duration)
thr_per_class[:unknown_id] *= dur_scale
min_spikes_for_known_test = max(3, int(min_spikes_for_known * dur_scale))
print(f"[Decoder] dur_scale={dur_scale:.2f} -> min_spikes_for_known_test={min_spikes_for_known_test}")
print("Scaled per-class thresholds:",
      {taste_map[idxs]: int(thr_per_class[idxs]) for idxs in range(num_tastes-1)})
recovery_between_trials = 100 * b.ms  # refractory recovery

exact_hits = 0
total_test = len(test_stimuli)

all_scores = []
all_targets = []
for step, (_rates_vec, true_ids, label) in enumerate(test_stimuli, start=1):
    # reset GDI
    if gdi_reset_each_trial:
        gdi_pool.x[:] = 0.0
    taste_neurons.ge[:] = 0 * b.nS
    taste_neurons.gi[:] = 0 * b.nS
    taste_neurons.wfast[:] = 0 * b.mV
    # initializing SPICY during test
    taste_neurons.spice_drive[spicy_id] = 0.0
    taste_neurons.a_spice[spicy_id]     = 0.0
    # progress bar + chrono + ETA
    frac   = step / total_test
    filled = int(frac * progress_bar_len)
    bar    = '█'*filled + '░'*(progress_bar_len - filled)

    elapsed = time.perf_counter() - test_t0
    eta = (elapsed/frac - elapsed) if frac > 0 else 0.0

    msg = (
      f"[{bar}] {int(frac*100)}% | Step {step}/{total_test} | Testing → {label}"
      f" | t={fmt_mmss(elapsed)} | ETA={fmt_mmss(eta)}"
    )
    pbar_update(msg)

    # deciding with "active" or "off" if there's need to applicate emotion test or not
    if test_emotion_mode != "off":
        DA_now = float(mod.DA[0])
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

    # 1) stimulus on target classes with UNKNOWN inputs
    if len(true_ids) == 1 and true_ids[0] == unknown_id:
        # OOD/NULL → no normalization
        set_stimulus_vector(_rates_vec, include_unknown=False)
    else:
        if USE_GDI:
            set_stimulus_vector(_rates_vec, include_unknown=False)
        else:
            set_stimulus_vect_norm(_rates_vec, total_rate=BASE_RATE_PER_CLASS * len(true_ids), include_unknown=False)

    # 2) spikes counting during trial
    prev_counts = spike_mon.count[:].copy()
    net.run(test_duration)
    diff_counts = spike_mon.count[:] - prev_counts

    # inject 5-HT for the generic TASTE aversion
    drv = np.array(taste_neurons.taste_drive[:unknown_id])
    thr = np.array(taste_neurons.thr_hi[:unknown_id])
    if (drv > thr).any():
        mod.HT[:] += 0.25

    # inject 5-HT for the SPICY aversion
    drv_now = float(taste_neurons.spice_drive[spicy_id])
    thr_now = float(taste_neurons.thr_spice[spicy_id])
    if drv_now > thr_now:
        mod.HT[:] += 0.25

    # 3) take the winners using per-class thresholds
    # strong decision with OOD and NULL
    scores = diff_counts.astype(float)
    scores[unknown_id] = -1e9
    mx = scores.max()

    all_scores.append(scores[:unknown_id].astype(float))
    tgt = np.zeros(num_tastes-1, dtype=int)
    for tid in true_ids:
        if tid != unknown_id:
            tgt[tid] = 1
    all_targets.append(tgt)

    sorted_idx = np.argsort(scores)[::-1]
    top_idx = int(sorted_idx[0])
    top = float(scores[top_idx])
    second = float(scores[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
    sep = (top - second) / (top + 1e-9)  # relative separation

    # scaled z-score during test
    pos_expect_test = np.maximum(ema_pos_m1 * float(test_duration / training_duration), 1e-6)
    z = scores[:unknown_id] / pos_expect_test

    # hyperparameters
    z_min = 0.25                        
    sep_min = 0.15 # just during the fallback
    abs_margin_test = 0.0 # to avoid margin during test
    #abs_margin_test = max(2.0, 5.0 * float(test_duration / training_duration))  # testing

    # multi-label candidates: threshold per-class + z-score
    strict_winners = [
        idx for idx in range(num_tastes-1)
        if (scores[idx] >= (thr_per_class[idx] + abs_margin_test)) and (z[idx] >= z_min)
    ]

    winners = list(strict_winners)

    # 2) relative add-on always available if there is a known top
    top_known = (scores[top_idx] >= (thr_per_class[top_idx] - 1)) or (z[top_idx] >= z_min)
    if use_rel_gate_in_test and top_known:
        rel_thr = rel_gate_ratio_test * top

        # absolute dynamic minimum per-class: half of positive expected (never < 1 spike)
        dyn_abs_min_i = np.maximum(min_norm_abs_spikes, dyn_abs_min_frac * pos_expect_test)

        add = [idx for idx in range(num_tastes-1)
            if (idx not in winners)
                and (scores[idx] >= rel_thr)
                and (z[idx] >= z_rel_min)
                and (scores[idx] >= mixture_thr_relax * thr_per_class[idx])
                and (scores[idx] >= dyn_abs_min_i[idx])]

        if add:
            add.sort(key=lambda idx: scores[idx], reverse=True)
            winners.extend(add)   # adding class to the final winner list

    # 3) fallback if there aren't winners
    if not winners:
        if (scores[top_idx] >= thr_per_class[top_idx] + abs_margin_test) and (z[top_idx] >= z_min) and (sep >= sep_min):
            winners = [top_idx]

    # 4) UNKNOWN labeling classification if there's no enough energy
    if (top < min_spikes_for_known_test) or (len(winners) == 0):
        winners = [unknown_id]

    order = np.argsort(scores)
    dbg = [(taste_map[idxs], int(scores[idxs])) for idxs in order[::-1]]
    print("\nTest scores:", dbg)

    # burst NE on ambiguities
    ambiguous = (second > 0 and top / (second + 1e-9) < 1.3) or (len(winners) > 2)
    if test_emotion_mode == "active" and ambiguous:
        mod.NE[:] += ne_pulse_amb
        mod.HI[:] += 0.5 * hi_pulse_nov
        inhibitory_S.inh_scale[:] = np.maximum(0.3, inhibitory_S.inh_scale[:] * 0.85) # if a class is weak, is not going to be suppressed in general cases except for ambiguous cases

    # Emotional burst in test phase
    if test_emotion_mode == "active":
        if set(winners) == set(true_ids):
            mod.DA[:] += da_pulse_reward  # gratification
        has_strong_fp = any(
            (i not in true_ids) and (float(diff_counts[i]) >= thr_per_class[i])
            for i in range(num_tastes-1)
        )
        if has_strong_fp:
            mod.HT[:] += ht_pulse_fp # prudence or fear increased

        if set(winners) != set(true_ids):  # miss
            mod.HI[:] += 0.5 * hi_pulse_miss
    
    if test_emotion_mode != "off":
        msg += f" | DA={float(mod.DA[0]):.2f} HT={float(mod.HT[0]):.2f} NE={float(mod.NE[0]):.2f} HI={float(mod.HI[0]):.2f} ACH={float(mod.ACH[0]):.2f} GABA={float(mod.GABA[0]):.2f}"
    # showing the log bar    
    pbar_update(msg)

    # to make a confrontation: expected vs predicted values
    expected  = [taste_map[idxs] for idxs in true_ids]
    predicted = [taste_map[w] for w in winners]
    hit = set(winners) == set(true_ids)

    # output visualization
    print(f"\n{label}\n  expected:   {expected}\n  predicted: {predicted}\n  exact:   {hit}")
    results.append((label, expected, predicted, hit))
    # 4) final valutation
    if set(winners) == set(true_ids):
        exact_hits += 1
    # 5) final refractory period after trial
    net.run(recovery_between_trials)

# unfreezing intrinsic homeostasis
taste_neurons.homeo_on = 1.0

# clean the bar
pbar_done()
print(f"\nTEST phase done (elapsed: {fmt_mmss(time.perf_counter()-test_t0)})")

# Metrics classification report with Jaccard class and confusion matrix
# a. Test accuracy
ok = 0
for label, exp, pred, hit in results:
    status = "OK" if hit else "MISS"
    print(f"{label:26s} | expected={exp} | predicted={pred} | {status}")
    ok += int(hit)
print(f"\nTest accuracy (exact-set match): {ok}/{len(results)} = {ok/len(results):.2%}")

# b. Jaccard class, recall, precision, f1-score
label_to_id = {lbl: idx for idx, lbl in taste_map.items()}
classes = [idx for idx in range(num_tastes) if idx != unknown_id]
# Class counters
stats = {idx: {'tp': 0, 'fp': 0, 'fn': 0} for idx in classes}
jaccard_per_case = []
for _, exp_labels, pred_labels, _ in results:
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
    return "—" if not np.isfinite(x) else f"{x*100:5.1f}%"

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

print("\n— Micro/Macro —")
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
    # class support pos/neg and AP baseline ≈ pos/N
    print("\n- Supports and AP baseline for every class -")
    for c in range(Y_scores.shape[1]):
        n_pos = int(Y_scores[:, c].sum())
        n_neg = int(Y_scores.shape[0] - n_pos)
        base_ap = (n_pos / (n_pos + n_neg)) if (n_pos + n_neg) > 0 else float('nan')
        print(f"{taste_map[c]:>6s}: support +={n_pos}, -={n_neg}, baseline AP≈{base_ap:.3f}")

    def roc_points(y, s):
        # y ∈ {0,1}, s = continual scores
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
        auc = float(np.trapz(TPR, FPR))
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
        return float(np.trapz(prec, rec))

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

    print("\n— AUC scores —")
    print("Per-class ROC-AUC:", [f"{x:.3f}" if np.isfinite(x) else "—" for x in roc_auc_per_class])
    print("Per-class PR-AUC: ", [f"{x:.3f}" if np.isfinite(x) else "—" for x in pr_auc_per_class])
    print("Per-class AP:     ", [f"{x:.3f}" if np.isfinite(x) else "—" for x in ap_per_class])
    print(f"Macro ROC-AUC={macro_roc_auc:.3f} | Macro PR-AUC={macro_pr_auc:.3f} | Macro mAP={macro_mAP:.3f}")
    print(f"Micro ROC-AUC={micro_roc_auc:.3f} | Micro PR-AUC={micro_pr_auc:.3f} | Micro mAP={micro_mAP:.3f}")
else:
    print("\n[INFO] Skipping AUC metrics: no stored per-trial score arrays.")

# Rejection UNKNOWN metrics
unknown_trials = sum(1 for _, exp, _, _ in results if exp == ['UNKNOWN'])
unknown_ok = sum(1 for _, exp, pred, _ in results if exp == ['UNKNOWN'] and ('UNKNOWN' in pred))
if unknown_trials > 0:
    print(f"\nRejection accuracy (UNKNOWN on OOD/NULL): {unknown_ok}/{unknown_trials} = {unknown_ok/unknown_trials:.2%}")
else:
    print("\n[WARN] No UNKNOWN/OOD trials were included in the test set.")

# to monitor strict rejection
unknown_strict_ok = sum(
    1 for _, exp, pred, _ in results
    if exp == ['UNKNOWN'] and pred == ['UNKNOWN']
)
if unknown_trials > 0:
    print(f"Rejection accuracy (STRICT): {unknown_strict_ok}/{unknown_trials} = {unknown_strict_ok/unknown_trials:.2%}")

# weight changes during test confrontation -> they don't change because STDP frozen during test phase
print("\nWeight changes during unsupervised test:")
for k in range(num_tastes-1):
    if k in diag_idx:
        si = diag_idx[k]
        print(f"  {taste_map[k]}→{taste_map[k]}: Δw = {float(S.w[si] - w_before_test[si]):+.4f}")

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

# b) Weight trajectories for diagonal synapses (i→i)
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
    print("[WARNING] No diagonal synapse (i→i) found among monitored synapses. Legend skipped.")

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
for k in ['DA','HT','NE','HI','ACH','GABA']:
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
    plt.axvline(thr_per_class[c], ls='--')
    plt.title(f'{taste_map[c]}  | thr={int(thr_per_class[c])}')
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
plt.title('Weights matrix (note→note)')
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
'''for idx in range(num_tastes-1):
    plt.figure(figsize=(10,3))
    plt.plot(hed_mon.t/b.ms, hed_mon.taste_drive[idx], label='drive')
    plt.plot(hed_mon.t/b.ms, hed_mon.thr_hi[idx], label='thr_hi')
    plt.plot(hed_mon.t/b.ms, hed_mon.thr_lo[idx], label='thr_lo')
    plt.legend(loc='upper right')
    plt.title(f'Hedonic window for {taste_map[idx]}')
    plt.xlabel('ms')
    plt.tight_layout()
    plt.show()'''
