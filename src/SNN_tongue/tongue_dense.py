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

# global base rate per every class -> not 500 in total to split among all the classes but 500 for everyone
BASE_RATE_PER_CLASS = 500

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

# Rates vector helper with normalization
def set_stimulus_vect_norm(rate_vec, total_rate=None):
    r = np.asarray(rate_vec, dtype=float).copy()
    r[unknown_id] = 0.0
    if total_rate is not None and r.sum() > 0:
        r *= float(total_rate) / r.sum()
    pg.rates = r * b.Hz

# Rates vector helper without normalization
def set_stimulus_vector(rate_vec):
    r = np.asarray(rate_vec, dtype=float).copy()
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
    1: "So acid!",
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
C                    = 200 * b.pF
gL                   =  10 * b.nS
EL                   = -70 * b.mV
Vth                  = -52 * b.mV
Vreset               = -60 * b.mV

# Synaptic reversal & time constants
Ee                   = 0*b.mV
Ei                   = -80*b.mV
taue                 = 10*b.ms
taui                 = 10*b.ms

# Size of conductance kick per spike (scaling)
g_step_exc           = 3.5 * b.nS       # excitation from inputs
g_step_bg            = 0.3 * b.nS       # tiny background excitation noise
g_step_inh_local     = 1.2 * b.nS       # lateral inhibition strength

# Neuromodulators (DA Dopamine: reward, 5-HT Serotonine: aversion/fear, NE Noradrenaline: arousal/attention)
tau_DA               = 300 * b.ms       # fast decay: short gratification
tau_HT               = 2 * b.second     # slow decay: prudence, residual fear
da_gain              = 1.0              # how much DA expand the positive reinforcement
ht_gain              = 1.0              # how much 5-HT expand the punishment or how much it stops LTP
# aversive state on the entire circuit
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
theta_init           = 0.0 *b.mV        # starting theta
rho_target = target_rate * tau_rate     # dimensionless (Hz*s)

# Decoder threshold parameters
k_sigma              = 1.6              # ↑ if it is too weak
q_neg                = 0.99             # negative quantile

# Multi-label RL + EMA decoder
ema_lambda            = 0.05            # 0 < λ ≤ 1
tp_gate_ratio         = 0.30            # threshold to reward winner classes
fp_gate_warmup_steps  = 50              # delay punitions to loser classes if EMA didn't stabilize them yet
decoder_adapt_on_test = False           # updating decoder EMA in test phase
ema_factor            = 0.5             # EMA factor to punish more easy samples

# Off-diag hyperparameters
beta                 = 0.03             # learning rate for negative reward
beta_offdiag         = 0.5 * beta       # off-diag parameter
use_offdiag_dopamine = True             # quick toggle

# Normalization per-column (synaptic scaling in input)
use_col_norm         = True             # on the normalization
col_norm_mode        = "l1"             # "l1" (sum=target) or "softmax"
col_norm_every       = 3                # execute norm every N trial
col_norm_temp        = 1.0              # temperature softmax (if mode="softmax")
col_norm_target      = None             # if None, calculating the target at the beginning of the trial
diag_bias_gamma      = 1.30             # >1.0 = light bias to the diagonal weight before normalization
col_floor            = 0.0              # floor (0 or light epsilon) before norm
col_allow_upscale    = True             # light up-scaling
col_upscale_slack    = 0.90             # if L1 < 90% target → boost
col_scale_max        = 1.2              # max factor per step

# STDP parameters
tau                  = 30 * b.ms        # STDP time constant
Te                   = 50 * b.ms        # eligibility trace decay time constant
A_plus               = 0.01             # dimensionless
A_minus              = -0.012           # dimensionless
alpha                = 0.1              # learning rate for positive reward
noise_mu             = 5                # noise mu constant
noise_sigma          = 0.8              # noise sigma constant
inhib_amp            = 0.1              # lateral inhibition constant
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
    v = np.zeros(num_tastes)
    for idx in ids: v[idx] = amp
    label = " + ".join([f"'{taste_map[idx]}'" for idx in ids])
    return v, ids, f"TASTE: {label}"

# helper to index training tastes in the stimuli list
def noisy_mix(ids, amp=250, mu=noise_mu, sigma=noise_sigma):
    v = np.clip(np.random.normal(mu, sigma, num_tastes), 0, None)
    for idx in ids:
        v[idx] = amp
    label = " + ".join([f"'{taste_map[idx]}'" for idx in ids])
    return v, ids, f"TASTE: {label} (train)"

# 4. LIF conductance-based OUTPUT taste neurons with intrinsic homeostatis
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
    ''',
    threshold='v > (Vth + theta + theta_bias + wfast)',
    reset='v = Vreset; s += 1; wfast += 0.3*mV',
    refractory=2*b.ms,
    method='euler',
    namespace={
        'C': C, 'gL': gL, 'EL': EL,
        'Ee': Ee, 'Ei': Ei,
        'taue': taue, 'taui': taui,
        'Vth': Vth, 'Vreset': Vreset,
        'tau_rate': tau_rate, 'tau_theta': tau_theta,
        'rho_target': rho_target
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

# STDP Synapses with eligibility trace, Noradrenaline NE, lateral inhibition WTA and EMA
S = b.Synapses(
    pg, taste_neurons,
    model='''
        w             : 1
        dApre/dt      = -Apre/tau   : 1 (event-driven)
        dApost/dt     = -Apost/tau  : 1 (event-driven)
        delig/dt      = -elig/Te    : 1 (clock-driven)
        ex_scale      : 1
    ''',
    on_pre='''
        ge_post += w * g_step_exc * ex_scale
        Apre    += A_plus
        elig    += Apost
    ''',
    on_post='''
        Apost   += A_minus
        elig    += Apre
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
taste_neurons.homeo_on = 1.0 # ON in training

# Diagonal or dense connection mode except for UNKNOWN
if connectivity_mode == "diagonal":
    S.connect('i == j and i != unknown_id') # diagonal-connected
else:  # dense
    S.connect('i != unknown_id') # fully-connected

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
    ij_to_si[(int(Si[k]), int(Sj[k]))] = int(k)

# Available diagonal index in 'diagonal' and 'dense'
diag_idx = {k: ij_to_si[(k, k)] for k in range(num_tastes-1) if (k, k) in ij_to_si}

# Background synapses (ambient excitation)
S_noise = b.Synapses(pg_noise, taste_neurons, on_pre='ge_post += g_step_bg',
                 namespace=dict(g_step_bg=g_step_bg))
S_noise.connect('i == j and i != unknown_id')

# Lateral inhibition for WTA (Winner-Take-All) with 5-HT modulation
inhibitory_S = b.Synapses(taste_neurons,
                    taste_neurons,
                    model='inh_scale : 1',
                    on_pre='gi_post += g_step_inh * inh_scale',
                    delay=0.2*b.ms,
                    namespace=dict(g_step_inh=g_step_inh_local))
inhibitory_S.connect('i != j')
inhibitory_S.inh_scale = 0.9

w_mon = b.StateMonitor(S, 'w', record=True)
weight_monitors.append((w_mon, S))

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
    ''',
    method='exact',
    namespace={'tau_DA': tau_DA, 'tau_HT': tau_HT, 'tau_NE': tau_NE, 
               'tau_HI' : tau_HI, 'tau_ACH' : tau_ACH, 'tau_GABA' : tau_GABA}
)
mod.DA = 0.0
mod.HT = 0.0
mod.NE = 0.0
mod.HI = 0.0
mod.ACH  = 0.0
mod.GABA = 0.0

# 8. Building the SNN network and adding levels
net = b.Network(
    taste_neurons,
    pg,
    pg_noise, # imput neurons noise introduced
    S,
    S_noise,
    inhibitory_S,
    spike_mon,
    state_mon,
    mod # neuromodulator neurons installed into the net
)
# monitoring all neuromodulators
net.add(w_mon)
theta_mon = b.StateMonitor(taste_neurons, 'theta', record=True)
s_mon = b.StateMonitor(taste_neurons, 's', record=True)
net.add(theta_mon, s_mon)
mod_mon = b.StateMonitor(mod, ['DA', 'HT', 'NE', 'HI', 'ACH', 'GABA'], record=True)
net.add(mod_mon)

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
    [0,4], [1,2], [3,5], [2,6],
    [0,1,4], [2,3,6], [0,2,4,6]
]
for _ in range(n_repeats):
    for mix in extra_mixes:
        mixture_train.append(noisy_mix(mix, amp=np.random.randint(180, 321)))

# 9C: test set -> couples and mixtures never seen during training
test_stimuli = [
    make_mix([0,4]),      # SWEET + UMAMI
    make_mix([1,2]),      # BITTER + SALTY
    make_mix([3,5]),      # SOUR + FATTY
    make_mix([0,2,4,6]),  # 4-way
    make_mix([2,6]),      # SALTY + SPICY
]
# total stimuli
training_stimuli = pure_train + mixture_train
random.shuffle(training_stimuli) # continually randomize the stimuli without adapting patterns

# decoder parameters initialization
pos_counts = {idx: [] for idx in range(num_tastes-1)}
neg_counts = {idx: [] for idx in range(num_tastes-1)}
ema_neg_m1 = np.zeros(num_tastes-1)  # E[x]
ema_neg_m2 = np.zeros(num_tastes-1)  # E[x^2]
ema_pos_m1 = np.zeros(num_tastes-1)
ema_pos_m2 = np.zeros(num_tastes-1)

# 10. Main "always-on" loop
print("\nStarting TRAINING phase...")
S.Apre[:]  = 0
S.Apost[:] = 0
S.elig[:]  = 0
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

    # 5-HT anticipatory serotonine -> aversive episode happened?
    aversive_now = any((cls in p_aversion) and (np.random.rand() < p_aversion[cls]) for cls in true_ids)
    # if there was a possible aversive episode
    if aversive_now:
        mod.HT[:] += ht_pulse_aversion # increase caution before imminent training

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

    # 1) training stimulus with masking on no target neurons
    masked = np.zeros_like(input_rates)
    masked[true_ids] = input_rates[true_ids]
    set_stimulus_vect_norm(masked, total_rate=BASE_RATE_PER_CLASS * len(true_ids))

    # 2) spikes counting during trial
    prev_counts = spike_mon.count[:].copy()
    net.run(training_duration)
    diff_counts = spike_mon.count[:] - prev_counts
    # to manage GABA during trial if there are too many spikes
    total_spikes  = float(np.sum(diff_counts[:unknown_id]))
    active_neurs  = int(np.sum(diff_counts[:unknown_id] > 0))
    if (active_neurs > gaba_active_neurons) or (total_spikes > gaba_total_spikes):
        mod.GABA[:] += gaba_pulse_stabilize
    # periodic ablation debug
    if step % 5 == 0:
        print(f"\n[dbg] step={step} spikes_tot={total_spikes:.0f} active={active_neurs} "
          f"WTA={float(np.mean(inhibitory_S.inh_scale[:])):.2f} "
          f"DA={DA_now:.2f} HT={HT_now:.2f} NE={NE_now:.2f} HI={HI_now:.2f} "
          f"ACH={ACH_now:.2f} GABA={GABA_now:.2f}")
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

       # safety clamp
       if not np.isfinite(tp_gate_i): tp_gate_i = 0.0
       if not np.isfinite(fp_gate_i): fp_gate_i = 0.0

       tp_gate[idx] = tp_gate_i
       fp_gate[idx] = fp_gate_i

    # selecting all the spiking winning neurons >= threshold_ratio
    sorted_idx = np.argsort(scores)[::-1]
    top, second = scores[sorted_idx[0]], (scores[sorted_idx[1]] if len(sorted_idx) > 1 else 0.0)
    second = scores[sorted_idx[1]] if len(sorted_idx) > 1 else 0.0
    margin_ok = (second <= 0) or (top / (second + 1e-9) >= top2_margin_ratio)
    winners = []
    if top >= min_spikes_for_known and second > 0 and (top / (second + 1e-9) >= top2_margin_ratio):
        winners = [int(sorted_idx[0])] # only one dominant taste
    else:
        thr = threshold_ratio * mx
        winners = [idx for idx,c in enumerate(scores) if c >= thr]
        if not winners:
            winners = [int(np.argmax(scores))] # tastes > 1

    # burst NE
    ambiguous = (second > 0 and top/(second + 1e-9) < 1.3) or (len(winners) > 2)
    if ambiguous:
        mod.NE[:] += ne_pulse_amb
        mod.HI[:] += hi_pulse_nov
        inhibitory_S.inh_scale[:] = np.maximum(0.3, inhibitory_S.inh_scale[:] * 0.85)

    # total scores printing
    order = np.argsort(scores)
    dbg = [(taste_map[idx], int(scores[idx])) for idx in order[::-1]]

    # matches evaluation
    if step % 5 == 0:
        T = set(true_ids); P = set(winners)
        jacc = len(T & P) / len(T | P) if (T | P) else 1.0
        print(f"[match] T={sorted(list(T))} P={sorted(list(P))} J={jacc:.2f} "
          f"top={top:.0f} second={second:.0f} thr_tp={int(np.mean(tp_gate))} thr_fp={int(np.mean(fp_gate))}")

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
            ht_eff = min(HT_now, 0.5)   # massimo 0.5 unità di serotonina come penalità
            r = (alpha * (1.0 + da_gain * DA_now) * (1.0 + ach_plasticity_gain * ACH_now)) / (1.0 + ht_gain * ht_eff)
            r *= (1.0 + ne_gain_r * NE_now) * (1.0 + hi_gain_r * HI_now)
            conf = np.clip((top - second) / (top + 1e-9), 0.0, 1.0)  # già calcoli top/second
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
                # punish only if: hot EMA and q break the big FP threshold
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
    # global reward if the prediction is correct
    '''if set(winners) == set(true_ids):
        mod.DA[:] += da_pulse_reward
    else:
        mod.HI[:] += hi_pulse_miss'''

    # qualità = Jaccard(T, P)
    T = set(true_ids); P = set(winners)
    jacc = len(T & P) / len(T | P) if (T | P) else 1.0

    if jacc >= 0.67:
        mod.DA[:] += da_pulse_reward * jacc         # gratifica scalata
    elif jacc > 0.0:
        mod.DA[:] += 0.4 * da_pulse_reward * jacc   # piccola gratifica parziale
        mod.HI[:] += 0.5 * hi_pulse_miss
    else:
        mod.HI[:] += hi_pulse_miss

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

    # light weight decay for all the weights to avoid constant saturation to w=1 -> TO REMOVE if homeostasis (in this problem HOMEOSTASIS is biologically better)
    '''if weight_decay > 0:
        S.w[:] = np.clip(S.w[:] * (1 - weight_decay), 0, 1)'''

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

# DEBUG: printing pos and neg values for each class
for idx in [0,2,3]: # SWEET, SALTY, SOUR
    print(taste_map[idx],
          "pos μ,σ=", np.mean(pos_counts[idx]), np.std(pos_counts[idx]),
          "neg μ,σ=", np.mean(neg_counts[idx]), np.std(neg_counts[idx]))

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
# Neuromodulator parameters in test
k_inh_HI_test   = -0.08
k_inh_HT_test = 0.4
k_theta_HI_test = -0.5
k_ex_HI_test    = 0.15
k_noise_HI_test = 0.15
ht_pulse_aversion = 0.5
S.pre.code  = 'ge_post += w * g_step_exc * ex_scale'
S.post.code = ''
taste_neurons.v[:] = EL
taste_neurons.ge[:] = 0 * b.nS
taste_neurons.gi[:] = 0 * b.nS
taste_neurons.s[:]  = 0
taste_neurons.wfast[:] = 0 * b.mV
pg_noise.rates = 0 * b.Hz # noise silenced test
S.Apre[:] = 0
S.Apost[:] = 0
S.elig[:] = 0
# Intrinsic homeostasis frozen
taste_neurons.homeo_on = 0.0
th = taste_neurons.theta[:]
th = th - np.mean(th) # centered
theta_min, theta_max = -10*b.mV, 10*b.mV
taste_neurons.theta[:] = np.clip(th, theta_min, theta_max)
# deactivate state effect for DA and 5-HT for clean test phase
taste_neurons.theta_bias[:] = 0 * b.mV
inhibitory_S.inh_scale = 1.0
mod.DA[:] = 0.0
mod.HT[:] = 0.0

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
    pg_noise.rates = baseline_hz * np.ones(num_tastes) * b.Hz
else:
    # test with neuromodulators
    HT_now = float(mod.HT[0])
    NE_now = float(mod.NE[0])
    HI_now = float(mod.HI[0])
    # thresholds (HT ↑ threshold, HI ↓ threshold)
    taste_neurons.theta_bias[:] = (k_theta_HT * HT_now + k_theta_HI_test * HI_now) * b.mV
    # synaptic gain
    S.ex_scale = (1.0 + k_ex_NE * NE_now) * (1.0 + k_ex_HI_test * HI_now)
    # WTA (HI push the model to explore better ⇒ decrease WTA a bit)
    _inh = 1.0 + k_inh_HT * HT_now + k_inh_NE * NE_now + k_inh_HI_test * HI_now
    inhibitory_S.inh_scale = max(0.3, _inh)
    # noise (NE decrease, HI increase)
    ne_noise_scale = max(0.05, 1.0 - k_noise_NE * NE_now)
    pg_noise.rates = baseline_hz * ne_noise_scale * (1.0 + k_noise_HI_test * HI_now) * np.ones(num_tastes) * b.Hz

# to compute more mixtures
inhibitory_S.namespace['g_step_inh'] = 0.5 * g_step_inh_local
inhibitory_S.delay = 0.5*b.ms
use_rel_gate_in_test = False # in multi-label is better to deactivate
rel_gate_ratio_test  = 0.10
rel_cap_abs = 10.0 # absolute value for spikes
# boosting parameters to push more weak examples
norm_rel_ratio_test = 0.15 # winners with z_i >= 15% normalized top
min_norm_abs_spikes = 1 # at least one real spike
eps_ema = 1e-3 

# 12. TEST PHASE
print("\nStarting TEST phase...")
results = []
# low ACh in test phase
mod.ACH[:] = ach_test_level
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

# function to inject UNKNOWN inside test set and confuse the net
def add_unknown(rate_vec, unk_min=20, unk_max=60):
    v = rate_vec.copy()
    v[unknown_id] = np.random.randint(unk_min, unk_max+1)
    return v

for step, (_rates_vec, true_ids, label) in enumerate(test_stimuli, start=1):
    taste_neurons.ge[:] = 0 * b.nS
    taste_neurons.gi[:] = 0 * b.nS
    taste_neurons.wfast[:] = 0 * b.mV
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
        pg_noise.rates = baseline_hz * ne_noise_scale * (1.0 + k_noise_HI_test * HI_now) * ach_noise_scale * np.ones(num_tastes) * b.Hz
        if test_emotion_mode == "active": # aversive anticipation: if SPICY, increase 5-HT 
            aversive_now = any( (cls in p_aversion) and (np.random.rand() < p_aversion[cls]) for cls in true_ids ) 
            if aversive_now: 
                mod.HT[:] += ht_pulse_aversion

    # 0) inject UNKNOWN taste during test phase
    _rates_vec = add_unknown(_rates_vec, 20, 60)
    # 1) stimulus on target classes
    set_stimulus_vect_norm(_rates_vec, total_rate=BASE_RATE_PER_CLASS * len(true_ids))

    # 2) spikes counting during trial
    prev_counts = spike_mon.count[:].copy()
    net.run(test_duration)
    diff_counts = spike_mon.count[:] - prev_counts
    # GABA stabilization as in training phase
    total_spikes  = float(np.sum(diff_counts[:unknown_id]))
    active_neurs  = int(np.sum(diff_counts[:unknown_id] > 0))
    if (active_neurs > gaba_active_neurons) or (total_spikes > gaba_total_spikes):
        mod.GABA[:] += gaba_pulse_stabilize
    # maintaining EMA during test phase
    if decoder_adapt_on_test:
        for idxs in range(num_tastes-1):
            if idxs in true_ids:
                ema_pos_m1[idxs], ema_pos_m2[idxs] = ema_update(ema_pos_m1[idxs], ema_pos_m2[idxs],
                                                          float(diff_counts[idxs]), ema_lambda)
            else:
                ema_neg_m1[idxs], ema_neg_m2[idxs] = ema_update(ema_neg_m1[idxs], ema_neg_m2[idxs],
                                                          float(diff_counts[idxs]), ema_lambda)
                # increment threshold if FP grow up
                sd_ema = ema_sd(ema_neg_m1[idxs], ema_neg_m2[idxs])
                thr_ema = ema_neg_m1[idxs] + k_sigma * sd_ema
                thr_per_class[idxs] = max(thr_per_class[idxs], thr_ema)

    # 3) take the winners using per-class thresholds
    scores = diff_counts.astype(float)
    scores[unknown_id] = -1e9
    mx = scores.max()

    sorted_idx = np.argsort(scores)[::-1]
    top = scores[sorted_idx[0]]
    second = scores[sorted_idx[1]] if len(sorted_idx) > 1 else 0.0

    if mx < min_spikes_for_known_test:
        winners = [unknown_id]
    else:
        # absolute gate for each class
        base_winners = [i for i in range(num_tastes-1) if scores[i] >= thr_per_class[i]]
        rel = min(rel_gate_ratio_test * mx, rel_cap_abs)
        rel_winners = [i for i in range(num_tastes-1) if scores[i] >= rel] if use_rel_gate_in_test else []
        pos_expect = np.maximum(ema_pos_m1, eps_ema)
        z = scores[:unknown_id] / pos_expect
        z_max = float(np.max(z)) if z.size else 0.0

        norm_winners = []
        for i in range(num_tastes-1):
            if (z[i] >= norm_rel_ratio_test * z_max) and (scores[i] >= min_norm_abs_spikes):
                # mini-soglia: almeno il 25% della soglia assoluta della classe
                mini = 0.25 * thr_per_class[i]
                if scores[i] >= mini:
                    norm_winners.append(i)

        winners = list(sorted(set(base_winners) | set(rel_winners) | set(norm_winners)))
        if not winners and mx >= min_spikes_for_known_test:
            winners = [int(np.argmax(scores))]

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

# weight changes during test confrontation -> they don't change because STDP frozen during test phase
print("\nWeight changes during unsupervised test:")
for k in range(num_tastes-1):
    if k in diag_idx:
        si = diag_idx[k]
        print(f"  {taste_map[k]}→{taste_map[k]}: Δw = {float(S.w[si] - w_before_test[si]):+.4f}")

print("\nEnded TEST phase successfully!")

# 13. Plots
# a) Spikes over time
plt.figure(figsize=(10,4))
plt.plot(spike_mon.t/b.ms, spike_mon.i, '.k')
plt.xlabel("Time (ms)")
plt.ylabel("Neuron index")
plt.title("All spikes (input neurons silent, only taste_neurons)")
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
