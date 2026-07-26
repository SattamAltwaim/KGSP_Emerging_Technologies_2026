# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo","numpy","matplotlib","qiskit","qiskit-aer","pylatexenc","scipy","qiskit-ibm-runtime"]
# ///
"""Day 5 lab · Marimo notebook · Molecules, money, and a real machine

Tapered-tier structure per PEDAGOGY.md § Difficulty polarity:
    Baseline (everyone with TA support): Section 1 — VQE on H₂
    Stretch (most students):             Section 2 — the QAOA portfolio
    Aspirational (top ~30%):             Section 3 — touch a real quantum computer

Each tier runs the `explore → predict → build` micro-structure (decision #23).
SINGLE NOTEBOOK (decision #27 / docs/lab-authoring-playbook.md): solutions in
`mo.accordion` folds under each build cell; Explore widgets run on hidden
reference engines so exploration works BEFORE anything is built; problem-
specific pieces are black boxes (`make_utility_layer` → one opaque utility(γ)
gate, echoing Day 4's `make_cost_layer`); guards on every student-code cell.

Stakes numbers in the prose were MEASURED (playbook §1), not guessed:
- H₂ Hamiltonians are the published 2-qubit BK-reduced coefficients from
  O'Malley et al., PRX 6, 031007 (2016), Table I (STO-6G; identity term
  includes nuclear repulsion — verified: eigvalsh gives −1.1456 Ha at
  R = 0.75 Å ≈ the FCI/STO-6G total energy). Exact ground: −1.1456 Ha
  (R = 0.75) / −1.0067 Ha (R = 1.50); Hartree–Fock (the |10⟩ ledge):
  −1.1246 / −0.9190 → correlation energy 21.0 mHa / 87.7 mHa. The 1-knob
  ansatz bowl bottoms at θ* ≈ −0.23 / −0.73. Shot-based VQE (COBYLA,
  8 192 shots, 40 iters; readout at 32 768) lands within a few mHa
  (measured mean |err| 2.3 / 1.6 mHa, max ~5 mHa over 8 runs each).
- Portfolio (6 assets, pick 3, λ = 1): optimum = SOLR+H2OX+MEDS, U* = 0.140
  (return 27 %, risk 0.13); the greed trio SOLR+CHIP+SHIP keeps only 0.010.
  QAOA p = 3 measured: P(optimal draw) ≈ 5–6 % vs blind 1.56 %,
  P(budget-feasible) ≈ 92–98 % vs blind 31 %; best sampled portfolio hit
  the true optimum in every measured run. λ = 0 winner: SOLR+CHIP+SHIP
  (0.35); λ = 2 winner: SHIP+H2OX+MEDS (0.02).
- GHZ on FakeBrisbane (noise model of ibm_brisbane): Hellinger fidelity
  ≈ 0.93 vs ideal, ~7 % of shots land on ideally-impossible bitstrings.
- Past the wall (Tier-3 finale), measured on the MPS referee
  (AerSimulator matrix_product_state; ~0.3 s per 56-qubit circuit):
  · Run A, the 56-carbon π chain (Hückel model, JW → −(t/2)Σ(XX+YY),
    open chain, t = 1): Givens-rotation ansatz p=2 trained at N=8 by
    exact-expectation COBYLA (2 fixed starts) → E = −4.594 vs exact
    −4.7588 (96.5 %); Hückel N/2-orbital referee ≡ 2⁸ qubit ED to
    ~2e−15 (checked at N=6, 8). SAME 4 params at N=56 (2 bases ×
    2,048 shots): E ≈ −33.2 ± 0.3 vs Hückel exact −35.292 vs |0101…⟩
    baseline 0 → ~94 % of the π delocalization energy, never re-tuned.
    110 Hamiltonian terms, 2 measurement settings. XXPlusYYGate needs
    beta=π/2 (plain RXX+RYY layers give hopping whose ⟨XX+YY⟩ vanishes
    identically — measured, not folklore).
  · Run B, Max-Cut on the chip's own wiring: 56-qubit BFS-induced
    subgraph of ibm_brisbane's heavy-hex coupling map → |E| = 59
    (degrees 1/2/3), connected, triangle-free, bipartite ⇒ true max
    cut = 59. Tuned on a 14-qubit patch by the closed-form p=1
    triangle-free formula (Wang et al. 2018): (γ*, β*) ≈ (0.814,
    0.390); formula ≡ exact statevector on the patch to ~2e−14. At
    56: formula predicts ⟨cut⟩ = 42.98/59 (ratio 0.728); MPS sampling
    42.9 ± 0.1 (best sampled cut 53–54/59), ~0.6 s. Sign landmine:
    qiskit's RZZ convention needs rzz(−γ); with +γ the result mirrors
    around |E|/2 (patch: 3.26 vs 9.74 around 6.5 — measured). SWAP
    tax on FakeBrisbane (ol=1, chip layout): 118 native 2q gates
    chip-aligned vs 1,087 with shuffled labels, depth 143 vs 1,046.
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt

    return AerSimulator, QuantumCircuit, minimize, mo, np, plt, transpile


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Day 5 lab · Molecules, money, and a real machine

    This morning was the landscape: the hardware zoo, the vendors renting
    it out by the second, the hype filters, the four clocks. This
    afternoon the week's machinery meets that world three times — and the
    last time, it isn't a simulator on the other end.

    **How to use this notebook**:
    - Cells run top-to-bottom. Try each in order.
    - Each section follows the same rhythm: **explore** (move a slider,
      watch the pictures), **predict** (commit an answer *before* you
      run), **build** (write the code from the recipe).
    - Stuck? Every build has a **💡 Solution** fold right under it —
      closed by default. Predict first, peek last.
    - No take-home. Anything you don't finish now, we don't chase later.

    **Three sections**:
    1. **The chemistry twin: VQE on H₂** — baseline. The algorithm we
       promised yesterday: same variational loop, and the score is now a
       molecule's *energy*.
    2. **The QAOA portfolio** — stretch. Yesterday's machine in a suit:
       six assets, room for three, risk on the meter.
    3. **Touch a real quantum computer** — aspirational. First one
       small circuit, over the cloud, to an actual dilution fridge (or
       its noise twin) — then the finale: **two 56-qubit runs on the
       far side of the wall where every statevector simulator on Earth
       dies**.
    """)
    return


@app.cell
def _(AerSimulator):
    # The week's trusty ideal simulator. It stays the referee all afternoon —
    # Section 3 is where a REAL backend (or its noise model) enters the story.
    backend = AerSimulator()
    return (backend,)


@app.cell(hide_code=True)
def _(transpile):
    def run_counts(qc, backend_, shots=2048):
        """Run a measured circuit; return counts keyed by BIT-ORDER
        bitstrings (character i = qubit i, read left to right). Qiskit's raw
        keys are little-endian — this reverses them once so no other cell
        has to think about it. (Same helper as Day 4.)"""
        tc = transpile(qc, backend_)
        raw = backend_.run(tc, shots=shots).result().get_counts()
        return {k[::-1]: v for k, v in raw.items()}

    return (run_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 1 · The chemistry twin: VQE on H₂ (baseline)

    Yesterday you built the variational machine. Today, its twin
    — **VQE**, the *variational quantum eigensolver*, the algorithm half
    of quantum-chemistry industry is betting on. Same loop as yesterday:
    a circuit with a knob, a measured score, a classical optimizer
    turning the knob. Only the score changed: it's now the **energy of a
    molecule**, and lower is better.

    The molecule is H₂ — two protons, two electrons, the simplest
    molecule there is. Why anyone pays for its energy: the ground-state
    energy is the number that decides bond lengths, reaction rates,
    whether a drug candidate sticks to its target. Chemistry's fast
    classical workhorse — **Hartree–Fock** — answers with the single
    best "both electrons in the lowest orbital" configuration. What it
    drops is the **correlation energy**: the sliver of energy the
    electrons save by *sharing a superposition* of configurations —
    exactly the thing a classical shortcut can't hold and a qubit can.

    **The stakes, measured**: on our two H₂ instances that sliver is
    **21.0 mHa** at the natural bond length (0.75 Å) and **87.7 mHa**
    with the bond stretched to 1.50 Å (1 Ha = 27.2 eV, so that's 0.57 eV
    and 2.39 eV — chemistry cares about 0.04 eV, "chemical accuracy").
    Your 3-gate circuit wins that sliver back.

    (Honesty up front: this H₂ lives in a 4×4 matrix — we will
    brute-force diagonalize it *in this notebook* as the referee. The
    bet the industry makes is on molecules needing ~50+ qubits, where
    the same matrix has 2⁵⁰ rows and diagonalization is dead. Today you
    run the method small enough to be graded against certain truth —
    the same deal as yesterday's 1,024 brute-forced splits.)
    """)
    return


@app.cell
def _():
    # ── The published Hamiltonians ───────────────────────────────────────────
    # 2-qubit reduced H₂ Hamiltonian, hardcoded from O'Malley et al.,
    # PRX 6, 031007 (2016), Table I (STO-6G basis, Bravyi–Kitaev mapping,
    # symmetry-reduced to 2 qubits):
    #
    #   H = g0·I + g1·Z0 + g2·Z1 + g3·Z0Z1 + g4·X0X1 + g5·Y0Y1
    #
    # g0 includes the proton–proton repulsion, so eigenvalues are TOTAL
    # energies in hartree (Ha). No chemistry libraries needed — the whole
    # molecule fits in six floats per bond length.
    H2_HAMS = {
        0.75: {"g0": 0.2252, "g1": 0.3435, "g2": -0.4347,
               "g3": 0.5716, "g4": 0.0910, "g5": 0.0910},
        1.50: {"g0": -0.2165, "g1": 0.1908, "g2": -0.0666,
               "g3": 0.4451, "g4": 0.1149, "g5": 0.1149},
    }
    return (H2_HAMS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The problem instance — a matrix, before any circuit

    Two protons a distance R apart, two electrons around them. The
    standard chemistry-to-qubits pipeline (pick an orbital basis, map
    electrons to qubits, cancel symmetries) lands the whole molecule
    **here**: a 4×4 matrix over the states |00⟩, |01⟩, |10⟩, |11⟩
    (character i = qubit i, as always this week). The job — the entire
    job — is *the smallest eigenvalue of this matrix*.

    Read the matrix like a map:
    - **|10⟩** — both electrons in the bonding orbital. This is
      **Hartree–Fock's answer**, and its diagonal entry is
      Hartree–Fock's energy.
    - **|01⟩** — both electrons in the *antibonding* orbital: the
      "wrong" configuration, expensive on its own.
    - The **off-diagonal coupling** between them (g4 + g5) is the
      quantum escape hatch: the true ground state is a *superposition*
      of the two, and mixing them buys the correlation energy.
    - |00⟩ and |11⟩ belong to a different electron-count symmetry —
      a proper run never lands there (which makes them a free noise
      alarm in Section 3's spirit).
    """)
    return


@app.cell(hide_code=True)
def _(H2_HAMS, np):
    def build_h2_matrix(g):
        """The literal 4×4 matrix of H = g0·I + g1·Z0 + g2·Z1 + g3·Z0Z1
        + g4·X0X1 + g5·Y0Y1, with rows/columns ordered |00⟩,|01⟩,|10⟩,|11⟩
        in the course's bit order (character i = qubit i)."""
        I2 = np.eye(2)
        Zm = np.diag([1.0, -1.0])
        Xm = np.array([[0.0, 1.0], [1.0, 0.0]])
        Ym = np.array([[0.0, -1j], [1j, 0.0]])
        # display index = 2·(bit of qubit 0) + (bit of qubit 1)
        M = (
            g["g0"] * np.eye(4)
            + g["g1"] * np.kron(Zm, I2)
            + g["g2"] * np.kron(I2, Zm)
            + g["g3"] * np.kron(Zm, Zm)
            + g["g4"] * np.kron(Xm, Xm)
            + (g["g5"] * np.kron(Ym, Ym)).real
        )
        return ["00", "01", "10", "11"], M.real

    H2_EXACT = {}
    H2_HF = {}
    for _R, _g in H2_HAMS.items():
        _labels, _M = build_h2_matrix(_g)
        H2_EXACT[_R] = float(np.linalg.eigvalsh(_M)[0])
        H2_HF[_R] = float(_M[2, 2])  # the |10⟩ diagonal = Hartree–Fock
    return H2_EXACT, H2_HF, build_h2_matrix


@app.cell(hide_code=True)
def _(H2_HAMS, build_h2_matrix, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for _ax, (_R, _g) in zip(_axes, H2_HAMS.items()):
        _labels, _M = build_h2_matrix(_g)
        _ax.imshow(_M, cmap="RdBu_r", vmin=-1.3, vmax=1.3)
        for _i in range(4):
            for _j in range(4):
                if abs(_M[_i, _j]) > 1e-12:
                    _ax.text(
                        _j,
                        _i,
                        f"{_M[_i, _j]:+.3f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="#1b2a4a",
                        fontweight="bold" if _i == _j == 2 else "normal",
                    )
        # amber box: the Hartree–Fock entry · magenta boxes: the quantum mix
        _ax.add_patch(
            plt.Rectangle((1.5, 1.5), 1, 1, fill=False, color="#D48F26", lw=2.5)
        )
        _ax.add_patch(
            plt.Rectangle((0.5, 1.5), 1, 1, fill=False, color="#B23B7B", lw=2)
        )
        _ax.add_patch(
            plt.Rectangle((1.5, 0.5), 1, 1, fill=False, color="#B23B7B", lw=2)
        )
        _ax.set_xticks(range(4))
        _ax.set_yticks(range(4))
        _ax.set_xticklabels([f"|{_b}⟩" for _b in _labels])
        _ax.set_yticklabels([f"|{_b}⟩" for _b in _labels])
        _hf = _M[2, 2]
        _ex = np.linalg.eigvalsh(_M)[0]
        _ax.set_title(
            f"R = {_R} Å · HF box {_hf:+.4f} · lowest eig {_ex:+.4f} Ha",
            fontsize=10,
        )
    _fig.suptitle(
        "the whole molecule · amber = Hartree–Fock's answer · "
        "magenta = the quantum escape hatch",
        fontsize=11,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(QuantumCircuit, backend, run_counts):
    # ── Hidden reference engines · Section 1 ────────────────────────────────
    # These power the EXPLORE widget (and the payoff fallback) so you can see
    # the physics BEFORE you've built anything. Same construction as the
    # solution folds — no peeking needed, and nothing here you won't build.
    def ref_ansatz(theta):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.ry(theta, 1)
        qc.cx(1, 0)
        return qc

    def ref_energy_full(theta, g, shots=4096):
        """Measured ⟨H⟩ exactly as the Section-1 solution does it:
        one Z-basis run + one X-basis run. Returns (E, z_counts)."""
        qz = ref_ansatz(theta)
        qz.measure_all()
        cz = run_counts(qz, backend, shots)
        qx = ref_ansatz(theta)
        qx.h([0, 1])
        qx.measure_all()
        cx_ = run_counts(qx, backend, shots)
        e_xx = pauli_exp(cx_, "Z0Z1")
        e = (
            g["g0"]
            + g["g1"] * pauli_exp(cz, "Z0")
            + g["g2"] * pauli_exp(cz, "Z1")
            + g["g3"] * pauli_exp(cz, "Z0Z1")
            + (g["g4"] + g["g5"]) * e_xx
        )
        return e, cz

    def ref_h2_energy(theta, g, shots=4096):
        return ref_energy_full(theta, g, shots)[0]

    return ref_energy_full, ref_h2_energy


@app.cell(hide_code=True)
def _(H2_EXACT, H2_HAMS, H2_HF, np):
    # The exact energy bowl per bond length — the 1-knob terrain. For the
    # state the ansatz family reaches, ⟨H⟩(θ) works out to
    #   E(θ) = (g0 − g3) + (g2 − g1)·cos θ + (g4 + g5)·sin θ
    # (algebra in the Section-1 folds). Precomputed as the Explore backdrop,
    # exactly like Day 4's precomputed (γ, β) terrain.
    BOWL_TH = np.linspace(-np.pi, np.pi, 361)
    BOWLS = {}
    BOWL_VIEW = {}
    for _R, _g in H2_HAMS.items():
        _e = (
            (_g["g0"] - _g["g3"])
            + (_g["g2"] - _g["g1"]) * np.cos(BOWL_TH)
            + (_g["g4"] + _g["g5"]) * np.sin(BOWL_TH)
        )
        BOWLS[_R] = _e
        _tstar = float(BOWL_TH[int(np.argmin(_e))])
        _gap = H2_HF[_R] - H2_EXACT[_R]
        BOWL_VIEW[_R] = {
            "tstar": _tstar,
            "zoom_x": (_tstar - 0.9, _tstar + 0.9),
            "zoom_y": (H2_EXACT[_R] - 0.18 * _gap, H2_HF[_R] + 0.85 * _gap),
        }
    return BOWLS, BOWL_TH, BOWL_VIEW


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Explore

    Pick a bond length, then drag **θ** — the one knob of a tiny ansatz
    circuit (you'll build it in a minute). **Left**: the measured
    Z-basis counts, 4 096 shots — watch amplitude move from |10⟩
    (Hartree–Fock's answer) into |01⟩ (the "wrong" configuration).
    **Middle**: the energy bowl — yesterday's (γ, β) terrain, now one
    knob wide — with a **live measured dot**. **Right**: the same bowl
    zoomed to the only part chemists care about.

    The dot starts parked at **θ = 0**, which *is* Hartree–Fock: the
    amber ledge. Drag θ negative and dive below the ledge — every
    milli-hartree you gain past amber is correlation energy, the part
    classical chemistry's shortcut writes off. The ink dotted line is
    the exact ground energy from the 4×4 matrix. Note the dot wobbles
    between reruns: it is a real 2×4096-shot *measurement*, not a
    formula.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bond_r = mo.ui.radio(
        options={
            "0.75 Å — natural bond length": 0.75,
            "1.50 Å — stretched (where the shortcut hurts)": 1.50,
        },
        value="1.50 Å — stretched (where the shortcut hurts)",
        label="bond length R",
    )
    theta_s = mo.ui.slider(
        start=-3.14,
        stop=3.14,
        value=0.0,
        step=0.02,
        label="θ · the ansatz knob",
        show_value=True,
    )
    mo.vstack([bond_r, theta_s])
    return bond_r, theta_s


@app.cell(hide_code=True)
def _(
    BOWLS,
    BOWL_TH,
    BOWL_VIEW,
    H2_EXACT,
    H2_HAMS,
    H2_HF,
    bond_r,
    np,
    plt,
    ref_energy_full,
    theta_s,
):
    _R = bond_r.value
    _g = H2_HAMS[_R]
    _e_meas, _cz = ref_energy_full(theta_s.value, _g, 4096)
    _hf, _ex = H2_HF[_R], H2_EXACT[_R]

    print(
        f"R = {_R} Å · θ = {theta_s.value:+.2f} · measured ⟨H⟩ = "
        f"{_e_meas:+.4f} Ha"
    )
    print(
        f"Hartree–Fock ledge = {_hf:+.4f} · exact ground = {_ex:+.4f} · "
        f"you are {1000 * (_e_meas - _ex):+.1f} mHa above exact"
    )

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(
        1, 3, figsize=(12.6, 4.1), gridspec_kw={"width_ratios": [0.9, 1.2, 1]}
    )

    # left — fixed-domain Z-basis histogram
    _dom = ["00", "01", "10", "11"]
    _tot = sum(_cz.values())
    _probs = [_cz.get(_b, 0) / _tot for _b in _dom]
    _cols = ["#6B7280", "#B23B7B", "#0E7C86", "#6B7280"]
    _ax1.bar(range(4), _probs, color=_cols)
    _ax1.set_xticks(range(4))
    _ax1.set_xticklabels(
        ["00\n(alarm)", "01\nantibonding²", "10\nbonding²\n★ HF", "11\n(alarm)"],
        fontsize=8,
    )
    _ax1.set_ylim(0, 1.05)
    _ax1.set_ylabel("probability")
    _ax1.set_title("Z-basis counts · 4096 shots")
    _ax1.grid(True, axis="y", alpha=0.3)

    # middle — the full bowl (the 1-knob terrain)
    _ax2.plot(BOWL_TH, BOWLS[_R], color="#0E7C86", lw=2)
    _ax2.axhline(_hf, color="#D48F26", ls="--", lw=1.5, label="Hartree–Fock")
    _ax2.axhline(_ex, color="#1b2a4a", ls=":", lw=1.5, label="exact ground")
    _ax2.scatter(
        [theta_s.value],
        [_e_meas],
        s=110,
        facecolor="#B23B7B",
        edgecolor="white",
        lw=1.5,
        zorder=3,
        label="your measurement",
    )
    _ax2.set_xlim(-np.pi, np.pi)
    _ax2.set_xlabel("θ")
    _ax2.set_ylabel("⟨H⟩ (Ha)")
    _ax2.set_title(f"the energy bowl · R = {_R} Å")
    _ax2.legend(loc="upper center", fontsize=8)
    _ax2.grid(True, alpha=0.3)

    # right — fixed zoom on the ledge-vs-bottom drama
    _v = BOWL_VIEW[_R]
    _ax3.plot(BOWL_TH, BOWLS[_R], color="#0E7C86", lw=2)
    _ax3.axhline(_hf, color="#D48F26", ls="--", lw=1.5)
    _ax3.axhline(_ex, color="#1b2a4a", ls=":", lw=1.5)
    _ax3.scatter(
        [theta_s.value],
        [_e_meas],
        s=110,
        facecolor="#B23B7B",
        edgecolor="white",
        lw=1.5,
        zorder=3,
    )
    _ax3.set_xlim(*_v["zoom_x"])
    _ax3.set_ylim(*_v["zoom_y"])
    _ax3.set_xlabel("θ")
    _ax3.set_title(
        f"zoom · correlation energy = {1000 * (_hf - _ex):.1f} mHa", fontsize=10
    )
    _ax3.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Predict

    Commit before building (out loud, or on paper — committing is the
    point):

    1. The stretched molecule's bowl bottoms out at θ ≈ −0.73; the
       natural-length bowl at θ ≈ −0.23, three times closer to zero.
       θ measures *how much of the second configuration is mixed in* —
       so which molecule is "more quantum", and what does that suggest
       about simulating **breaking** bonds (i.e., chemistry *happening*)?
    2. Can the dot ever sit **below** the ink dotted line? Careful —
       two answers: one for the *true* energy of any circuit state
       (yesterday's variational logic), one for a noisy 4096-shot
       *estimate* of it.
    3. The correlation gap at 1.50 Å vs at 0.75 Å: 2×? 4×? 10×? Check
       yourself against the zoom-panel titles after.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Build 1 — the ansatz (3 gates, 1 knob)

    Complete `h2_ansatz` below. The recipe:

    1. **`X` on qubit 0** — (given) start at Hartree–Fock's answer
       |10⟩: both electrons parked in the bonding orbital. At θ = 0
       your circuit must *be* the classical shortcut.
    2. **TODO — `RY(θ)` on qubit 1**: open the door to the second
       configuration by a controlled amount. Same one-qubit lean as
       Tuesday's Ry.
    3. **TODO — `CX(1, 0)`** (control qubit 1, target qubit 0): tie the
       two orbitals together — if the antibonding slot fills, the
       bonding slot must empty. This is the entangler; without it you
       have two independent qubits and no molecule.

    Result: cos(θ/2)·|10⟩ + sin(θ/2)·|01⟩ — a two-configuration
    superposition with one honest knob. That family contains the exact
    ground state (💡 fold explains why one knob is enough here).
    """)
    return


@app.cell
def _(QuantumCircuit):
    def h2_ansatz(theta: float) -> QuantumCircuit:
        """The 1-knob H₂ ansatz: |ψ(θ)⟩ = cos(θ/2)|10⟩ + sin(θ/2)|01⟩.
        No measurements here — the energy evaluator adds them per basis."""
        qc = QuantumCircuit(2)

        # (1) Start at chemistry's answer |10⟩. (Given.)
        qc.x(0)

        # (2) TODO — the knob: RY(theta) on qubit 1.

        # (3) TODO — the tie: CX with control qubit 1, target qubit 0.

        return qc

    return (h2_ansatz,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Solution — the ansatz (open if stuck)": mo.md("""
    Replace the two `TODO` comments with:

    ```python
    # (2) the knob
    qc.ry(theta, 1)

    # (3) the tie
    qc.cx(1, 0)
    ```

    Trace it: `X` makes |10⟩. `RY(θ)` on qubit 1 leans it toward
    cos(θ/2)·|10⟩ + sin(θ/2)·|11⟩. The `CX(1, 0)` then flips qubit 0
    *only* in the branch where qubit 1 is 1 — turning |11⟩ into |01⟩:

    cos(θ/2)·|10⟩ + sin(θ/2)·|01⟩.

    Exactly one electron pair, always — the circuit can't leave the
    physical sector, which is why |00⟩ and |11⟩ stay empty.
    """),
            "🧠 Why is |10⟩ 'chemistry's answer', and why does one knob suffice?": mo.md(r"""
    After the reduction, qubit 0 answers "is the **bonding**
    configuration occupied?" and qubit 1 "is the **antibonding** one?".
    Hartree–Fock picks the single cheapest configuration — both
    electrons in the bonding orbital, |10⟩ — and its energy is exactly
    the amber diagonal entry of the 4×4 matrix.

    The exact ground state can only mix states with the same electron
    count, so it lives in span{|10⟩, |01⟩}. A two-dimensional space
    needs exactly **one** mixing angle:

    $$|\psi(\theta)\rangle = \cos\tfrac{\theta}{2}\,|10\rangle +
      \sin\tfrac{\theta}{2}\,|01\rangle,$$

    and the measured energy works out to a plain sinusoid,

    $$E(\theta) = (g_0 - g_3) + (g_2 - g_1)\cos\theta +
      (g_4 + g_5)\sin\theta,$$

    which is the bowl the Explore widget draws. One knob, the whole
    physical space — for H₂. A 50-orbital molecule needs thousands of
    knobs, and *that* is when the classical optimizer starts earning
    its keep (and struggling — yesterday's ridges, at scale).
    """),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Build 2 — the energy meter

    A circuit doesn't print "energy". You *measure* it, the Day-4 way:
    run shots, read ±1 values off bitstrings, take the
    **probability-weighted average** — once per Hamiltonian term, then
    add them up with their g-coefficients. The helper `pauli_exp` below
    (given) does one term's average; your job in `h2_energy` is the two
    TODOs:

    - **TODO (a)** — the XX term is *invisible* to a Z-basis
      measurement (a phase-level fact by now). Rotate the X view into
      the Z view before measuring: `H` on **both** qubits — Tuesday's
      basis-change move.
    - **TODO (b)** — assemble the total: constant + each coefficient ×
      its measured average.
    """)
    return


@app.function
def pauli_exp(counts, which: str) -> float:
    """Probability-weighted average of a ±1 reading — Day 4's ⟨C⟩ move,
    pointed at physics. `which` picks the reading per bitstring:
    "Z0" → +1 if qubit 0 read 0, else −1;  "Z1" → same for qubit 1;
    "Z0Z1" → +1 if the two bits AGREE, else −1 (the parity)."""
    total = sum(counts.values())
    val = 0.0
    for bits, ct in counts.items():
        if which == "Z0":
            v = +1 if bits[0] == "0" else -1
        elif which == "Z1":
            v = +1 if bits[1] == "0" else -1
        else:  # "Z0Z1"
            v = +1 if bits[0] == bits[1] else -1
        val += v * ct
    return val / total


@app.cell
def _(backend, h2_ansatz, run_counts):
    def h2_energy(theta: float, g: dict, shots: int = 4096):
        """Measured ⟨H⟩ for the ansatz at angle theta: one Z-basis run for
        the Z0 / Z1 / Z0Z1 terms, one X-basis run for the XX (and YY) term,
        then the g-weighted sum. Returns energy in hartree."""
        # -- Z-basis run (given): three averages from one set of shots --
        qz = h2_ansatz(theta)
        qz.measure_all()
        cz = run_counts(qz, backend, shots)
        e_z0 = pauli_exp(cz, "Z0")
        e_z1 = pauli_exp(cz, "Z1")
        e_zz = pauli_exp(cz, "Z0Z1")

        # -- X-basis run --
        qx = h2_ansatz(theta)
        # TODO (a): rotate the X view into the Z view — H on BOTH qubits.

        qx.measure_all()
        cx_ = run_counts(qx, backend, shots)
        e_xx = pauli_exp(cx_, "Z0Z1")  # parity in the rotated basis = ⟨X0X1⟩
        e_yy = e_xx  # for this ansatz's states YY reads the same (🧠 fold)

        # TODO (b): assemble the probability-weighted total — replace None
        # with: constant + each g-coefficient × its measured average
        # (g0, then g1·e_z0, g2·e_z1, g3·e_zz, g4·e_xx, g5·e_yy).
        E = None
        return E

    return (h2_energy,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Solution — the energy meter (open if stuck)": mo.md("""
    **TODO (a)** — one line after `qx = h2_ansatz(theta)`:

    ```python
    qx.h([0, 1])
    ```

    **TODO (b)** — replace `E = None` with:

    ```python
    E = (
        g["g0"]
        + g["g1"] * e_z0
        + g["g2"] * e_z1
        + g["g3"] * e_zz
        + g["g4"] * e_xx
        + g["g5"] * e_yy
    )
    ```

    Six numbers, one weighted sum — that's a Hamiltonian expectation
    value. Real VQE codes do exactly this, just with thousands of terms
    and smarter shot-sharing between them.
    """),
            "🧠 Why is energy a probability-weighted average — again?": mo.md(r"""
    Yesterday: $\langle C\rangle = \sum_x P(x)\,C(x)$ — sample candidate
    splits, average their scores, and the smooth average is what the
    optimizer climbs. Today is the same statement in physics clothing:

    $$\langle H\rangle = \sum_{\text{terms } P} g_P \,\langle P\rangle,
      \qquad \langle P\rangle = \sum_{\text{bitstrings}} P(\text{bits})
      \cdot (\pm 1),$$

    each term's ±1 read off the measured bits, averaged with their
    probabilities. Quantum mechanics simply *defines* the energy of a
    state as this weighted average of readings — which is why a noisy
    estimate of it can wobble *below* the exact ground energy even
    though the true expectation never can (the variational bound). Keep
    that distinction — it settles Predict #2.
    """),
            "🧠 Why H-gates for XX — and where did YY go?": mo.md(r"""
    A Z-basis measurement only sees amplitude *sizes*, and the XX term
    lives in the *relative phase* between |10⟩ and |01⟩ — same
    invisibility as Wednesday's oracle mark. `H` on both qubits rotates
    X-eigenstates into Z-eigenstates, so the ordinary parity ⟨Z0Z1⟩ of
    the rotated circuit **is** ⟨X0X1⟩ of the original state.

    YY: our ansatz only makes states $\cos\frac{\theta}{2}|10\rangle +
    \sin\frac{\theta}{2}|01\rangle$ with real amplitudes, and on that
    family $\langle Y_0Y_1\rangle = \langle X_0X_1\rangle$ exactly
    (both equal $\sin\theta$). Since $g_4 = g_5$, we measure XX once
    and count it twice. A general-purpose VQE measures YY in its own
    basis (S† then H) — one more run, same idea. And if this shortcut
    were a lie, the exact-diagonalization referee below would catch it.
    """),
        }
    )
    return


@app.cell(hide_code=True)
def _(H2_HAMS, H2_HF):
    # ── Guard helpers · Section 1 (teaching diagnostics, not magic) ─────────
    def ansatz_is_todo(fn):
        """Detect the shipped/TODO state of h2_ansatz."""
        try:
            ops = fn(0.7).count_ops()
        except Exception:
            return True
        return "ry" not in ops or "cx" not in ops

    def energy_state(fn):
        """Diagnose h2_energy: 'todo' (sum not assembled), 'xbasis'
        (H-rotation missing → variational-bound violation), or 'ok'."""
        g = H2_HAMS[0.75]
        try:
            e = fn(0.0, g, shots=2048)
        except Exception:
            return "todo"
        if e is None:
            return "todo"
        if abs(e - H2_HF[0.75]) > 0.05:
            return "xbasis"
        return "ok"

    return ansatz_is_todo, energy_state


@app.cell(hide_code=True)
def _(
    H2_EXACT,
    H2_HAMS,
    ansatz_is_todo,
    energy_state,
    h2_ansatz,
    h2_energy,
    ref_h2_energy,
):
    # Check: YOUR ansatz + YOUR energy meter at the stretched molecule's
    # sweet spot θ = −0.73, against the exact eigenvalue.
    _g = H2_HAMS[1.50]
    _st_a = ansatz_is_todo(h2_ansatz)
    _st_e = energy_state(h2_energy)

    if _st_a:
        print("⚠️  Your ansatz still answers |10⟩ for every θ — the RY knob")
        print("    and the CX tie are TODO. A knob-less circuit is just")
        print("    Hartree–Fock on repeat: the optimizer would see a flat")
        print("    line and learn nothing.")
    if _st_e == "todo":
        print("⚠️  h2_energy returned None — TODO (b) isn't assembled yet.")
        print("    Six averages are measured and waiting; they just need")
        print("    their g-weights and a sum.")
    elif _st_e == "xbasis":
        print("⚠️  Your energy dips ~0.2 Ha BELOW the exact ground energy —")
        print("    impossible for any real state (the variational bound!).")
        print("    That's the fingerprint of TODO (a): without the H-gates")
        print("    you read the Z parity twice and call it XX. Rotate the")
        print("    basis and rerun.")
    if not _st_a and _st_e == "ok":
        _e_you = h2_energy(-0.73, _g, 8192)
        print(f"Your meter at (R = 1.50 Å, θ = −0.73): {_e_you:+.4f} Ha")
        print(f"Exact ground (matrix eigenvalue):      {H2_EXACT[1.50]:+.4f} Ha")
        print("Within a few mHa? Then your circuit and your meter agree")
        print("with the referee — the machine is real. Hand it the knob.")
    else:
        _e_ref = ref_h2_energy(-0.73, _g, 8192)
        print(f"(Reference meter at the same point: {_e_ref:+.4f} Ha vs exact")
        print(f" {H2_EXACT[1.50]:+.4f} Ha — this is what yours will read once")
        print("  the TODOs are in.)")

    h2_ansatz(-0.73).draw("mpl")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Payoff — the optimizer earns the correlation energy

    Same closing move as yesterday: hand the knob to COBYLA, feed it
    **−nothing** this time — the energy is already a minimization — and
    let it walk downhill from θ = 0 (start where chemistry starts).
    Both bond lengths, ~40 measured evaluations each, then a high-shot
    readout of the final answer. The referee: `numpy.linalg.eigvalsh`
    on the same 4×4 matrix — the exact answer, printed next to yours.
    """)
    return


@app.cell
def _(
    H2_EXACT,
    H2_HAMS,
    H2_HF,
    ansatz_is_todo,
    energy_state,
    h2_ansatz,
    h2_energy,
    minimize,
    plt,
    ref_h2_energy,
):
    # ── The VQE run: both bond lengths, YOUR machine if it's ready ─────────
    if ansatz_is_todo(h2_ansatz) or energy_state(h2_energy) != "ok":
        _efn = ref_h2_energy
        print("⚠️  Running on the REFERENCE machine — a Section-1 TODO is")
        print("    still open (see the check above). The physics below is")
        print("    real; finish the build to make the energies YOURS.")
    else:
        _efn = h2_energy
        print("Engine: YOUR ansatz + YOUR meter. These energies are yours.")

    vqe_results = {}
    for _R, _g in H2_HAMS.items():
        _res = minimize(
            lambda _t: _efn(_t[0], _g, 8192),
            [0.0],
            method="COBYLA",
            options={"maxiter": 40, "rhobeg": 0.35},
        )
        _e_final = _efn(_res.x[0], _g, 32768)
        vqe_results[_R] = (float(_res.x[0]), float(_e_final))
        _err = 1000 * (_e_final - H2_EXACT[_R])
        print(
            f"R = {_R} Å: θ* = {_res.x[0]:+.3f} → VQE {_e_final:+.4f} Ha · "
            f"exact {H2_EXACT[_R]:+.4f} · HF {H2_HF[_R]:+.4f} · "
            f"error {_err:+.1f} mHa"
        )

    print("Referee (numpy eigvalsh on the same matrices): "
          + ", ".join(f"{_R} Å → {H2_EXACT[_R]:+.4f}" for _R in H2_HAMS))
    print("Chemical accuracy is ±1.6 mHa — the gray band below. A noisy")
    print("estimate may even dip a hair BELOW exact; the true expectation")
    print("never does (Predict #2).")

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for _ax, _R in zip(_axes, H2_HAMS):
        _hf, _ex = H2_HF[_R], H2_EXACT[_R]
        _gap = _hf - _ex
        _th, _ev = vqe_results[_R]
        _ax.axhline(_hf, color="#D48F26", ls="--", lw=2,
                    label="Hartree–Fock (best single answer)")
        _ax.axhline(_ex, color="#1b2a4a", ls=":", lw=2,
                    label="exact (diagonalization)")
        _ax.axhspan(_ex - 0.0016, _ex + 0.0016, color="#6B7280", alpha=0.3,
                    label="chemical accuracy ±1.6 mHa")
        _ax.scatter([0.5], [_ev], s=170, facecolor="#0E7C86",
                    edgecolor="white", lw=1.6, zorder=3,
                    label=f"your VQE = {_ev:+.4f} Ha")
        _ax.annotate(
            "", xy=(0.8, _ex), xytext=(0.8, _hf),
            arrowprops=dict(arrowstyle="<->", color="#B23B7B", lw=1.6),
        )
        _ax.annotate(
            f"correlation energy\n{1000 * _gap:.1f} mHa — what the\n"
            "quantum part earns",
            (0.82, _ex + 0.45 * _gap), fontsize=8.5, color="#B23B7B",
        )
        _ax.set_xlim(0, 1.15)
        _ax.set_xticks([])
        _ax.set_ylim(_ex - 0.35 * _gap, _hf + 0.35 * _gap)
        _ax.set_ylabel("energy (Ha)")
        _ax.set_title(f"R = {_R} Å")
        _ax.legend(loc="upper left", fontsize=8)
        _ax.grid(True, axis="y", alpha=0.3)
    _fig.suptitle("VQE vs the classical shortcut vs certain truth")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Baseline check** — before moving on:

    - Did your VQE land within a few mHa of −1.1456 Ha (0.75 Å) and
      −1.0067 Ha (1.50 Å)? That wobble is shot noise; ±1.6 mHa
      ("chemical accuracy") is roughly the noise floor at this shot
      budget. More shots buy more digits — with real money, on real
      clouds (this morning's per-shot pricing, remember).
    - Hartree–Fock got 98 % of the energy at 0.75 Å. So why does anyone
      care? Because chemistry lives in *differences*: stretch the bond
      — the very act of a reaction — and HF's miss quadruples to
      88 mHa (2.4 eV), far bigger than the energies that decide whether
      reactions go. Bond-breaking is where classical shortcuts die and
      quantum simulation earns its keep.
    - Say the loop once more, new nouns: *quantum circuit reports ⟨H⟩,
      classical optimizer proposes θ, repeat.* Yesterday it cut graphs;
      today it solved a molecule. Same machine — that's why it's
      one lecture, not two.

    (One promise before the suit goes on: the same loop meets a
    **56-carbon molecule** at the end of this notebook — past where any
    statevector simulator can follow.)

    Molecule paid. Now the same machine puts on a suit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2 · The QAOA portfolio (stretch)

    **You manage a fund now.** Six assets on the board, room in the
    mandate for exactly **three**. Every asset promises a return; every
    *pair* of assets whispers a correlation — solar and chips boom
    together, and the desalination utility quietly moves *against* the
    solar farm. Your score is Markowitz's classic:

    > **utility = expected return − λ · risk**, risk = how much the
    > portfolio wobbles as a whole — which is a *pairwise* affair.

    A portfolio is a **bitstring** — bit i says whether asset i is in —
    so 6 assets make 64 candidate portfolios, and the objective is
    linear-plus-pairwise in the bits. That shape has a name from this
    morning's vendor slides: a **QUBO** — and it means yesterday's
    machine runs it *unchanged*. Same stamp, same mixer; the cost box
    just prices covariance instead of counting friendships. (Max-Cut,
    it turns out, was a QUBO in a graph costume all along.)

    **The stakes, measured**: at risk-aversion λ = 1 the greedy
    top-returns trio keeps a utility of just **0.010** after the risk
    bill — while the true optimum keeps **0.140**, fourteen times more,
    by hiring the *hedge*. Your machine's job: find it, and get graded
    against all 64 in public. (Honesty paren: n = 6 brute-forces in
    microseconds — banks' interest is this *encoding* at n where 2ⁿ
    dies, and it remains a bet, not a result.)
    """)
    return


@app.cell
def _(np):
    # ── THE MARKET · EDIT ME ────────────────────────────────────────────────
    # Six made-up-but-plausible assets: expected yearly return (fraction)
    # and covariance of returns. Tweak and rerun — everything recomputes.
    TICKERS = ["SOLR", "CHIP", "SHIP", "BANK", "H2OX", "MEDS"]
    #            solar   semis  logist  bank   desal  pharma
    MU = np.array([0.14, 0.12, 0.09, 0.07, 0.05, 0.08])

    # Covariance matrix Σ (symmetric, positive semidefinite — eigvalsh > 0):
    # diagonal = each asset's own variance; off-diagonal = how pairs co-move.
    # Note Σ[SOLR, H2OX] < 0: the desal utility runs OPPOSITE the solar farm.
    SIGMA = np.array(
        [
            [0.100, 0.040, 0.010, 0.010, -0.010, 0.000],
            [0.040, 0.070, 0.010, 0.010, 0.000, 0.010],
            [0.010, 0.010, 0.050, 0.020, 0.000, 0.000],
            [0.010, 0.010, 0.020, 0.030, 0.000, 0.010],
            [-0.010, 0.000, 0.000, 0.000, 0.010, 0.000],
            [0.000, 0.010, 0.000, 0.010, 0.000, 0.040],
        ]
    )

    N_ASSETS = len(TICKERS)
    K = 3  # the mandate: exactly 3 assets
    LAM = 1.0  # risk aversion for the QAOA run (Explore has its own slider)
    PENALTY_A = 0.4  # fine per squared unit of budget violation (QUBO-style)
    return K, LAM, MU, N_ASSETS, PENALTY_A, SIGMA, TICKERS


@app.cell(hide_code=True)
def _(np, plt):
    def draw_market(names, mu, Sigma, chosen=None, title=""):
        """The market as a graph: nodes = assets (size = own variance,
        label = ticker + expected return), edges = covariances (width =
        strength; gray solid = move together, teal dashed = move opposite —
        a hedge). chosen = set of indices to highlight in amber."""
        n = len(names)
        ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
        x, y = np.cos(ang), np.sin(ang)
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        for i in range(n):
            for j in range(i + 1, n):
                c = Sigma[i, j]
                if abs(c) < 1e-12:
                    continue
                ax.plot(
                    [x[i], x[j]],
                    [y[i], y[j]],
                    color="#0E7C86" if c < 0 else "#9a9a9a",
                    lw=1.0 + 70 * abs(c),
                    ls=(0, (4, 2)) if c < 0 else "-",
                    zorder=1,
                )
        for i in range(n):
            in_pick = chosen is not None and i in chosen
            ax.scatter(
                [x[i]],
                [y[i]],
                s=1150 + 12000 * Sigma[i, i],
                c="#D48F26" if in_pick else "#1b2a4a",
                zorder=2,
            )
            ax.annotate(
                f"{names[i]}\nq{i} · {100 * mu[i]:.0f}%",
                (x[i], y[i]),
                ha="center",
                va="center",
                fontsize=9.5,
                color="white",
                fontweight="bold",
                zorder=3,
            )
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)
        fig.tight_layout()
        return fig

    return (draw_market,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The market you're cutting into

    Before any encoding: *this* is the problem. Node size = how much an
    asset wobbles on its own (SOLR is the wildest thing on the board).
    Edge width = how strongly a pair moves **together** (gray) — and
    the one **teal dashed** edge is a pair that moves **opposite**:
    SOLR and H2OX, growth vs. defensive. Asset i is qubit i is bit
    position i — portfolio `100011` means SOLR + H2OX + MEDS are in.
    """)
    return


@app.cell(hide_code=True)
def _(MU, SIGMA, TICKERS, draw_market):
    draw_market(
        TICKERS,
        MU,
        SIGMA,
        title="the market · 6 assets · node ↔ qubit ↔ bit position",
    )
    return


@app.cell(hide_code=True)
def _(K, MU, N_ASSETS, PENALTY_A, SIGMA, np):
    # ── Hidden reference engine · Section 2 scorekeeper ─────────────────────
    # Powers the Explore widget (and the payoff fallback) — same formula as
    # the Section-2 solution fold.
    from itertools import combinations

    def ref_utility(bits, lam):
        x = np.array([int(ch) for ch in bits], float)
        ret = float(MU @ x)
        risk = float(x @ SIGMA @ x)
        return ret - lam * risk - PENALTY_A * (x.sum() - K) ** 2

    FEAS = [
        "".join("1" if i in cmb else "0" for i in range(N_ASSETS))
        for cmb in combinations(range(N_ASSETS), K)
    ]
    return FEAS, ref_utility


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Explore

    The slider is **λ — the risk-aversion knob**: how many points of
    return you'd pay to shed one point of wobble. **Left**: all 20
    legal (3-asset) portfolios in a fixed line-up; heights are their
    utilities at your λ, winner in amber. **Right**: the same 20 as
    dots on the **return-vs-risk plane** — the dots never move (returns
    and risks are properties of the portfolios); λ only changes *which
    dot wins*.

    It starts parked at **λ = 1**, where the story flips: the winner is
    SOLR + H2OX + MEDS — the wildest asset *plus its hedge*. Now drag λ
    to **0**: greed rules, the three biggest returns win (utility
    0.35). Drag toward **2**: everything sinks as the risk bill eats
    the returns, and the winner goes fully defensive. One knob, three
    different funds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    lam_s = mo.ui.slider(
        start=0.0,
        stop=2.0,
        value=1.0,
        step=0.05,
        label="λ · risk aversion",
        show_value=True,
    )
    lam_s
    return (lam_s,)


@app.cell(hide_code=True)
def _(FEAS, MU, SIGMA, TICKERS, lam_s, np, plt, ref_utility):
    _lam = lam_s.value
    _utils = [ref_utility(_b, _lam) for _b in FEAS]
    _iwin = int(np.argmax(_utils))
    _names = [
        "+".join(TICKERS[_i][:2] for _i in range(6) if _b[_i] == "1")
        for _b in FEAS
    ]
    _wtick = [TICKERS[_i] for _i in range(6) if FEAS[_iwin][_i] == "1"]
    print(
        f"λ = {_lam:.2f} · best portfolio: {' + '.join(_wtick)} "
        f"(bits {FEAS[_iwin]}) · utility {_utils[_iwin]:+.3f}"
    )

    _fig, (_ax1, _ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.3), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    _cols = ["#D48F26" if _i == _iwin else "#0E7C86" for _i in range(20)]
    _ax1.bar(range(20), _utils, color=_cols)
    _ax1.set_xticks(range(20))
    _ax1.set_xticklabels(_names, rotation=90, fontsize=7.5)
    _ax1.set_ylim(-0.45, 0.45)
    _ax1.axhline(0, color="#9a9a9a", lw=0.8)
    _ax1.set_ylabel("utility = return − λ·risk")
    _ax1.set_title(f"all 20 legal portfolios at λ = {_lam:.2f} · winner in amber")
    _ax1.grid(True, axis="y", alpha=0.3)

    _rets = [float(MU @ np.array([int(c) for c in _b])) for _b in FEAS]
    _risks = [
        float(
            np.array([int(c) for c in _b])
            @ SIGMA
            @ np.array([int(c) for c in _b])
        )
        for _b in FEAS
    ]
    _ax2.scatter(_risks, _rets, s=55, c="#0E7C86", zorder=2)
    _ax2.scatter(
        [_risks[_iwin]],
        [_rets[_iwin]],
        s=260,
        marker="*",
        c="#D48F26",
        edgecolor="#1b2a4a",
        zorder=3,
        label="winner at this λ",
    )
    _ax2.set_xlim(0.05, 0.40)
    _ax2.set_ylim(0.18, 0.37)
    _ax2.set_xlabel("risk (portfolio variance x·Σ·x)")
    _ax2.set_ylabel("expected return μ·x")
    _ax2.set_title("the fixed cloud of portfolios — λ picks the corner")
    _ax2.legend(fontsize=8, loc="lower right")
    _ax2.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Predict

    Commit before you build:

    1. SOLR is by far the most volatile thing on the board (its own
       variance is 0.10 — triple anyone else's). At λ = 1, does it
       survive the cut? Why might a wild asset be *safer inside* a
       portfolio than its variance suggests? (Look at the teal dashed
       edge.)
    2. H2OX has the *smallest* return of all six. Why does every
       risk-aware winner keep it anyway?
    3. Commit to three tickers you believe win at λ = 1 — then check
       against the amber bar above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Build 1 — the scorekeeper

    Complete `portfolio_utility` below — this section's `maxcut_cost`.
    Given a portfolio bitstring, return **return − λ·risk − budget
    fine**:

    - `ret` (given): μ·x, the weighted sum of expected returns.
    - **TODO — risk**: the quadratic form `x @ SIGMA @ x` — every pair
      counted, because wobbles add *pairwise* (🧠 fold below).
    - **TODO — the total**: `ret - lam * risk - PENALTY_A * (x.sum() - K) ** 2`.
      The last piece is the **budget fine**: portfolios of the wrong
      size pay A per squared asset of violation. Soft constraints are
      how QUBOs say "exactly k, please".

    Every later cell — the QAOA score, the honesty check, your final
    fund — is judged by *your* scorekeeper.
    """)
    return


@app.cell
def _(MU, np):
    def portfolio_utility(bits: str, lam: float):
        """Utility of a portfolio bitstring: expected return − λ·risk −
        budget fine. bits[i] = '1' iff asset i is in the portfolio."""
        x = np.array([int(ch) for ch in bits], float)

        # Expected return: μ·x. (Given.)
        ret = float(MU @ x)

        # TODO — risk: the portfolio's variance, x·Σ·x  (x @ SIGMA @ x).
        risk = None

        # TODO — total: ret − lam·risk − PENALTY_A·(x.sum() − K)²
        U = None
        return U

    return (portfolio_utility,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Solution — the scorekeeper (open if stuck)": mo.md("""
    ```python
    risk = float(x @ SIGMA @ x)
    U = ret - lam * risk - PENALTY_A * (x.sum() - K) ** 2
    ```

    Three terms, three stories: what you expect to earn, what you pay
    for wobble, and the fine for breaking the "exactly 3" mandate.
    """),
            "🧠 Why is risk a PAIRWISE sum (x·Σ·x)?": mo.md(r"""
    A portfolio's return is a random variable — next year's market is a
    coin flip weighted by probabilities, and this week you've averaged
    over enough coin flips to smell what's coming: the *variance* of a
    sum is not the sum of variances,

    $$\mathrm{Var}\Big(\sum_i x_i R_i\Big) =
      \sum_i \sum_j x_i x_j\,\mathrm{Cov}(R_i, R_j)
      = x^\top \Sigma\, x.$$

    Every **pair** contributes: two assets that surge together
    (Cov > 0) *add* wobble beyond their own variances — and a pair that
    moves oppositely (Cov < 0) *cancels* wobble. That cancellation is
    the entire meaning of the word **hedge**, and it's why the wild
    SOLR plus quiet anti-correlated H2OX can be tamer than two
    medium-risk assets that co-move. Diversification isn't owning many
    things; it's owning things that *disagree*.
    """),
            "🧠 The budget fine — how a QUBO says 'exactly 3'": mo.md(r"""
    Quantum circuits sample **all** bitstrings; nothing physically
    forbids a 5-asset draw. So the constraint moves into the score:

    $$-A\,(x_1 + \dots + x_6 - 3)^2$$

    is 0 for legal portfolios and grows quadratically with violation —
    with A = 0.4 chosen so that no return can bribe its way past the
    fine (max return gain from one extra asset: 0.14 < 0.4). Expand the
    square and it's linear-plus-pairwise in the bits — so it slots
    into the same QUBO the phase stamp already prices. Watch the
    honesty check later: the machine *learns the mandate* — feasible
    draws jump from a blind 31 % to ~95 %.
    """),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Build 2 — yesterday's machine, rehired

    The phase stamp is again a **black box**: call
    `make_utility_layer(gamma, lam)` and get one opaque `utility(γ)`
    gate that stamps every portfolio |x⟩ with a phase set by *its own
    utility* — exactly like yesterday's `make_cost_layer`, except the
    box prices returns, covariances and the budget fine instead of
    counting friendships (a "🔍 how is it built?" fold waits below).

    Complete `portfolio_qaoa`. You wrote this exact loop yesterday:

    1. `H` on every qubit — all 64 portfolios at once. (Given.)
    2. **TODO — the stamp**: `qc.append(make_utility_layer(gamma, lam),
       range(N_ASSETS))`.
    3. **TODO — the mixer**: `RX(2β)` on every qubit — the stamp writes
       phases, the mixer cashes them into probabilities. No mixer, no
       machine.
    4. Measure. (Given.)
    """)
    return


@app.cell(hide_code=True)
def _(K, MU, N_ASSETS, PENALTY_A, QuantumCircuit, SIGMA, np):
    # ── THE BLACK BOX · given ────────────────────────────────────────────────
    # Same contract as Day 4's make_cost_layer: a gate you may CALL, not
    # open. One opaque box that phase-stamps every portfolio |x⟩ according
    # to its own utility U(x). Internals in the "How is utility(γ) built?"
    # fold — finish your build before opening it.
    def make_utility_layer(gamma: float, lam: float):
        """The black box: one opaque gate stamping each basis state |x⟩
        with a phase set by its utility U(x) = μ·x − λ·x·Σ·x − A(Σx−K)².
        Call it with `qc.append(make_utility_layer(g, lam), range(N_ASSETS))`."""
        _lin = MU - lam * np.diag(SIGMA) - PENALTY_A * (1 - 2 * K)
        _quad = -2.0 * lam * SIGMA - 2.0 * PENALTY_A  # off-diagonal use only
        _h = np.array(
            [
                _lin[_i] / 2
                + sum(_quad[_i, _j] for _j in range(N_ASSETS) if _j != _i) / 4
                for _i in range(N_ASSETS)
            ]
        )
        _J = {
            (_i, _j): -_quad[_i, _j] / 4
            for _i in range(N_ASSETS)
            for _j in range(N_ASSETS)
            if _i < _j
        }
        _s = 1.0 / max(np.abs(_h).max(), max(abs(_v) for _v in _J.values()))
        _qc = QuantumCircuit(N_ASSETS, name="utility")
        for _i in range(N_ASSETS):
            _qc.rz(2 * gamma * _s * _h[_i], _i)
        for (_i, _j), _v in _J.items():
            _qc.rzz(2 * gamma * _s * _v, _i, _j)
        return _qc.to_gate(label="utility(γ)")

    return (make_utility_layer,)


@app.cell(hide_code=True)
def _(N_ASSETS, QuantumCircuit, make_utility_layer):
    # Reference engine (hidden): a full working portfolio QAOA, powering the
    # check baseline and the payoff fallback — same deal as Day 4's qaoa_ref.
    def pf_ref(gammas, betas, lam):
        qc = QuantumCircuit(N_ASSETS, N_ASSETS)
        qc.h(range(N_ASSETS))
        for g, b in zip(gammas, betas):
            qc.append(make_utility_layer(g, lam), range(N_ASSETS))
            for q in range(N_ASSETS):
                qc.rx(2 * b, q)
        qc.measure(range(N_ASSETS), range(N_ASSETS))
        return qc

    return (pf_ref,)


@app.cell
def _(N_ASSETS, QuantumCircuit):
    def portfolio_qaoa(gammas, betas, lam: float) -> QuantumCircuit:
        """p-layer QAOA for the portfolio: H on all qubits, then per layer a
        utility stamp (black box, knob gammas[k]) and an RX(2·betas[k]) mixer,
        then measure. len(gammas) == len(betas) == p."""
        qc = QuantumCircuit(N_ASSETS, N_ASSETS)

        # (1) Every portfolio at once. (Given.)
        qc.h(range(N_ASSETS))

        for gamma, beta in zip(gammas, betas):
            # (2) TODO — THE STAMP: call the black box (don't open it):
            pass

            # (3) TODO — THE MIXER: RX(2*beta) on EVERY qubit:
            pass

        # (4) Measure. (Given — leave this so the cell runs while you build.)
        qc.measure(range(N_ASSETS), range(N_ASSETS))
        return qc

    return (portfolio_qaoa,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Solution — one QAOA layer (open if stuck)": mo.md("""
    Replace the two `TODO` blocks (and both `pass` lines) with:

    ```python
    # (2) the stamp — the black box, called as a function.
    qc.append(make_utility_layer(gamma, lam), range(N_ASSETS))

    # (3) the mixer — RX(2β) on every qubit.
    for q in range(N_ASSETS):
        qc.rx(2 * beta, q)
    ```

    Byte-for-byte the rhythm of yesterday's `qaoa_circuit` — only the
    box changed jobs. That's the point.
    """),
            "🔍 How is the black box utility(γ) built? (open after your build works)": mo.md(r"""
    Expand U(x) and every piece is linear or pairwise in the bits:

    $$U(x) = \sum_i a_i x_i + \sum_{i<j} b_{ij}\, x_i x_j + \text{const},$$

    with $a_i = \mu_i - \lambda\Sigma_{ii} - A(1 - 2K)$ and
    $b_{ij} = -2\lambda\Sigma_{ij} - 2A$. Substitute
    $x_i = (1 - z_i)/2$ and the whole thing becomes fields and
    couplings on spins — so the stamp is just:

    ```python
    for i in range(N_ASSETS):
        qc.rz(2 * gamma * s * h[i], i)        # one RZ per asset
    for (i, j), J in couplings.items():
        qc.rzz(2 * gamma * s * J[i, j], i, j)  # one RZZ per PAIR
    return qc.to_gate(label="utility(γ)")
    ```

    — yesterday's `rzz`-per-edge, plus an `rz` per node because assets
    (unlike friendships) also have *individual* scores. The factor `s`
    rescales the coefficients so γ's useful range matches yesterday's
    sliders. Note what isn't in the box: nothing about which portfolio
    wins. It only knows μ, Σ, λ, A — the phases do the ranking.
    """),
            "🧠 Why does the SAME machine run finance and graphs?": mo.md(r"""
    QAOA never cared about graphs. It cares that the score is
    **diagonal** — each bitstring has a number — and that the number is
    at most pairwise in the bits, so it can be stamped with RZ/RZZ
    phases. Any such problem is a **QUBO** (quadratic unconstrained
    binary optimization): Max-Cut is the special case where every
    $a_i = 0$ and every pair scores alike; portfolios, timetables,
    logistics routes, protein side-chains — same box, different
    numbers. This is why this morning's vendor slides kept saying
    "QUBO": it's the *file format* of quantum optimization. Learn to
    write your problem as one, and every QAOA machine — and every
    quantum annealer — will take it as input.
    """),
        }
    )
    return


@app.cell(hide_code=True)
def _(N_ASSETS, np):
    # ── Guard helpers · Section 2 ───────────────────────────────────────────
    def utility_is_todo(fn):
        """Detect the shipped/TODO state of portfolio_utility."""
        try:
            vals = [fn(format(k, f"0{N_ASSETS}b"), 1.0) for k in range(2**N_ASSETS)]
        except Exception:
            return True
        if any(v is None for v in vals):
            return True
        return len(set(np.round(np.array(vals, float), 9))) == 1

    def pf_circuit_is_todo(fn):
        """Detect the shipped/TODO state of portfolio_qaoa."""
        try:
            ops = fn([0.4], [0.4], 1.0).count_ops()
        except Exception:
            return True
        return "utility" not in ops or "rx" not in ops

    return pf_circuit_is_todo, utility_is_todo


@app.cell(hide_code=True)
def _(
    LAM,
    backend,
    pf_circuit_is_todo,
    pf_ref,
    portfolio_qaoa,
    portfolio_utility,
    ref_utility,
    run_counts,
    utility_is_todo,
):
    # Check: YOUR circuit at fixed knobs, scored by YOUR scorekeeper.
    # Drawn at p = 2 so the stamp–stir rhythm shows (one closed utility(γ)
    # box per layer, matching the lecture's framing).
    _util = portfolio_utility if not utility_is_todo(portfolio_utility) else ref_utility
    if utility_is_todo(portfolio_utility):
        print("ℹ️  Your portfolio_utility still returns None (Build 1 TODO) —")
        print("    borrowing the reference scorer so this check can speak.")

    _counts_you = run_counts(portfolio_qaoa([0.9], [0.6], LAM), backend, 4096)
    _counts_ref = run_counts(pf_ref([0.9], [0.6], LAM), backend, 4096)
    _mu_you = sum(_util(_b, LAM) * _c for _b, _c in _counts_you.items()) / 4096
    _mu_ref = sum(_util(_b, LAM) * _c for _b, _c in _counts_ref.items()) / 4096
    print(f"Your machine at (γ, β) = (0.90, 0.60): ⟨U⟩ = {_mu_you:+.4f}")
    print(f"Reference machine at the same knobs:   ⟨U⟩ = {_mu_ref:+.4f}")

    if pf_circuit_is_todo(portfolio_qaoa):
        _ops = portfolio_qaoa([0.9], [0.6], LAM).count_ops()
        if "utility" not in _ops:
            print("⚠️  No utility(γ) box in your circuit → TODO (2) isn't in.")
            print("    Unstamped portfolios all look alike — there's nothing")
            print("    for interference to sort.")
        if "rx" not in _ops:
            print("⚠️  No RX mixer in your circuit → TODO (3) isn't in.")
            print("    ⟨U⟩ sits at the blind-guessing level: the stamp only")
            print("    turns phases, and a phase changes NO probabilities —")
            print("    Wednesday's invisible mark, third day running.")
    else:
        print("If the two ⟨U⟩ readings agree to ~±0.05, your machine matches")
        print("the reference engine — it's real. On to the knobs.")

    portfolio_qaoa([0.9, 0.9], [0.6, 0.6], LAM).draw("mpl")
    return


@app.cell
def _(backend, minimize, np, run_counts):
    def optimize_portfolio(
        p, inits, circuit_fn, util_fn, lam, shots=2048, maxiter=60
    ):
        """COBYLA-tune the 2p knobs to maximize ⟨U⟩ (by minimizing −⟨U⟩) —
        the same loop as Day 4's optimize_qaoa, wearing the new scoreboard.
        Tries each start in `inits`, keeps the best; returns (params, ⟨U⟩).
        Layout: x = [γ₁..γₚ, β₁..βₚ]."""

        def _neg_utility(x):
            counts = run_counts(circuit_fn(x[:p], x[p:], lam), backend, shots)
            total = sum(counts.values())
            return -sum(util_fn(b, lam) * c for b, c in counts.items()) / total

        best_x, best_val = None, -1e9
        for x0 in inits:
            res = minimize(
                _neg_utility,
                np.asarray(x0, float),
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 0.4},
            )
            val = -_neg_utility(res.x)  # fresh estimate at the optimum
            if val > best_val:
                best_x, best_val = res.x, val
        return best_x, best_val

    return (optimize_portfolio,)


@app.cell
def _(
    K,
    LAM,
    backend,
    np,
    optimize_portfolio,
    pf_circuit_is_todo,
    pf_ref,
    portfolio_qaoa,
    portfolio_utility,
    ref_utility,
    run_counts,
    utility_is_todo,
):
    # ── The run: QAOA p = 3 at λ = 1 (~30 s of honest shots) ────────────────
    # Fallback guards: if a Build TODO is still open, the run borrows the
    # reference engine so the fund ALWAYS lands — but it says so, and
    # finishing your build makes the portfolio genuinely yours.
    if pf_circuit_is_todo(portfolio_qaoa):
        _engine = pf_ref
        print("⚠️  Running on the REFERENCE machine — your portfolio_qaoa")
        print("    TODOs aren't in yet (an all-H circuit guesses blindly).")
        print("    Finish Build 2 and rerun to make the fund YOURS.")
    else:
        _engine = portfolio_qaoa
        print("Engine: YOUR portfolio_qaoa. The fund below is genuinely yours.")
    if utility_is_todo(portfolio_utility):
        pf_util = ref_utility
        print("⚠️  Scoring with the reference scorekeeper — your")
        print("    portfolio_utility still returns None (Build 1 TODO).")
    else:
        pf_util = portfolio_utility

    _p = 3
    _ramp = np.array([(_k + 0.5) / _p for _k in range(_p)])
    _inits = [
        np.concatenate([0.7 * _ramp, 0.7 * (1 - _ramp)]),
        np.concatenate([1.6 * _ramp, 0.9 * (1 - _ramp)]),
    ]
    _x, _v = optimize_portfolio(_p, _inits, _engine, pf_util, LAM)
    pf_counts = run_counts(_engine(_x[:_p], _x[_p:], LAM), backend, 4096)

    _feasible = {_b: _c for _b, _c in pf_counts.items() if _b.count("1") == K}
    pf_best = max(
        _feasible or pf_counts,
        key=lambda _b: (pf_util(_b, LAM), pf_counts.get(_b, 0)),
    )
    print(
        f"p = {_p}, knobs tuned: ⟨U⟩ = {_v:+.4f} · best sampled legal "
        f"portfolio: {pf_best} (U = {pf_util(pf_best, LAM):+.3f}) — "
        f"verdict below."
    )
    return pf_best, pf_counts, pf_util


@app.cell(hide_code=True)
def _(LAM, MU, SIGMA, TICKERS, mo, np, pf_best, pf_util):
    _x = np.array([int(_c) for _c in pf_best], float)
    _in = [TICKERS[_i] for _i in range(len(TICKERS)) if pf_best[_i] == "1"]
    _out = [TICKERS[_i] for _i in range(len(TICKERS)) if pf_best[_i] == "0"]
    _ret = float(MU @ _x)
    _risk = float(_x @ SIGMA @ _x)
    mo.md(f"""
    # 📈 Your fund

    <span style="font-size:1.35em; color:#D48F26; font-weight:bold">
    In &nbsp;·&nbsp; {" · ".join(_in)}</span>

    <span style="font-size:1.1em; color:#6B7280">
    Out · {" · ".join(_out)}</span>

    **Expected return {100 * _ret:.0f} % · risk {_risk:.2f} · utility
    {pf_util(pf_best, LAM):+.3f} at λ = {LAM:.1f}.** Picked by the same
    machine you built yesterday — the honesty check below
    says how good it really is.
    """)
    return


@app.cell(hide_code=True)
def _(LAM, MU, SIGMA, TICKERS, draw_market, pf_best, pf_util):
    _chosen = {_i for _i in range(len(TICKERS)) if pf_best[_i] == "1"}
    draw_market(
        TICKERS,
        MU,
        SIGMA,
        chosen=_chosen,
        title=f"your fund · {pf_best} · utility {pf_util(pf_best, LAM):+.3f}",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The honesty check — grading the machine

    Six assets is small enough that a `for` loop scores **all 64
    portfolios** and names the true optimum, so once again we grade the
    quantum machine against certain truth, in public. Two grades this
    time: did it find *the* winner, and did it learn *the mandate*
    (exactly 3 assets) that we never hard-coded — only fined.
    """)
    return


@app.cell(hide_code=True)
def _(K, LAM, N_ASSETS, np, pf_best, pf_counts, pf_util, plt):
    # Brute force: every one of the 64 portfolios, scored by the same
    # scorekeeper that scored the QAOA samples.
    _all = [format(_k, f"0{N_ASSETS}b") for _k in range(2**N_ASSETS)]
    _u = np.array([pf_util(_b, LAM) for _b in _all])
    _umax = float(_u.max())
    _opt = [_b for _b, _v in zip(_all, _u) if _v == _umax]

    _shots = sum(pf_counts.values())
    _p_opt = sum(pf_counts.get(_b, 0) for _b in _opt) / _shots
    _p_feas = (
        sum(_c for _b, _c in pf_counts.items() if _b.count("1") == K) / _shots
    )
    _feas_draws = {_b: _c for _b, _c in pf_counts.items() if _b.count("1") == K}
    _mean_feas = (
        sum(pf_util(_b, LAM) * _c for _b, _c in _feas_draws.items())
        / max(sum(_feas_draws.values()), 1)
    )
    _blind_feas = sum(1 for _b in _all if _b.count("1") == K) / len(_all)

    print(
        f"brute force (all 64): optimum U* = {_umax:+.3f} · "
        f"achieved by {', '.join(_opt)}"
    )
    print(
        f"your best sampled legal portfolio: {pf_best} → "
        f"U = {pf_util(pf_best, LAM):+.3f} "
        f"({pf_util(pf_best, LAM) / _umax:.0%} of optimal)"
    )
    print(
        f"average LEGAL draw: U = {_mean_feas:+.3f} → ratio "
        f"{_mean_feas / _umax:.2f} of optimal"
    )
    print(
        f"P(drawing an optimal portfolio): QAOA {_p_opt:.1%} vs blind "
        f"{len(_opt) / 64:.2%} — a ~{_p_opt / (len(_opt) / 64):.1f}× tilt"
    )
    print(
        f"P(drawing a LEGAL 3-asset portfolio): QAOA {_p_feas:.0%} vs blind "
        f"{_blind_feas:.0%} — the machine learned the mandate from the fine"
    )

    _fig, (_ax1, _ax2) = plt.subplots(
        1, 2, figsize=(11.4, 4.2), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    # left — the leaderboard: top 20 of all 64, ranked by utility
    # (a stated cap — the other 44 exist, they're just worse)
    _TOP = 20
    _order = np.argsort(-_u, kind="stable")[:_TOP]
    _cols = ["#D48F26" if _u[_i] == _umax else "#0E7C86" for _i in _order]
    _ax1.bar(np.arange(_TOP), _u[_order], color=_cols)
    _top = [_all[_i] for _i in _order]
    if pf_best in _top:
        _rank = _top.index(pf_best)
        _ax1.annotate(
            "← your fund",
            (_rank + 0.3, _u[_order[_rank]] - 0.012),
            fontsize=10,
            color="#B23B7B",
            fontweight="bold",
            rotation=90,
            va="top",
        )
    _ax1.axhline(
        float(_u.mean()),
        color="#6B7280",
        ls=":",
        label=f"blind average = {_u.mean():+.2f}",
    )
    _ax1.set_xticks(range(0, _TOP, 2))
    _ax1.set_xlabel(f"rank (top {_TOP} of all 64 portfolios)")
    _ax1.set_ylabel("utility")
    _ax1.set_ylim(-0.2, 0.2)
    _ax1.set_title("the leaderboard · optimal fund in amber")
    _ax1.legend(fontsize=9)
    _ax1.grid(True, axis="y", alpha=0.3)

    # right — did it learn the rules? fixed two-group comparison
    _xpos = np.arange(2)
    _ax2.bar(
        _xpos - 0.18,
        [_p_feas, _p_opt],
        width=0.36,
        color="#0E7C86",
        label="QAOA draws",
    )
    _ax2.bar(
        _xpos + 0.18,
        [_blind_feas, len(_opt) / 64],
        width=0.36,
        color="#6B7280",
        label="blind guessing",
    )
    _ax2.set_xticks(_xpos)
    _ax2.set_xticklabels(["legal (3 assets)", "optimal fund"])
    _ax2.set_ylim(0, 1.05)
    _ax2.set_ylabel("probability per draw")
    _ax2.set_title("what the machine learned")
    _ax2.legend(fontsize=9)
    _ax2.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Stretch check** — before the finale:

    - Did the machine's fund match the amber bar you predicted at
      λ = 1? (In our measured runs the best sampled portfolio hit the
      true optimum — SOLR + H2OX + MEDS — every time.)
    - Read the two grades honestly. The mandate was *learned
      beautifully* (legal draws ~90 %+ vs blind 31 %) — but the tilt
      toward the single optimal fund is modest, ~3–4× blind, humbler
      than yesterday's Max-Cut tilt. Rougher encoding (individual
      scores + fines), lumpier terrain. An honest machine, not a magic
      one — put *that* sentence in your hype filter.
    - The three caveats, out loud: (1) n = 6 brute-forces in
      microseconds — quantum buys nothing here; the bet is n where 2ⁿ
      dies. (2) No optimality certificate — we only knew the grade
      because we *could* enumerate. (3) Real portfolio desks solve the
      *continuous-weights* version classically in milliseconds; the
      QUBO shape is what banks pay to explore, not today's returns.
      "Not replacing classical finance — scouting where it might
      scale" is the honest sentence.
    - **Bonus**: λ is a knob of the *problem*, not the machine. Change
      `LAM` to 0.0 in the market cell and rerun — does your machine
      chase the greed trio? At 2.0, the defensive one?

    (And a promise here too: the same machine meets **the quantum
    chip's own wiring** at the end of this notebook — tuned at toy
    size, deployed past the simulator wall.)

    Two jobs done on a polite simulator. Time to meet an impolite
    machine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3 · Touch a real quantum computer (aspirational)

    Everything this week — Grover, the walker, the Max-Cut machine, the
    molecule, the fund — ran on a simulator that lies politely: perfect
    gates, patient qubits, free shots. This section ends the week in
    **two acts**. Act one sends **one circuit** over the cloud to an
    actual IBM machine — a chip at 15 millikelvin under half a ton of
    dilution fridge, the kind this morning's montage opened with — and
    reads back what reality does to it. Act two goes **past the
    simulator wall**: two 56-qubit jobs, the first work all week that
    a statevector simulator physically cannot check.

    The circuit is a **3-qubit GHZ state** — the week's very first
    trick, scaled by one: H, then a chain of CXs. Ideally it measures
    `000` or `111`, *nothing else, ever* — which is exactly why it's
    the right hello-world for noisy hardware: **every other bitstring
    is a certified error**, no statistics degree required.

    Three ways this section can go, all of them fine:
    - **No IBM account** → the same circuit runs on a **noise model**
      of the real `ibm_brisbane` (its measured error rates, replayed by
      our simulator), clearly labeled as such.
    - **Token pasted** → the circuit goes to the **least-busy real
      device** on your account. Free tier includes enough runtime for
      this many times over (~1 s of QPU).
    - **Queue too long** → you get a job id and pickup code; the
      comparison below falls back to the noise model so nothing blocks.

    **The stakes, measured**: ideal vs noisy differ by a fidelity gap
    of ~7 % on this 3-gate circuit. That gap has a name — you've heard
    it all morning — and closing it is the entire race.
    """)
    return


@app.cell
def _(QuantumCircuit):
    def make_ghz() -> QuantumCircuit:
        """The 3-qubit GHZ circuit: (|000⟩ + |111⟩)/√2, then measure."""
        qc = QuantumCircuit(3)

        # TODO (1): the coin — H on qubit 0.

        # TODO (2): the chain — CX(0, 1), then CX(1, 2): copy the branch
        #           down the line so all three qubits agree in both worlds.

        qc.measure_all()  # (Given.)
        return qc

    return (make_ghz,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Solution — the GHZ circuit (open if stuck)": mo.md("""
    ```python
    # (1) the coin
    qc.h(0)

    # (2) the chain
    qc.cx(0, 1)
    qc.cx(1, 2)
    ```

    Day 2's Bell state plus one link: H splits qubit 0 into both
    worlds, each CX drags the next qubit into whichever world it's in.
    Three qubits, two branches, zero legal disagreements.
    """),
            "🧠 Why is GHZ the right hardware hello-world?": mo.md("""
    It needs the two ingredients that make quantum hardware hard —
    **superposition** (the H) and **entanglement** (the CXs) — yet its
    ideal output is brutally simple: half `000`, half `111`. So every
    shot that reads anything else is an unambiguous, countable failure
    of the machine: a qubit that decayed mid-circuit, a gate that
    slipped, a readout that misfired. No estimation, no priors — just
    *count the impossible*. Hardware teams use exactly this trick (at
    larger sizes) to benchmark devices; you're reading the same gauge
    they do.
    """),
        }
    )
    return


@app.cell(hide_code=True)
def _(QuantumCircuit, np):
    # ── Hidden helpers · Section 3 ──────────────────────────────────────────
    import time

    from qiskit_ibm_runtime.fake_provider import FakeBrisbane

    def ref_ghz():
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        return qc

    def ghz_is_todo(fn):
        try:
            ops = fn().count_ops()
        except Exception:
            return True
        return "h" not in ops or ops.get("cx", 0) < 2

    def hellinger(c1, c2):
        """Hellinger fidelity between two counts dicts (1.0 = identical)."""
        n1, n2 = sum(c1.values()), sum(c2.values())
        keys = set(c1) | set(c2)
        bc = sum(
            np.sqrt(c1.get(k, 0) / n1 * c2.get(k, 0) / n2) for k in keys
        )
        return float(bc**2)

    return FakeBrisbane, ghz_is_todo, hellinger, ref_ghz, time


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Predict, then choose your path

    1. Of the 8 possible 3-bit outcomes, how many are *legal* for an
       ideal GHZ? So if I hand you one noisy histogram, how fast can
       you spot the noise?
    2. Commit: what fraction of 1,024 shots lands on **illegal**
       bitstrings on a real 2025-era device — 0.1 %? 7 %? 30 %?
    3. Day 3's Grover circuit was ~10× deeper. If ~7 % of shots go bad
       in 3 gates, what does that say about running *this week's other
       labs* on today's hardware? (That answer is the NISQ era.)

    Then: paste a token (or don't) and run the cell below. To get a
    token: quantum.cloud.ibm.com → sign up (free) → copy your API key.
    Leave the box empty to use the noise model — the physics lesson is
    identical, and honestly labeled.
    """)
    return


@app.cell
def _(mo):
    ibm_token = mo.ui.text(
        label="IBM Quantum API token (optional — empty = noise model)",
        kind="password",
        full_width=True,
    )
    send_real = mo.ui.run_button(label="🚀 submit to IBM Quantum")
    mo.vstack([ibm_token, send_real])
    return ibm_token, send_real


@app.cell
def _(
    AerSimulator,
    FakeBrisbane,
    backend,
    ghz_is_todo,
    ibm_token,
    make_ghz,
    ref_ghz,
    run_counts,
    send_real,
    time,
    transpile,
):
    # ── The hardware path: real device / noise model, never a dead end ─────
    if ghz_is_todo(make_ghz):
        ghz_used = ref_ghz()
        print("⚠️  Your make_ghz is still all TODOs — every shot would read")
        print("    000 (no coin, no chain, no superposition to break).")
        print("    Borrowing the reference GHZ so the hardware path still")
        print("    flies; finish the build to send YOUR circuit.")
    else:
        ghz_used = make_ghz()
        print("Circuit: YOUR GHZ. What comes back is genuinely yours.")

    ideal_counts = run_counts(ghz_used, backend, 1024)

    hw_counts, hw_label = None, ""
    _token = ibm_token.value.strip()
    if _token and send_real.value:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

            _service = QiskitRuntimeService(
                channel="ibm_quantum_platform", token=_token
            )
            _real = _service.least_busy(operational=True, simulator=False)
            print(
                f"🛰  least-busy real device: {_real.name} "
                f"({_real.num_qubits} qubits)"
            )
            _tqc = transpile(ghz_used, _real, optimization_level=3)
            print(
                f"    transpiled for its hardware: depth {_tqc.depth()}, "
                f"ops {dict(_tqc.count_ops())}"
            )
            _sampler = SamplerV2(mode=_real)
            _job = _sampler.run([_tqc], shots=1024)
            print(f"    job id: {_job.job_id()}")
            try:
                _qpos = _job.metrics().get("position_in_queue")
                if _qpos is not None:
                    print(f"    queue position: {_qpos}")
            except Exception:
                pass
            _deadline = time.time() + 75
            _status = str(_job.status())
            while time.time() < _deadline and _status not in (
                "DONE",
                "ERROR",
                "CANCELLED",
            ):
                time.sleep(5)
                _status = str(_job.status())
            if _status == "DONE":
                _raw = _job.result()[0].data.meas.get_counts()
                hw_counts = {k[::-1]: v for k, v in _raw.items()}
                hw_label = f"REAL device · {_real.name}"
                print("    ✅ real counts retrieved — the comparison below is")
                print("       simulator vs actual superconducting qubits.")
            else:
                print(f"    ⏳ still {_status} after 75 s — the queue owns it")
                print(f"       now, not us. Your job id: {_job.job_id()}")
                print("       Pick it up later with the retrieval fold below;")
                print("       today's comparison uses the noise model instead.")
        except Exception as _err:
            print(f"⚠️  Real-hardware path failed: {_err}")
            print("    (Bad token? No network? No matter —")
            print("    falling back to the noise model so the payoff lands.)")
    elif _token:
        print("🔑 Token pasted — now click '🚀 submit to IBM Quantum' above")
        print("   to actually send it. (Noise model below until then.)")

    if hw_counts is None:
        _fake = FakeBrisbane()
        _noisy = AerSimulator.from_backend(_fake)
        _tqc = transpile(ghz_used, _noisy, optimization_level=1)
        _raw = _noisy.run(_tqc, shots=1024).result().get_counts()
        hw_counts = {k[::-1]: v for k, v in _raw.items()}
        if not hw_label:
            hw_label = "noise MODEL of ibm_brisbane"
        if not (_token and send_real.value):
            print("═" * 68)
            print("🔌 NO TOKEN → this run is a noise MODEL of the real")
            print("   ibm_brisbane (127-qubit Eagle): its calibrated gate")
            print("   errors, decay times and readout errors, replayed by our")
            print("   simulator. To touch the real one, paste your IBM token")
            print("   above and click submit — same circuit, same comparison.")
            print("═" * 68)
        print(
            f"    your {sum(ghz_used.count_ops().values()) - 4} tidy gates "
            f"became: depth {_tqc.depth()}, ops {dict(_tqc.count_ops())}"
        )
        print("    (heavy-hex wiring + native gates only — that's the")
        print("     transpiler paying the hardware's entry fee)")
    return hw_counts, hw_label, ideal_counts


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📦 Pick up a queued job later (code)": mo.md("""
    If your job outlived the 75-second poll, retrieve it any time —
    from any Python anywhere (marimo, Colab, your laptop):

    ```python
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform", token="PASTE_YOUR_TOKEN"
    )
    job = service.job("PASTE_YOUR_JOB_ID")
    print(job.status())
    counts = job.result()[0].data.meas.get_counts()
    ```

    The queue is part of the lesson: real QPUs are shared scientific
    instruments, not serverless functions — this morning's cloud
    landscape, felt personally.
    """),
            "🧠 Where does the ~7 % go? (the NISQ arithmetic)": mo.md("""
    Rough budget for 3 qubits on a Brisbane-class device:

    - **Readout**: ~1–2 % misread per qubit × 3 qubits ≈ 3–5 %
    - **Two-qubit gates**: ~0.5–1 % error × 2 CXs ≈ 1–2 %
    - **Decoherence**: qubits relax during the schedule ≈ another 1–2 %

    Small numbers — but they *compound*, and they compound per gate.
    That's the hype-detection arithmetic from this morning: when a
    vendor quotes "99.9 % gate fidelity", multiply it out over a
    1,000-gate algorithm before applauding (0.999¹⁰⁰⁰ ≈ 37 %). Our
    3-gate circuit loses ~7 %; Day 3's Grover would lose far more;
    Shor-scale circuits lose everything — *unless* errors are
    corrected faster than they accumulate. That's why all four clocks
    this morning were really one clock: the error-correction clock.
    """),
            "🔍 What did the transpiler actually do?": mo.md("""
    Your three textbook gates aren't native to the chip. The
    transpiler (1) **maps** your 3 logical qubits onto 3 physical
    qubits that are actually wired together on the heavy-hex lattice,
    (2) **translates** H and CX into the device's native gate set (RZ,
    SX, X, and a native two-qubit gate — H alone becomes an
    RZ·SX·RZ sandwich), and (3) **optimizes** the result. The printout
    above shows the after: more gates, more depth, same mathematics.
    Every extra native gate is another roll of the error dice — which
    is why transpiler quality is itself a competitive product in this
    morning's cloud landscape.
    """),
        }
    )
    return


@app.cell(hide_code=True)
def _(hellinger, hw_counts, hw_label, ideal_counts, np, plt):
    # ── The payoff: ideal vs reality, side by side ──────────────────────────
    _dom = [format(_k, "03b") for _k in range(8)]
    _n_i = sum(ideal_counts.values())
    _n_h = sum(hw_counts.values())
    _pi = [ideal_counts.get(_b, 0) / _n_i for _b in _dom]
    _ph = [hw_counts.get(_b, 0) / _n_h for _b in _dom]
    _fid = hellinger(ideal_counts, hw_counts)
    _illegal = (
        sum(_v for _b, _v in hw_counts.items() if _b not in ("000", "111"))
        / _n_h
    )

    print(f"ideal simulator vs {hw_label}:")
    print(f"  fidelity (Hellinger) = {_fid:.3f}   (1.000 = indistinguishable)")
    print(
        f"  illegal bitstrings   = {100 * _illegal:.1f} % of shots "
        f"(ideal: exactly 0)"
    )
    print("  Every illegal shot is a certified hardware error — no other")
    print("  explanation exists. THE GAP IS THE NISQ ERA, measured by you.")

    _fig, _ax = plt.subplots(figsize=(9.6, 4.2))
    _xs = np.arange(8)
    _ax.bar(
        _xs - 0.2, _pi, width=0.4, color="#0E7C86", label="ideal simulator"
    )
    _ax.bar(_xs + 0.2, _ph, width=0.4, color="#D48F26", label=hw_label)
    _ax.set_xticks(_xs)
    _ax.set_xticklabels(
        [
            "000\nlegal",
            "001",
            "010",
            "011",
            "100",
            "101",
            "110",
            "111\nlegal",
        ]
    )
    _ax.set_ylim(0, 0.7)
    _ax.set_ylabel("probability")
    _ax.set_title(
        f"GHZ · 1,024 shots each · fidelity {_fid:.3f} · "
        f"illegal {100 * _illegal:.1f} %"
    )
    for _k in range(1, 7):
        if _ph[_k] > 0.004:
            _ax.annotate(
                "err",
                (_k + 0.2, _ph[_k] + 0.012),
                ha="center",
                fontsize=8,
                color="#B23B7B",
                fontweight="bold",
            )
    _ax.annotate(
        "the gap IS the NISQ era",
        (3.5, 0.55),
        ha="center",
        fontsize=12,
        color="#B23B7B",
        fontweight="bold",
    )
    _ax.legend(fontsize=9)
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Aspirational check** — if you're here:

    - Did your illegal-shot prediction hold? (Noise model measures
      ~7 %; real devices vary day to day — calibration data is
      published live, another tab worth bookmarking.)
    - Say the ending honestly: **everything this week ran inside that
      gap.** Grover's needle, the walker's race, the max cut,
      the molecule, the fund — all of it assumed the teal bars. The
      amber bars are what today's machines actually deliver on the
      *easiest circuit we know*. Quantum error correction is the bet
      that buys back the difference, and the four clocks from this
      morning tick at exactly the rate that bet pays off.
    - If you ran on the real device: you are now, permanently, a
      person who has run a program on a quantum computer. The queue
      receipt is the souvenir.

    Warm-up done — the machine said hello back. Now hand it something
    no laptop can even *hold*: **the wall is next.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Past the simulator wall — two jobs at 56 qubits (the finale)

    The GHZ was reconnaissance: 3 qubits any laptop can shadow. Now the
    week earns this morning's decision rule (slide C5) the hard way. A
    statevector simulator stores **2ᴺ amplitudes**: at 40 qubits that's
    ~16 TB — the RAM cliff from the slides — and at **56 qubits it's
    about an exabyte**: a warehouse of a million laptops, for *one*
    circuit's state. Nobody brute-forces that. Whatever numbers come
    back below did **not** come from an amplitude array — this is the
    first work all week that *necessitates* something better.

    Two runs, both reusing machinery you already own:

    - **Run A — the π electrons of a 56-carbon molecule.** Section 1's
      VQE loop, pointed at a conjugated polymer 28× wider than H₂.
    - **Run B — Max-Cut on the quantum chip's own wiring.** Yesterday's
      QAOA (and Section 2's suit-wearing twin), pointed at the
      machine's own coupling map: split the chip against itself,
      graded against a known-perfect answer.

    Both lean on the same research-grade trick: **tune on the toy,
    deploy on the beast.** The knobs depend only on the problem's
    *local* shape — never on its size — so you tune them where it's
    cheap (an 8-qubit simulation; even a pencil-and-paper formula) and
    transfer them, unchanged, across the wall.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    > **⚠️ The referee that survives the wall — read this honestly.**
    > A statevector simulator died at ~40 qubits, but the two circuits
    > below are *shallow* and live on thin, sparse graphs (a chain; a
    > slice of heavy-hex), so they build up very little entanglement —
    > and a **tensor-network simulator**
    > (`AerSimulator(method="matrix_product_state")`) can still fake
    > them, in well under a second, by storing only the entanglement
    > the state actually has instead of all 2⁵⁶ amplitudes. The wall
    > is **entanglement-shaped, not qubit-count-shaped**. Make these
    > circuits deeper, or their graphs denser, and this trick dies
    > too — while the real machine doesn't care either way. Below,
    > MPS plays *referee and fallback*; paste a token and the very
    > same circuits go to actual hardware.
    """)
    return


@app.cell
def _(ibm_token, mo):
    send_wall = mo.ui.run_button(
        label="🚀 submit both 56-qubit runs to IBM Quantum"
    )
    mo.vstack(
        [
            mo.md(
                "Same token box as the GHZ run — it stays in sync, so if "
                "you pasted a token up there it's already here. Leave "
                "empty for the MPS referee only."
            ),
            ibm_token,
            send_wall,
        ]
    )
    return (send_wall,)


@app.cell(hide_code=True)
def _(AerSimulator, FakeBrisbane, QuantumCircuit, np, time, transpile):
    # ── Hidden engines · past the wall ──────────────────────────────────────
    # Run A model: Hückel (tight-binding) π chain, H = −t Σ (c†ᵢcᵢ₊₁ + h.c.),
    # open chain, t = 1. Under Jordan–Wigner this is the qubit Hamiltonian
    #     H = −(t/2) Σᵢ (XᵢXᵢ₊₁ + YᵢYᵢ₊₁).
    # Circuit design (Givens rotations as the number-conserving building
    # block) borrowed from PennyLane's Givens-rotations tutorial:
    #     https://pennylane.ai/qml/demos/tutorial_givens_rotations
    # ("like Lego blocks, they can be used to construct any kind of
    #  particle-conserving circuit").
    from qiskit.circuit.library import XXPlusYYGate
    from qiskit.quantum_info import SparsePauliOp, Statevector

    N_TOY, N_WALL = 8, 56
    T_HOP = 1.0

    def pi_chain_ansatz(n, thetas):
        """Givens-rotation ansatz for the π chain, translation-uniform.
        Start |0101…⟩ (π electrons parked on alternating carbons — zero
        bonds), then per layer: Givens(θ_even) on even bonds, Givens(θ_odd)
        on odd bonds. thetas = (e₁, o₁, e₂, o₂, …), 2 knobs per layer —
        and no knob ever mentions n, which is what makes them transfer.
        NOTE the beta=π/2 on XXPlusYYGate: it makes the hopping REAL.
        With beta=0 (or plain RXX+RYY layers) the hopping picks up a
        phase and every measured ⟨XX⟩+⟨YY⟩ reads zero — we hit this
        while authoring; the 🔍 fold has the autopsy."""
        qc = QuantumCircuit(n)
        for i in range(1, n, 2):
            qc.x(i)
        p = len(thetas) // 2
        for k in range(p):
            t_e, t_o = thetas[2 * k], thetas[2 * k + 1]
            for i in range(0, n - 1, 2):
                qc.append(XXPlusYYGate(t_e, beta=np.pi / 2), [i, i + 1])
            for i in range(1, n - 1, 2):
                qc.append(XXPlusYYGate(t_o, beta=np.pi / 2), [i, i + 1])
        return qc

    def pi_qubit_op(n):
        """The JW qubit Hamiltonian as a SparsePauliOp — for exact work at
        small n only (at n = 56 its matrix would have 2⁵⁶ rows)."""
        terms = []
        for i in range(n - 1):
            for pauli in ("X", "Y"):
                s = ["I"] * n
                s[i] = pauli
                s[i + 1] = pauli
                terms.append(("".join(reversed(s)), -T_HOP / 2))
        return SparsePauliOp.from_list(terms)

    def pi_energy_from_counts(cx, cy, n):
        """⟨H⟩ from exactly TWO measured settings: the all-X counts serve
        every ⟨XᵢXᵢ₊₁⟩ bond at once, the all-Y counts every ⟨YᵢYᵢ₊₁⟩.
        2·(n−1) = 110 Hamiltonian terms at n = 56 — two settings."""
        tx, ty = sum(cx.values()), sum(cy.values())
        e = 0.0
        for bits, c in cx.items():
            for i in range(n - 1):
                e += -(T_HOP / 2) * (1 if bits[i] == bits[i + 1] else -1) * c / tx
        for bits, c in cy.items():
            for i in range(n - 1):
                e += -(T_HOP / 2) * (1 if bits[i] == bits[i + 1] else -1) * c / ty
        return e

    def huckel_exact(n):
        """Chemistry's referee, unchanged since 1931: diagonalize the n×n
        Hückel hopping matrix and fill the lowest n/2 molecular orbitals.
        An n×n eigensolve instead of a 2ⁿ×2ⁿ one — the classical backdoor
        THIS molecule has and correlated molecules don't (🧠 fold)."""
        M = np.zeros((n, n))
        for i in range(n - 1):
            M[i, i + 1] = M[i + 1, i] = -T_HOP
        ev = np.linalg.eigvalsh(M)
        return float(ev[: n // 2].sum())

    # ── Run B engine · the chip's own wiring ────────────────────────────────
    # The Max-Cut instance is read off the quantum computer itself: the
    # 56-qubit BFS-induced subgraph of the heavy-hex coupling map
    # (FakeBrisbane's map by default; on the token path the LIVE selected
    # backend's own map, so "the machine's wiring" stays literally true).
    def chip_graph(chip_backend, n_take):
        """BFS from physical qubit 0 across a backend's coupling map,
        collecting the first n_take qubits reached. Returns (phys, edges,
        deg): physical-qubit ids in BFS order (which doubles as the
        transpiler's initial_layout), the induced couplers as logical
        (a, b) pairs, and each logical node's degree."""
        phys_edges = {
            tuple(sorted(e)) for e in chip_backend.coupling_map.get_edges()
        }
        adjacency = {}
        for a, b in phys_edges:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        order, seen, queue = [], {0}, [0]
        while queue and len(order) < n_take:
            v = queue.pop(0)
            order.append(v)
            for w in sorted(adjacency[v]):
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
        order = order[:n_take]
        index = {p: i for i, p in enumerate(order)}
        edges = sorted(
            (min(index[a], index[b]), max(index[a], index[b]))
            for a, b in phys_edges
            if a in index and b in index
        )
        deg = [0] * n_take
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        return order, edges, deg

    def two_coloring(n, edges):
        """BFS 2-coloring; returns the color list, or None if some edge
        ends up monochrome (i.e. the graph is not bipartite)."""
        color = {0: 0}
        neigh = {}
        for a, b in edges:
            neigh.setdefault(a, set()).add(b)
            neigh.setdefault(b, set()).add(a)
        queue = [0]
        while queue:
            v = queue.pop(0)
            for w in neigh.get(v, ()):
                if w not in color:
                    color[w] = 1 - color[v]
                    queue.append(w)
                elif color[w] == color[v]:
                    return None
        return [color.get(i, 0) for i in range(n)]

    _fb = FakeBrisbane()
    WALL_PHYS, WALL_EDGES, WALL_DEG = chip_graph(_fb, N_WALL)
    PATCH_PHYS, PATCH_EDGES, PATCH_DEG = chip_graph(_fb, 14)
    WALL_COLOR = two_coloring(N_WALL, WALL_EDGES)

    def cut_of(bits, edges):
        """Cut value of a coloring: how many edges have ends that disagree."""
        return sum(1 for a, b in edges if bits[a] != bits[b])

    def maxcut_p1_expect(edges, deg, gamma, beta):
        """The known CLOSED FORM for p = 1 QAOA's ⟨cut⟩ on triangle-free
        graphs (Wang, Hadfield, Jiang & Rieffel 2018, arXiv:1706.02998):
        each edge (a, b) contributes
            ½ + ¼·sin4β·sinγ·(cos^(d_a−1)γ + cos^(d_b−1)γ).
        Pure math — no simulator, no shots, no quantum computer."""
        s = 0.0
        for a, b in edges:
            s += 0.5 + 0.25 * np.sin(4 * beta) * np.sin(gamma) * (
                np.cos(gamma) ** (deg[a] - 1) + np.cos(gamma) ** (deg[b] - 1)
            )
        return s

    def chip_qaoa_ref(n, edges, gamma, beta, measured=True):
        """p = 1 Max-Cut QAOA on the given edges. NOTE the rzz(−γ):
        qiskit's RZZ(θ) = e^(−iθ·ZZ/2), so matching the cost convention
        e^(−iγ(1−ZᵢZⱼ)/2) behind the analytic formula takes a MINUS sign
        (the 🧠 fold has the autopsy of what happens without it)."""
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for a, b in edges:
            qc.rzz(-gamma, a, b)
        for q in range(n):
            qc.rx(2 * beta, q)
        if measured:
            qc.measure_all()
        return qc

    # The tensor-network referee (see the banner above for what it can —
    # and, more importantly, cannot — fake).
    mps_backend = AerSimulator(method="matrix_product_state")

    def wall_pick_backend(token):
        """The least-busy ≥56-qubit real device on the account — or None,
        with the reason printed (never a dead end)."""
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            service = QiskitRuntimeService(
                channel="ibm_quantum_platform", token=token
            )
            real = service.least_busy(
                operational=True, simulator=False, min_num_qubits=N_WALL
            )
            print(
                f"🛰  least-busy ≥{N_WALL}-qubit device: {real.name} "
                f"({real.num_qubits} qubits)"
            )
            return real
        except Exception as err:
            print(f"⚠️  Could not reach a real device: {err}")
            print("    (Bad token? No ≥56-qubit device on this account? No")
            print("     matter — the MPS referee carries the comparison.)")
            return None

    def wall_submit(circuits, real, shots=2048, initial_layout=None):
        """Batch-submit measured circuits to a real device (ONE SamplerV2
        job), poll ≤75 s, never block: returns a list of bit-order counts
        dicts, or None. Same job-id / poll / walk-away plumbing as the
        GHZ cell."""
        try:
            from qiskit_ibm_runtime import SamplerV2

            tqcs = [
                transpile(
                    qc,
                    real,
                    optimization_level=3,
                    initial_layout=initial_layout,
                )
                for qc in circuits
            ]
            for k, tqc in enumerate(tqcs):
                print(
                    f"    circuit {k}: transpiled depth {tqc.depth()}, "
                    f"{sum(tqc.count_ops().values())} ops"
                )
            job = SamplerV2(mode=real).run(tqcs, shots=shots)
            print(
                f"    job id: {job.job_id()} · {len(tqcs)} circuit(s) "
                f"batched, {shots} shots each"
            )
            deadline = time.time() + 75
            status = str(job.status())
            while time.time() < deadline and status not in (
                "DONE",
                "ERROR",
                "CANCELLED",
            ):
                time.sleep(5)
                status = str(job.status())
            if status == "DONE":
                out = []
                for r in job.result():
                    raw = r.data.meas.get_counts()
                    out.append({k[::-1]: v for k, v in raw.items()})
                print("    ✅ real 56-qubit counts retrieved.")
                return out
            print(f"    ⏳ still {status} after 75 s — the queue owns it")
            print(f"       now, not us. Job id: {job.job_id()} — pick it up")
            print("       later with the retrieval fold in the GHZ section.")
            print("       The MPS referee carries today's comparison.")
            return None
        except Exception as err:
            print(f"⚠️  Hardware path failed: {err}")
            print("    (No matter — the MPS referee carries the comparison.)")
            return None

    return (
        N_TOY,
        N_WALL,
        PATCH_DEG,
        PATCH_EDGES,
        Statevector,
        WALL_COLOR,
        WALL_DEG,
        WALL_EDGES,
        WALL_PHYS,
        chip_graph,
        chip_qaoa_ref,
        cut_of,
        huckel_exact,
        maxcut_p1_expect,
        mps_backend,
        pi_chain_ansatz,
        pi_energy_from_counts,
        pi_qubit_op,
        two_coloring,
        wall_pick_backend,
        wall_submit,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Run A — the π electrons of a 56-carbon molecule (VQE past the wall)

    Meet **polyacetylene**: a chain of carbons, one π electron each —
    the molecule whose conductive form won the **2000 Nobel Prize in
    Chemistry**. Section 1's H₂ was *full* quantum chemistry on 2
    qubits; this is chemistry's beloved *simplified* model — **Hückel
    theory**, the pencil-and-paper method every chemist since 1931 has
    used for benzene and polyenes — on **56 qubits**, one per carbon's
    p_z orbital. Qubit i answers: "is a π electron on carbon i?"

    The physics in one line: each π electron can **hop** to a
    neighboring carbon (strength t — chemists call it β, worth a couple
    of eV). Park the electrons on alternating carbons — the state
    |0101…⟩ — and the energy is **exactly zero**: no sharing, no π
    bonds, no molecule to speak of. Every joule below zero is
    **delocalization (resonance) energy** — electrons lowering their
    energy by *spreading out in superposition* across carbons. The
    same sliver Section 1 fought for as "correlation energy" is here
    the *entire* prize: whatever your circuit measures below zero *is*
    the chemistry.

    The plan is Section 1's loop with a scale-crossing twist:

    1. **Train small** (given, below): a 4-knob ansatz at N = 8, where
       the statevector simulator still hands out exact expectations.
       The moves are **Givens rotations** — the number-conserving
       building block of real quantum-chemistry circuits ("like Lego
       blocks", says PennyLane's Givens-rotations tutorial, our
       circuit-design source — see the engine cell for the URL).
    2. **Transfer** (your build): the knobs never mention the chain
       length — apply the *same four numbers* at **N = 56**.
    3. **Measure cheap**: the 56-carbon Hamiltonian has **110 terms**,
       yet every ⟨XᵢXᵢ₊₁⟩ reads off ONE all-X run and every ⟨YᵢYᵢ₊₁⟩
       off ONE all-Y run. **Two measurement settings.** (Section 1's
       meter, industrial edition.)
    4. **Grade honestly**: Hückel's referee is a 56×56 eigensolve —
       *this* molecule has a classical backdoor, and that is the only
       reason we can grade the machine today. Add electron–electron
       correlation (real chemistry) and the backdoor slams shut —
       that's the regime the machine is actually *for* (🧠 fold).
    """)
    return


@app.cell(hide_code=True)
def _(plt):
    # The problem instance, drawn before any circuit: the molecule itself.
    _fig, _ax = plt.subplots(figsize=(10.4, 2.7))
    _show = 12
    _xs = [0.5 * _i for _i in range(_show)]
    _ys = [0.18 if _i % 2 else -0.18 for _i in range(_show)]
    for _i in range(_show - 1):
        _ax.plot(
            [_xs[_i], _xs[_i + 1]],
            [_ys[_i], _ys[_i + 1]],
            color="#1b2a4a",
            lw=1.8,
            zorder=1,
        )
        if _i % 2 == 0:  # the alternating double bond of the cartoon
            _ax.plot(
                [_xs[_i] + 0.04, _xs[_i + 1] + 0.04],
                [_ys[_i] + 0.055, _ys[_i + 1] + 0.055],
                color="#1b2a4a",
                lw=1.8,
                zorder=1,
            )
    _ax.scatter(_xs, _ys, s=240, c="#0E7C86", zorder=2)
    for _i in range(_show):
        _ax.annotate(
            f"C{_i}",
            (_xs[_i], _ys[_i]),
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
            zorder=3,
        )
    _ax.annotate(
        "· · · on to C55",
        (_xs[-1] + 0.28, 0.0),
        fontsize=11,
        color="#1b2a4a",
        va="center",
    )
    _ax.annotate(
        "one qubit per carbon · qubit i answers “is a π electron on "
        "carbon i?” · |0101…⟩ = all parked, E = 0",
        (0.0, -0.46),
        fontsize=9,
        color="#6B7280",
    )
    _ax.set_xlim(-0.4, 7.8)
    _ax.set_ylim(-0.6, 0.5)
    _ax.axis("off")
    _ax.set_title(
        "the instance · a 56-carbon conjugated chain "
        "(polyacetylene fragment, Hückel model)",
        fontsize=11,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    N_TOY,
    Statevector,
    huckel_exact,
    minimize,
    np,
    pi_chain_ansatz,
    pi_qubit_op,
):
    # ── Run A step 1 · train the 4 knobs at N = 8 (below the wall) ──────────
    # Below the wall the statevector simulator hands out EXACT expectation
    # values — use them while they exist. Two fixed COBYLA starts keep this
    # quick and reproducible (the second is a pre-found good region).
    _op = pi_qubit_op(N_TOY)

    def _e_toy(params):
        _sv = Statevector(pi_chain_ansatz(N_TOY, params))
        return float(_sv.expectation_value(_op).real)

    _best = None
    for _x0 in ([0.8, 0.8, 0.5, 0.5], [1.0, 0.7, 0.6, 0.4]):
        _res = minimize(
            _e_toy,
            np.array(_x0),
            method="COBYLA",
            options={"maxiter": 150, "rhobeg": 0.5},
        )
        if _best is None or _res.fun < _best.fun:
            _best = _res
    poly_thetas = tuple(float(_v) for _v in _best.x)
    e_toy_trained = float(_best.fun)

    # The referee, cross-examined: chemistry's 1931 shortcut (fill the
    # lowest N/2 Hückel orbitals) vs the honest 2⁸×2⁸ diagonalization.
    _e_ed = float(np.linalg.eigvalsh(_op.to_matrix()).min())
    _e_hk = huckel_exact(N_TOY)
    print(f"N = 8 trained (p = 2, four Givens knobs): E = {e_toy_trained:+.4f}")
    print(
        f"N = 8 exact (2⁸×2⁸ diagonalization):      E = {_e_ed:+.4f}"
        f"  → ansatz reaches {e_toy_trained / _e_ed:.1%} of it"
    )
    print(
        f"N = 8 Hückel referee (an 8×8 eigensolve): E = {_e_hk:+.4f}"
        f"  (|ED − Hückel| = {abs(_e_ed - _e_hk):.1e} — same physics,"
    )
    print("                                          8 rows instead of 256)")
    print(f"knobs: θ = ({', '.join(f'{_t:+.3f}' for _t in poly_thetas)})")
    print("Four numbers — and none of them mentions '8'. Hold that thought.")
    return e_toy_trained, poly_thetas


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🧠 Chemistry's classical backdoor — and why it's THE honest caveat": mo.md(r"""
    Hückel theory ignores electron–electron repulsion, so the
    56-carbon problem separates into **independent one-electron
    orbitals**: diagonalize the $56\times 56$ hopping matrix
    ($M_{i,i+1} = M_{i+1,i} = -t$), fill the lowest $N/2$ orbitals,
    sum. The printout above cross-examines this at $N = 8$: the
    orbital-filling answer matches the honest $2^8\times 2^8$
    diagonalization of the qubit Hamiltonian to ~$10^{-15}$ — same
    physics, 8 rows instead of 256. At $N = 56$ the honest matrix
    would have $2^{56}$ rows; the backdoor still costs microseconds.

    Say the caveat like a scientist: **this molecule is gradeable
    precisely because it's easy.** The moment you restore what Hückel
    drops — electrons repelling each other, the *correlation* that
    Section 1 fought for on 2 qubits — the problem stops separating,
    the backdoor slams shut, and no classical referee exists at 56
    orbitals. That's not a flaw in today's exercise; it's the exact
    shape of the industry's bet: quantum machines earn their keep
    where the referee *can't* follow. (A chemistry footnote: our chain
    is spinless — one electron per orbital. A chemist's polyene holds
    **two** electrons per filled MO, so their tabulated
    $E_\pi = 2\times$ our qubit value. Same orbitals, double
    occupancy.)
    """),
            "🔍 The given engine — Givens rotations, and the β = π/2 landmine": mo.md(r"""
    The circuit (in the hidden engine cell) is three moves:

    ```python
    for i in range(1, n, 2):      # |0101…⟩ — park the electrons
        qc.x(i)
    for i in range(0, n - 1, 2):  # even bonds, one shared knob
        qc.append(XXPlusYYGate(theta_e, beta=np.pi / 2), [i, i + 1])
    for i in range(1, n - 1, 2):  # odd bonds, one shared knob
        qc.append(XXPlusYYGate(theta_o, beta=np.pi / 2), [i, i + 1])
    # (× 2 layers)
    ```

    `XXPlusYYGate` is a **Givens rotation**: it rotates only in the
    span of |01⟩ and |10⟩ — an electron *hopping* between two
    orbitals — and never creates or destroys electrons. PennyLane's
    Givens-rotations tutorial (our design source:
    https://pennylane.ai/qml/demos/tutorial_givens_rotations) calls
    these the Lego blocks of particle-conserving circuits, and real
    quantum-chemistry ansätze are stacked out of exactly them. Your
    circuit always holds exactly 28 electrons — like Section 1's
    ansatz never leaving its 2-electron sector, at 28× the width.

    **The landmine we stepped on so you don't**: with `beta=0` — or a
    hand-rolled `rxx(θ); ryy(θ)` pair — the hop happens but picks up
    an **imaginary phase** (|01⟩ → i·|10⟩-ish), and the measured
    ⟨XX⟩+⟨YY⟩ of every bond reads **exactly zero**: the state moves,
    the meter stays dead. `beta=π/2` makes the hopping amplitude
    real, phases aligned with the Hamiltonian we're scoring. One
    keyword argument between "94 % of the resonance energy" and
    "zero, identically" — phase bookkeeping, the week's oldest lesson,
    still collecting rent on day 5.
    """),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Build — the transfer (one honest line)

    Cross the wall. `poly_thetas` holds the four knobs trained at
    N = 8; the ansatz is translation-uniform, so the *same numbers*
    drive the 56-carbon circuit. Set the TODO — and notice that the
    build being a one-liner **is the finding**: parameters tuned on a
    toy are the deployment artifact. Nothing about them needs
    re-tuning at 7× the size (Run B makes the same point on a graph,
    and cites the research phenomenon by name).
    """)
    return


@app.cell
def _():
    # The four knobs trained at N = 8 are sitting in `poly_thetas`,
    # and the ansatz never asks how long the chain is.
    # TODO — send the SAME knobs across the wall:
    wall_thetas = None  # ← replace None with: poly_thetas
    return (wall_thetas,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Solution — the transfer (open if stuck)": mo.md("""
    ```python
    wall_thetas = poly_thetas
    ```

    That's the whole build. If it feels like cheating, that's the
    lesson: a translation-uniform ansatz has knobs that are properties
    of the *local physics*, not of the system size — so the expensive
    part (training) happens where simulators are cheap, and the scarce
    part (the real machine, this morning's per-shot pricing) only ever
    runs final circuits. This is a real deployment pattern for NISQ
    hardware, not a classroom trick.
    """),
        }
    )
    return


@app.cell
def _(
    N_TOY,
    N_WALL,
    e_toy_trained,
    huckel_exact,
    ibm_token,
    mps_backend,
    pi_chain_ansatz,
    pi_energy_from_counts,
    plt,
    poly_thetas,
    run_counts,
    send_wall,
    wall_pick_backend,
    wall_submit,
    wall_thetas,
):
    # ── Run A · the wall crossing ────────────────────────────────────────────
    if wall_thetas is None:
        _thetas = poly_thetas
        print("⚠️  The transfer TODO is still None — borrowing the trained")
        print("    knobs so the crossing still happens. The build is one")
        print("    honest line; write it and these numbers become YOURS.")
    else:
        _thetas = wall_thetas
        print("Knobs: YOUR transfer. The 56-carbon energy below is yours.")

    _qx = pi_chain_ansatz(N_WALL, _thetas)
    _qx.h(range(N_WALL))  # X-basis: every ⟨XᵢXᵢ₊₁⟩ from one run
    _qx.measure_all()
    _qy = pi_chain_ansatz(N_WALL, _thetas)
    _qy.sdg(range(N_WALL))  # Y-basis: S† then H — Section 1's move, plus one
    _qy.h(range(N_WALL))
    _qy.measure_all()

    # Referee/fallback: the tensor-network simulator (see the banner).
    _cx = run_counts(_qx, mps_backend, 2048)
    _cy = run_counts(_qy, mps_backend, 2048)
    _e_wall_mps = pi_energy_from_counts(_cx, _cy, N_WALL)

    _e_wall_hw, _hw_name = None, None
    _tok = ibm_token.value.strip()
    if _tok and send_wall.value:
        _real = wall_pick_backend(_tok)
        if _real is not None:
            _hw = wall_submit([_qx, _qy], _real, shots=2048)
            if _hw is not None:
                _e_wall_hw = pi_energy_from_counts(_hw[0], _hw[1], N_WALL)
                _hw_name = _real.name

    _e_exact = huckel_exact(N_WALL)
    print()
    print("E(|0101…⟩ baseline — π bonds off)  =   +0.00")
    print(
        f"E(transferred knobs, MPS referee)  = {_e_wall_mps:+8.2f}"
        "   ← measured: 2 settings × 2,048 shots, 110 terms"
    )
    if _e_wall_hw is not None:
        print(
            f"E(transferred knobs, {_hw_name})  = {_e_wall_hw:+8.2f}"
            "   ← REAL hardware"
        )
    print(
        f"E(Hückel exact, 56×56 referee)     = {_e_exact:+8.2f}"
        "   ← chemistry's 1931 backdoor"
    )
    print(
        f"→ {_e_wall_mps / _e_exact:.0%} of the π delocalization energy, "
        "on four knobs trained at N = 8"
    )
    print("  and never re-tuned.")
    if _e_wall_hw is not None:
        print(
            f"→ hardware sits {_e_wall_hw - _e_wall_mps:+.2f} above the MPS "
            "referee — that gap is the noise"
        )
        print("  floor you met in the GHZ run, now taxed at every gate of a")
        print("  56-qubit circuit.")

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for _ax, _n, _pts in (
        (_axes[0], N_TOY, [(e_toy_trained, "#0E7C86", "trained here (exact sim)")]),
        (
            _axes[1],
            N_WALL,
            [(_e_wall_mps, "#0E7C86", "transferred knobs (MPS, measured)")]
            + (
                [(_e_wall_hw, "#D48F26", f"REAL device · {_hw_name}")]
                if _e_wall_hw is not None
                else []
            ),
        ),
    ):
        _ex = huckel_exact(_n)
        _ax.axhline(
            0.0,
            color="#D48F26",
            ls="--",
            lw=2,
            label="|0101…⟩ — π electrons parked (E = 0)",
        )
        _ax.axhline(_ex, color="#1b2a4a", ls=":", lw=2, label="Hückel exact")
        for _k, (_ev, _col, _lab) in enumerate(_pts):
            _ax.scatter(
                [0.42 + 0.22 * _k],
                [_ev],
                s=170,
                facecolor=_col,
                edgecolor="white",
                lw=1.6,
                zorder=3,
                label=_lab,
            )
        _ax.annotate(
            "",
            xy=(0.9, _ex),
            xytext=(0.9, 0.0),
            arrowprops=dict(arrowstyle="<->", color="#B23B7B", lw=1.6),
        )
        _ax.annotate(
            f"π delocalization\nenergy = {abs(_ex):.2f} t",
            (0.92, 0.55 * _ex),
            fontsize=8.5,
            color="#B23B7B",
        )
        _ax.set_xlim(0, 1.25)
        _ax.set_xticks([])
        _ax.set_ylim(1.14 * _ex, -0.16 * _ex)
        _ax.set_ylabel("energy (units of t)")
        _ax.set_title(
            f"N = {_n} carbons" + ("" if _n == N_TOY else " — past the wall")
        )
        _ax.legend(loc="upper left", fontsize=8)
        _ax.grid(True, axis="y", alpha=0.3)
    _fig.suptitle("the same four knobs, on both sides of the wall")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Run B — split the quantum chip against itself (QAOA past the wall)

    Yesterday's machine gets the same treatment — and this time **the
    problem instance is the machine**. The figure below is read
    straight off the backend object (`backend.coupling_map`): the
    56-qubit patch of the **heavy-hex lattice** you met on this
    morning's transpilation slide. The task: *partition the chip's 56
    qubits into two control groups so that as many physical couplers
    as possible run between the groups* — Max-Cut on the chip's own
    wiring. That's the shape of real chip-engineering problems
    (crosstalk and frequency-collision partitioning), and it's exactly
    the graph family hardware-native QAOA benchmarks use — because
    every RZZ lands on a *physical coupler*, so the transpiler has no
    routing to do (we'll measure that claim in gates, below).

    Three referees this run, all exact — and none is a big simulator:

    - **A pencil-and-paper formula.** Heavy-hex has no triangles
      (verified below), and for triangle-free graphs p = 1 QAOA's
      ⟨cut⟩ has a known **closed form** (Wang et al. 2018). Tuning
      (γ, β) becomes a pure-math grid search: no simulator, no shots,
      no quantum computer. Sit with that: for shallow QAOA, classical
      math runs *ahead* of the machine.
    - **The perfect answer is known.** Heavy-hex is **bipartite**
      (verified below by 2-coloring): color the qubits alternately
      and every one of the **59** couplers is cut. We know the
      optimum exactly — watch how close p = 1 gets.
    - **The MPS referee** actually samples the 56-qubit circuit, and
      its ⟨cut⟩ should land on the formula's prediction to within
      shot noise — a three-way agreement worth witnessing.

    The arc: tune on a **14-qubit patch** of the same lattice (by
    formula), transfer the two angles to all 56, sample, grade — the
    transfer works for the same local-structure reason as Run A, and
    it has a research name (🧠 fold). (Honesty paren: a 59-edge
    bipartite Max-Cut is classically trivial — today's demonstration
    is *hardware-native deployment plus an honestly graded heuristic*,
    not advantage.)
    """)
    return


@app.cell(hide_code=True)
def _(WALL_COLOR, WALL_EDGES, WALL_PHYS, plt):
    # The problem instance, drawn before any circuit: the machine's own
    # wiring. Coordinates are derived from the heavy-hex structure itself
    # (long rows = runs of consecutive physical ids; connector rungs hang
    # between them), so this is the chip as IBM draws it.
    _ids = sorted(WALL_PHYS)
    _eset = {
        (min(WALL_PHYS[_a], WALL_PHYS[_b]), max(WALL_PHYS[_a], WALL_PHYS[_b]))
        for _a, _b in WALL_EDGES
    }
    _rows, _cur = [], [_ids[0]]
    for _p in _ids[1:]:
        if _p == _cur[-1] + 1 and (_cur[-1], _p) in _eset:
            _cur.append(_p)
        else:
            _rows.append(_cur)
            _cur = [_p]
    _rows.append(_cur)
    _rowof = {}
    for _r, _row in enumerate(_rows):
        for _p in _row:
            _rowof[_p] = _r
    _X, _Y = {}, {}
    for _i, _p in enumerate(_rows[0]):
        _X[_p], _Y[_p] = float(_i), 0.0
    _placed = {0}
    while len(_placed) < len(_rows):
        _prog = False
        for _a, _b in sorted(_eset):
            if _rowof[_a] == _rowof[_b]:
                continue
            for _u, _v in ((_a, _b), (_b, _a)):
                if _rowof[_u] in _placed and _rowof[_v] not in _placed:
                    _row = _rows[_rowof[_v]]
                    _off = _X[_u] - (_v - _row[0])
                    for _i, _p in enumerate(_row):
                        _X[_p] = _off + _i
                        _Y[_p] = _Y[_u] + (1.0 if _v > _u else -1.0)
                    _placed.add(_rowof[_v])
                    _prog = True
        if not _prog:
            break

    _fig, _ax = plt.subplots(figsize=(10.6, 4.4))
    for _a, _b in WALL_EDGES:
        _pa, _pb = WALL_PHYS[_a], WALL_PHYS[_b]
        _ax.plot(
            [_X[_pa], _X[_pb]],
            [-_Y[_pa], -_Y[_pb]],
            color="#9a9a9a",
            lw=1.4,
            zorder=1,
        )
    for _i, _p in enumerate(WALL_PHYS):
        _ax.scatter(
            [_X[_p]],
            [-_Y[_p]],
            s=170,
            c="#0E7C86" if (WALL_COLOR or [0] * 56)[_i] == 0 else "#D48F26",
            zorder=2,
        )
        _ax.annotate(
            str(_p),
            (_X[_p], -_Y[_p]),
            ha="center",
            va="center",
            fontsize=5.6,
            color="white",
            fontweight="bold",
            zorder=3,
        )
    _ax.axis("off")
    _ax.set_aspect("equal")
    _ax.set_title(
        "the instance · 56 qubits of ibm_brisbane's own coupling map "
        "(labels = physical qubits) ·\nteal/amber = the perfect 2-coloring "
        "— all 59 couplers cut, our known-best referee",
        fontsize=10,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    PATCH_DEG,
    PATCH_EDGES,
    Statevector,
    WALL_COLOR,
    WALL_EDGES,
    chip_qaoa_ref,
    cut_of,
    maxcut_p1_expect,
    np,
    plt,
):
    # ── Run B step 1 · check the referees, then tune BY PURE MATH ──────────
    # The formula's precondition (no triangles) and the perfect-answer
    # claim (bipartite) are VERIFIED here, not assumed:
    _nb = {}
    for _a, _b in WALL_EDGES:
        _nb.setdefault(_a, set()).add(_b)
        _nb.setdefault(_b, set()).add(_a)
    _tri_free = all(not (_nb[_a] & _nb[_b]) for _a, _b in WALL_EDGES)
    print(
        f"chip graph: 56 nodes, {len(WALL_EDGES)} couplers · "
        f"triangle-free: {_tri_free} (the formula's precondition) · "
        f"bipartite: {WALL_COLOR is not None} → true max cut = "
        f"{len(WALL_EDGES)}"
    )

    # Tune (γ, β) on the 14-qubit patch — with the CLOSED FORM. Note what
    # is not here: no backend, no shots, no Statevector. Just the formula.
    _gs = np.linspace(0.02, np.pi, 60)
    _bs = np.linspace(0.02, np.pi / 4, 30)
    _grid = np.array(
        [
            [maxcut_p1_expect(PATCH_EDGES, PATCH_DEG, _gv, _bv) for _bv in _bs]
            for _gv in _gs
        ]
    )
    _ia, _ib = np.unravel_index(int(np.argmax(_grid)), _grid.shape)
    gamma_star, beta_star = float(_gs[_ia]), float(_bs[_ib])
    cut_patch = float(_grid[_ia, _ib])
    print(
        f"patch tune (14 qubits, {len(PATCH_EDGES)} edges, 60×30 formula "
        f"evaluations): (γ*, β*) = ({gamma_star:.3f}, {beta_star:.3f})"
    )
    print(
        f"formula's ⟨cut⟩ at the peak = {cut_patch:.4f} / {len(PATCH_EDGES)}"
        f" → ratio {cut_patch / len(PATCH_EDGES):.3f}"
        "   (β* is exactly π/8 ≈ 0.393: sin 4β factors out)"
    )

    # Cross-examine the formula against an exact statevector of the patch —
    # the last size where that's possible.
    _cuts = np.array(
        [
            cut_of(format(_k, "014b")[::-1], PATCH_EDGES)
            for _k in range(2**14)
        ]
    )
    _sv = Statevector(
        chip_qaoa_ref(14, PATCH_EDGES, gamma_star, beta_star, measured=False)
    )
    _exact = float(_sv.probabilities() @ _cuts)
    print(
        f"referee cross-check on the patch: formula {cut_patch:.6f} vs "
        f"exact statevector {_exact:.6f}"
    )
    print(
        f"  → |difference| = {abs(cut_patch - _exact):.1e}. "
        "The pencil beats the wall."
    )

    _fig, _ax = plt.subplots(figsize=(7.8, 4.4))
    _im = _ax.imshow(
        _grid.T,
        origin="lower",
        aspect="auto",
        extent=(_gs[0], _gs[-1], _bs[0], _bs[-1]),
        cmap="viridis",
    )
    _ax.scatter(
        [gamma_star],
        [beta_star],
        s=240,
        marker="*",
        c="#D48F26",
        edgecolor="#1b2a4a",
        zorder=3,
        label=f"(γ*, β*) → ⟨cut⟩ = {cut_patch:.2f} / {len(PATCH_EDGES)}",
    )
    _ax.set_xlabel("γ · stamp strength")
    _ax.set_ylabel("β · mixer strength")
    _ax.set_title(
        "the (γ, β) terrain on the 14-qubit patch · "
        "computed from the closed form — no simulator anywhere"
    )
    _ax.legend(loc="upper right", fontsize=9)
    _fig.colorbar(_im, ax=_ax, label=f"⟨cut⟩ (of {len(PATCH_EDGES)})")
    _fig.tight_layout()
    _fig
    return beta_star, cut_patch, gamma_star


@app.cell
def _(
    FakeBrisbane,
    N_WALL,
    WALL_EDGES,
    WALL_PHYS,
    beta_star,
    chip_qaoa_ref,
    gamma_star,
    np,
    transpile,
):
    # ── The hardware-native dividend · the SWAP tax, measured ──────────────
    # Same abstract graph twice: once with our chip-aligned labels, once
    # with the labels randomly shuffled. Same transpiler, same layout.
    _fb = FakeBrisbane()
    _perm = np.random.default_rng(11).permutation(N_WALL)
    _edges_shuffled = [
        (int(min(_perm[_a], _perm[_b])), int(max(_perm[_a], _perm[_b])))
        for _a, _b in WALL_EDGES
    ]
    _n2q = {}
    for _name, _edges in (
        ("chip-aligned (ours)", WALL_EDGES),
        ("same graph, shuffled labels", _edges_shuffled),
    ):
        _tqc = transpile(
            chip_qaoa_ref(N_WALL, _edges, gamma_star, beta_star),
            _fb,
            initial_layout=WALL_PHYS,
            optimization_level=1,
        )
        _n2q[_name] = sum(
            1 for _inst in _tqc.data if _inst.operation.num_qubits == 2
        )
        print(
            f"{_name:<28s} → native two-qubit gates: {_n2q[_name]:5d} · "
            f"depth {_tqc.depth():5d}"
        )
    _ours, _theirs = _n2q.values()
    print()
    print(f"Every RZZ in our circuit sits on a physical coupler, so routing")
    print(f"costs NOTHING: {len(WALL_EDGES)} edges → {_ours} native 2-qubit")
    print(f"gates (2 per RZZ) and not one more. Shuffle the labels and the")
    print(f"SAME graph costs {_theirs} — a {_theirs / _ours:.1f}× SWAP tax,")
    print("straight off this morning's transpilation slide. Every extra gate")
    print("is another roll of the GHZ error dice — hardware-native instances")
    print("aren't an aesthetic choice, they're an error budget.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Build — the chip-graph circuit, from the patch template

    Complete `chip_qaoa56`. It's yesterday's stamp-then-stir, one
    layer, on the machine's own 59 couplers:

    1. `H` on every qubit — all 2⁵⁶ partitions at once. (Given.)
    2. **TODO — the stamp**: `RZZ(-gamma)` on every coupler `(a, b)`
       in `WALL_EDGES` — one loop, 59 edges. **Mind the minus sign**:
       qiskit's RZZ convention points the other way from the formula's
       (the 🧠 fold shows the measured wreck the + sign produces — and
       the payoff cell will catch you if it happens).
    3. **TODO — the mixer**: `RX(2*beta)` on every qubit.
    4. Measure. (Given.)
    """)
    return


@app.cell
def _(N_WALL, QuantumCircuit):
    def chip_qaoa56(gamma: float, beta: float) -> QuantumCircuit:
        """p = 1 Max-Cut QAOA on the chip's own 59 couplers (WALL_EDGES).
        Stamp: RZZ(−γ) per coupler — mind the minus (🧠 fold);
        mixer: RX(2β) per qubit; then measure."""
        qc = QuantumCircuit(N_WALL)

        # (1) Every one of the 2⁵⁶ partitions at once. (Given.)
        qc.h(range(N_WALL))

        # (2) TODO — the stamp: for each (a, b) in WALL_EDGES,
        #     qc.rzz(-gamma, a, b).   ← yes, MINUS gamma.

        # (3) TODO — the mixer: RX(2*beta) on every qubit.

        # (4) Measure. (Given — leave this so the cell runs while you build.)
        qc.measure_all()
        return qc

    return (chip_qaoa56,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Solution — the chip-graph circuit (open if stuck)": mo.md("""
    Replace the two `TODO` comments with:

    ```python
    # (2) the stamp — one RZZ per physical coupler
    for a, b in WALL_EDGES:
        qc.rzz(-gamma, a, b)

    # (3) the mixer
    for q in range(N_WALL):
        qc.rx(2 * beta, q)
    ```

    Byte-for-byte yesterday's rhythm — the news is that the edge list
    came off a real machine's spec sheet, and that 56 sits where a
    simulable number used to.
    """),
            "🧠 The formula, and the sign landmine (measured autopsy)": mo.md(r"""
    **The formula.** For a triangle-free graph, each edge's p = 1
    contribution to ⟨cut⟩ depends only on its endpoints' degrees:

    $$\langle C_{ab}\rangle = \tfrac12 + \tfrac14 \sin 4\beta \sin\gamma
      \left(\cos^{d_a - 1}\gamma + \cos^{d_b - 1}\gamma\right)$$

    (Wang, Hadfield, Jiang & Rieffel 2018, arXiv:1706.02998). Sum over
    edges, done — the tune cell's cross-check against an exact
    statevector agreed to ~10⁻¹⁴, because for triangle-free graphs
    this **is** the exact answer, not an approximation. Two lessons
    hide in it: β enters only through sin 4β (so β* = π/8, always),
    and each edge only *sees its own neighborhood* — the locality that
    makes patch-tuned angles valid on the full chip. That transfer
    trick is a studied research technique: optimal QAOA angles
    *concentrate* across instances and sizes of similar local
    structure (Brandão et al., arXiv:1812.04170; Galda et al.,
    arXiv:2106.07531).

    **The landmine.** Qiskit defines RZZ(θ) = e^(−iθ·ZZ/2); the
    formula's convention stamps e^(−iγ(1−ZZ)/2) per edge. Equal and
    *opposite* — so the stamp must be `rzz(-gamma, a, b)`. We shipped
    the + sign first and measured the wreck: on the 14-qubit patch the
    formula promised ⟨cut⟩ = 9.74 and the circuit delivered **3.26** —
    a perfect mirror image around the coin-flip value |E|/2 = 6.5
    (the wrong sign runs the interference in reverse, *concentrating*
    amplitude on bad cuts exactly as hard as the right sign
    concentrates it on good ones). If your ⟨cut⟩ lands symmetrically
    *below* 29.5 in the payoff, this is why — the payoff cell watches
    for that fingerprint. Phase bookkeeping: still collecting rent.
    """),
        }
    )
    return


@app.cell
def _(
    N_WALL,
    PATCH_EDGES,
    WALL_DEG,
    WALL_EDGES,
    beta_star,
    chip_graph,
    chip_qaoa56,
    chip_qaoa_ref,
    cut_of,
    cut_patch,
    gamma_star,
    ibm_token,
    maxcut_p1_expect,
    mps_backend,
    np,
    plt,
    run_counts,
    send_wall,
    two_coloring,
    wall_pick_backend,
    wall_submit,
):
    # ── Run B · the wall crossing ────────────────────────────────────────────
    def _chip_todo(fn):
        try:
            _ops = fn(0.5, 0.3).count_ops()
        except Exception:
            return True
        return _ops.get("rzz", 0) < len(WALL_EDGES) or "rx" not in _ops

    if _chip_todo(chip_qaoa56):
        _qc = chip_qaoa_ref(N_WALL, WALL_EDGES, gamma_star, beta_star)
        print("⚠️  Your chip_qaoa56 is still TODO — an all-H circuit splits")
        print("    every coupler 50/50 (⟨cut⟩ = 29.5: a coin flip per edge).")
        print("    Borrowing the reference circuit so the crossing still")
        print("    lands; finish the build to make the split yours.")
    else:
        _qc = chip_qaoa56(gamma_star, beta_star)
        print("Circuit: YOUR chip split, driven by the formula-tuned angles.")

    _counts = run_counts(_qc, mps_backend, 2048)
    _E = len(WALL_EDGES)
    _tot = sum(_counts.values())
    _mean = sum(cut_of(_b, WALL_EDGES) * _c for _b, _c in _counts.items()) / _tot
    _best = max(cut_of(_b, WALL_EDGES) for _b in _counts)
    _pred = maxcut_p1_expect(WALL_EDGES, WALL_DEG, gamma_star, beta_star)

    # Hardware path — reread the wiring off the LIVE machine, so "the
    # chip's own graph" stays literally true there too.
    _hw_stats, _hw_name = None, None
    _tok = ibm_token.value.strip()
    if _tok and send_wall.value:
        _real = wall_pick_backend(_tok)
        if _real is not None:
            _phys_l, _edges_l, _deg_l = chip_graph(_real, N_WALL)
            print(
                f"    wiring reread off {_real.name}: {len(_edges_l)} "
                f"couplers · bipartite: "
                f"{two_coloring(N_WALL, _edges_l) is not None}"
            )
            _hw = wall_submit(
                [chip_qaoa_ref(N_WALL, _edges_l, gamma_star, beta_star)],
                _real,
                shots=2048,
                initial_layout=_phys_l,
            )
            if _hw is not None:
                _tot_l = sum(_hw[0].values())
                _mean_l = (
                    sum(cut_of(_b, _edges_l) * _c for _b, _c in _hw[0].items())
                    / _tot_l
                )
                _hw_stats = (
                    _mean_l,
                    max(cut_of(_b, _edges_l) for _b in _hw[0]),
                    len(_edges_l),
                    maxcut_p1_expect(_edges_l, _deg_l, gamma_star, beta_star),
                )
                _hw_name = _real.name

    print()
    print(
        f"patch (14 qubits), formula-tuned:  ⟨cut⟩ = {cut_patch:5.2f} / "
        f"{len(PATCH_EDGES)}   → ratio {cut_patch / len(PATCH_EDGES):.3f}"
    )
    print("full chip graph, SAME two angles:")
    print(
        f"  the formula predicts     ⟨cut⟩ = {_pred:5.2f} / {_E}   "
        f"→ ratio {_pred / _E:.3f}"
    )
    print(
        f"  the MPS referee measured ⟨cut⟩ = {_mean:5.2f} / {_E}   "
        f"→ ratio {_mean / _E:.3f}   (2,048 shots)"
    )
    if _hw_stats is not None:
        _mean_l, _best_l, _E_l, _pred_l = _hw_stats
        print(
            f"  REAL {_hw_name} measured  ⟨cut⟩ = {_mean_l:5.2f} / {_E_l}"
            f"   (formula said {_pred_l:.2f}; the shortfall is"
        )
        print(
            "      the GHZ noise floor at scale — noise drags every edge"
            " toward the coin flip)"
        )
    print(
        f"  best sampled split: {_best} / {_E} couplers cut · "
        f"perfect (bipartite) = {_E}"
    )
    if _mean < _E / 2 - 4:
        print()
        print("⚠️  Your ⟨cut⟩ sits mirror-image BELOW the coin-flip line")
        print("    29.5 — the sign landmine's fingerprint: RZZ(+γ) where")
        print("    RZZ(−γ) belongs. The 🧠 fold has the autopsy; flip the")
        print("    sign and rerun.")
    else:
        print()
        print("Prediction, simulation and (with a token) hardware — three")
        print("referees, one number. Tuned by pencil, deployed past the wall.")

    # fixed-domain histogram over CUT VALUES (2⁵⁶ bitstrings can't be an
    # axis — the cut score is the honest working set)
    _dom = np.arange(24, 60)
    _hist = np.zeros(len(_dom))
    for _b, _c in _counts.items():
        _cv = cut_of(_b, WALL_EDGES)
        if _dom[0] <= _cv <= _dom[-1]:
            _hist[_cv - _dom[0]] += _c / _tot
    _fig, _ax = plt.subplots(figsize=(10.2, 4.2))
    _ax.bar(_dom, _hist, width=0.8, color="#0E7C86", label="MPS referee draws")
    _ax.axvline(
        _E / 2, color="#6B7280", ls="-.", lw=1.6, label=f"coin flip = {_E / 2:.1f}"
    )
    _ax.axvline(
        _pred,
        color="#1b2a4a",
        ls=":",
        lw=2,
        label=f"formula's prediction = {_pred:.2f}",
    )
    _ax.axvline(
        _mean, color="#0E7C86", ls="-", lw=2, label=f"measured ⟨cut⟩ = {_mean:.2f}"
    )
    _ax.axvline(
        _E, color="#B23B7B", ls="--", lw=2, label=f"perfect split = {_E}"
    )
    _ax.set_xlim(_dom[0] - 0.7, _dom[-1] + 0.7)
    _ax.set_xlabel("couplers cut by a sampled 56-qubit partition")
    _ax.set_ylabel("fraction of 2,048 shots")
    _ax.set_title(
        f"the chip split against itself · ⟨cut⟩ = {_mean:.1f} / {_E} "
        f"(ratio {_mean / _E:.3f}) · best sampled {_best}"
    )
    _ax.legend(fontsize=8, loc="upper left")
    _ax.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Wall check** — say the finale precisely; it's the week's last
    honesty drill:

    - **What ran**: two 56-qubit circuits — sizes where a statevector
      simulator would need an exabyte. The morning's cliff (slide C5),
      crossed twice before dinner.
    - **The asterisk, out loud**: our fallback executor was a
      tensor-network simulator, and it worked *only because* shallow
      circuits on thin, sparse graphs are entanglement-poor — the
      wall is entanglement-shaped, not
      qubit-count-shaped. Add depth or a second dimension and the MPS
      referee dies too; the QPU doesn't care. "56 qubits" alone
      impresses no one who read this paragraph — put *that* in your
      hype filter, right next to the gate-fidelity arithmetic.
    - **The load-bearing trick**: parameter transfer. Run A trained
      its knobs on an 8-qubit simulation; Run B tuned its two by
      pencil-and-paper formula — and the priced, queued, noisy
      machine only ever runs final circuits. That's not a classroom
      convenience — it's how variational algorithms are actually
      deployed on NISQ hardware, and you know its research name now
      (parameter concentration).
    - **The grading confession**: Run A was gradeable because Hückel
      theory is a backdoor; Run B because heavy-hex is bipartite and
      59 edges are classically trivial. The problems worth money —
      correlated molecules, lumpy graphs — have no referee at this
      size. When the machine finally beats classical there, the
      *proof* will be the hard part. Which is why the four clocks
      from this morning all wait on the same unlock: error
      correction — the difference between "past the simulator wall"
      and "past all classical reach."
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Wrap — the week, and where the door is

    You built, in five days:
    - **Day 1** — the déjà vu: bits that took a room, and the argument
      that qubits are ENIAC, not magic.
    - **Day 2** — the qubit itself: superposition, measurement,
      the Bloch picture, and the first entangled pair.
    - **Day 3** — interference as an *algorithm*: Grover's needle, the
      walker's race — amplitudes made to cancel on command.
    - **Day 4** — the variational machine: a circuit with knobs, an
      optimizer on the knobs, and Max-Cut bent to its will.
    - **Day 5** — the machine meets the world: a molecule's energy bill
      paid (21 → 88 mHa of correlation won back), a fund picked and
      honestly graded, one circuit sent to a machine that costs
      more than the building — which lost ~7 % of its shots to
      reality — and then **the simulator wall crossed twice at 56
      qubits**: a 56-carbon molecule's resonance energy recovered to
      ~94 %, and the chip split against its own wiring at ~73 % of a
      known-perfect cut — knobs tuned at toy size (one by an 8-qubit
      simulation, one by pencil-and-paper) and never re-tuned. That
      ~7 % loss, not
      any demo, is the state of the art: the entire industry is a race
      to buy those percentage points back.

    **Group presentations — 3 minutes per table, right now.** Each
    table takes the tier it pushed furthest and answers exactly three
    questions, one slide or zero: *What number did you find?* (an
    energy, a fund, a fidelity gap) · *How do you know it's honest?*
    (you diagonalized / enumerated / counted the impossible) · *What's
    the one caveat you'd tell a CEO before they wire money?* The
    honest caveat is the skill this course was secretly about.

    **Where to go next** — grab the handout on your way out (QR codes
    on it): Quantum Country (spaced-repetition essays that keep this
    week's quiz schedule going), Nielsen & Chuang for depth, Preskill's
    Ph 219 notes for rigor, the Qiskit tutorials for more of exactly
    what you did today, and the community rooms where the arguing
    happens. The clocks from this morning run whether you watch them
    or not — but you now read them natively. 🎓
    """)
    return


if __name__ == "__main__":
    app.run()
