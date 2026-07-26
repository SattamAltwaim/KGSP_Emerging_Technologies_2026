# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo","numpy","matplotlib","qiskit","qiskit-aer","pylatexenc","scipy"]
# ///
"""Day 4 lab · Marimo notebook · QAOA on Max-Cut → the seating chart

Tapered-tier structure per PEDAGOGY.md § Difficulty polarity:
    Baseline (everyone with TA support): Section 1 — the seating-chart problem
    Stretch (most students):             Section 2 — the QAOA machine
    Aspirational (top ~30%):             Section 3 — the class chart, for real

Each tier runs the `explore → predict → build` micro-structure (decision #23):
an `mo.ui` widget to explore, a predict-then-run commitment, then a scaffolded
build. TAs cover mechanical snags (uv, imports, molab hiccups).

SINGLE NOTEBOOK (conventions per decision #27 / docs/lab-authoring-playbook.md):
solutions live HERE, collapsed in `mo.accordion` blocks under each build cell —
hidden by default, one click to open. The Explore widgets run on hidden
reference engines (`ref_cut`, `qaoa_ref`, hide_code cells) so exploration works
BEFORE the student builds anything; the student's own `maxcut_cost` /
`qaoa_circuit` drive the check and payoff cells. The cost layer is a BLACK BOX:
students call `make_cost_layer(n_nodes, edges, gamma)` and get one opaque
cost(γ) gate (one closed box in circuit drawings, matching the morning's
framing); its internals are in an accordion. The PEP 723 header lets molab /
`marimo edit --sandbox` auto-install qiskit + qiskit-aer + scipy.

Stakes numbers in the prose were MEASURED (playbook §1), not guessed: the
demo-graph p=1 terrain peaks at ⟨C⟩ ≈ 3.24 at (γ, β) = (2.85, 1.89) — the
slider defaults; the p-sweep climbs from ≈ 0.81 at p=1 to ≈ 0.95–0.99 by p=3–4 (p=2 varies
0.82–0.96 run to run);
the 10-node class graph has max cut 11/13 and QAOA p=3 reaches ratio
≈ 0.70–0.81 (median ≈ 0.78 over repeated runs) with ~2–9% of draws on an
optimal chart (blind guessing: 0.4%).
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
    # Day 4 lab · QAOA seating charts

    This morning you crawled the (γ, β) landscape — a quantum circuit
    with *knobs*, a classical optimizer turning them, and a terrain of
    scores underneath. This afternoon that machine gets a real job: by
    the end of this notebook it will have computed the **actual seating
    chart** for the rest of today — the class's own friendship graph
    goes in, two project teams come out.

    **How to use this notebook**:
    - Cells run top-to-bottom. Try each in order.
    - Each section follows the same rhythm: **explore** (move a slider,
      watch the pictures), **predict** (commit an answer *before* you
      run), **build** (write the code from the recipe).
    - Stuck? Every build has a **💡 Solution** fold right under it —
      closed by default. Predict first, peek last.
    - No take-home. Anything you don't finish now, we don't chase later.

    **Three sections**:
    1. **The seating-chart problem** — baseline. Everyone finishes.
    2. **The QAOA machine** — stretch. Build the circuit, then put an
       optimizer on the knobs.
    3. **The class chart, for real** — aspirational. Your machine, our
       graph, a seating chart we actually use.
    """)
    return


@app.cell
def _():
    # Backend selector. Leave this False on molab/Colab — the local simulator is
    # fast, deterministic, and needs no queue. Flip to True only if you've linked
    # an IBM Quantum account and filled in the service-load in the next cell.
    USE_REAL_HW = False
    return (USE_REAL_HW,)


@app.cell
def _(AerSimulator, USE_REAL_HW):
    if not USE_REAL_HW:
        backend = AerSimulator()
        print(
            "Using local simulator (AerSimulator). Fast, deterministic, no queue."
        )
    else:
        # To use real hardware, link your IBM Quantum account and uncomment:
        #   from qiskit_ibm_runtime import QiskitRuntimeService
        #   service = QiskitRuntimeService()
        #   backend = service.least_busy(operational=True, simulator=False)
        print(
            "Real hardware selected, but no service configured — using simulator."
        )
        backend = AerSimulator()  # fallback
    return (backend,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 1 · The seating-chart problem (baseline)

    Ten people, two project teams — but every cohort has cliques, and a
    chart that seats a whole clique together makes one loud table and
    one silent one. A good chart **splits the cliques up**: every
    friendship that *crosses* the team line is one more bridge between
    the tables.

    So here's the game, warmed up on four people first:

    > Given a friendship graph, put each person on **Team Teal** or
    > **Team Amber** so that as many friendships as possible **cross**
    > between the teams.

    This problem has a name — **Max-Cut** (a crossing friendship is
    "cut" by the team line) — and it is famously hard at scale. A
    seating chart is just a **bitstring**: one bit per person, `0` =
    Team Teal, `1` = Team Amber. Four people → 4 bits → **16 possible
    charts**. The *score* of a chart = how many friendships cross.
    """)
    return


@app.cell
def _():
    # The warm-up graph: 4 people, 5 friendships.
    # (A–B, A–C, A–D, B–C, C–D — everyone is friends with A and with C.)
    DEMO_NODES = ["A", "B", "C", "D"]
    DEMO_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
    return DEMO_EDGES, DEMO_NODES


@app.cell(hide_code=True)
def _(np, plt):
    # Shared graph drawer for every section — same layout, same colours, so
    # every graph picture in the notebook speaks the same visual language.
    def draw_graph(names, edges, bits=None, title="", figsize=(5.4, 5.4)):
        """Circular layout, node 0 at the top, clockwise. Node i ↔ qubit i ↔
        bit position i (left-to-right in the chart bitstring). bits=None →
        uncoloured problem instance; bits given → Team Teal ('0') / Team
        Amber ('1'), crossing friendships dashed magenta."""
        n = len(names)
        ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
        x, y = np.cos(ang), np.sin(ang)
        fig, ax = plt.subplots(figsize=figsize)
        for i, j in edges:
            crossing = bits is not None and bits[i] != bits[j]
            ax.plot(
                [x[i], x[j]],
                [y[i], y[j]],
                color="#B23B7B" if crossing else "#9a9a9a",
                lw=3.2 if crossing else 1.6,
                ls=(0, (4, 2)) if crossing else "-",
                zorder=1,
            )
        if bits is None:
            cols = ["#1b2a4a"] * n
        else:
            cols = ["#0E7C86" if b == "0" else "#D48F26" for b in bits]
        ax.scatter(x, y, s=1600 if n <= 6 else 1250, c=cols, zorder=2)
        for i in range(n):
            ax.annotate(
                f"{names[i]}\nq{i}",
                (x[i], y[i]),
                ha="center",
                va="center",
                fontsize=11 if n <= 6 else 9.5,
                color="white",
                fontweight="bold",
                zorder=3,
            )
        ax.set_xlim(-1.42, 1.42)
        ax.set_ylim(-1.42, 1.42)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)
        fig.tight_layout()
        return fig

    return (draw_graph,)


@app.cell(hide_code=True)
def _(DEMO_EDGES, transpile):
    # ── Hidden reference engines ─────────────────────────────────────────────
    # These power the EXPLORE widgets (and the guards) only, so you can explore
    # real behaviour BEFORE you've built anything. Your own `maxcut_cost` /
    # `qaoa_circuit` are what the check and payoff cells use. No peeking
    # needed.
    def ref_cut(bits, edges):
        """Reference cut score (this is also the Section-1 solution)."""
        return sum(1 for (i, j) in edges if bits[i] != bits[j])

    def run_counts(qc, backend_, shots=2048):
        """Run a measured circuit; return counts keyed by NODE-ORDER
        bitstrings (bit i = node i, read left to right). Qiskit's raw keys
        are little-endian — this reverses them once so no other cell has to
        think about it."""
        tc = transpile(qc, backend_)
        raw = backend_.run(tc, shots=shots).result().get_counts()
        return {k[::-1]: v for k, v in raw.items()}

    def expected_cut(counts, edges, cost_fn):
        """⟨C⟩ = Σₓ P(x)·C(x): the average cut score of the samples.
        (Why the average? See the 🧠 fold in Section 2.)"""
        total = sum(counts.values())
        return sum(cost_fn(b, edges) * c for b, c in counts.items()) / total

    def cost_is_todo(cost_fn):
        """Guard helper: detect the shipped/TODO state of a cost function."""
        try:
            vals = [cost_fn(format(k, "04b"), DEMO_EDGES) for k in range(16)]
        except Exception:
            return True
        return any(v is None for v in vals) or all(v == 0 for v in vals)

    def circuit_is_todo(circuit_fn):
        """Guard helper: detect the shipped/TODO state of a QAOA circuit."""
        try:
            ops = circuit_fn(4, DEMO_EDGES, [0.4], [0.4]).count_ops()
        except Exception:
            return True
        return "cost" not in ops or "rx" not in ops

    return circuit_is_todo, cost_is_todo, expected_cut, ref_cut, run_counts


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The graph you're cutting

    Before any circuits: *this* is the problem. Four people, five
    friendships (gray lines). Every node is also a **qubit** and a **bit
    position**: node `A` is qubit `q0` and the *leftmost* bit of the
    chart bitstring, `B` is `q1` and the second bit, and so on. That's
    the entire encoding — chart `0101` means A·Teal, B·Amber, C·Teal,
    D·Amber.
    """)
    return


@app.cell(hide_code=True)
def _(DEMO_EDGES, DEMO_NODES, draw_graph):
    draw_graph(
        DEMO_NODES,
        DEMO_EDGES,
        title="the warm-up graph · node ↔ qubit ↔ bit position",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Explore

    The slider scrubs through **all 16 seating charts** — the chart
    number in binary *is* the bitstring. Teal and amber nodes are the
    two teams; a friendship that crosses the line lights up **magenta**,
    and the score is how many light up.

    Hunt for the best chart. What's the highest score you can find — and
    how many different charts reach it?
    """)
    return


@app.cell
def _(mo):
    cut_pick = mo.ui.slider(
        start=0,
        stop=15,
        value=1,
        step=1,
        label="seating chart # (its 4-bit binary = the chart)",
        show_value=True,
    )
    cut_pick
    return (cut_pick,)


@app.cell
def _(DEMO_EDGES, DEMO_NODES, cut_pick, draw_graph, ref_cut):
    _bits = format(cut_pick.value, "04b")
    _score = ref_cut(_bits, DEMO_EDGES)
    _teal = [DEMO_NODES[_i] for _i in range(4) if _bits[_i] == "0"]
    _amber = [DEMO_NODES[_i] for _i in range(4) if _bits[_i] == "1"]
    print(
        f"chart #{cut_pick.value} = {_bits} · Team Teal: "
        f"{', '.join(_teal) or '—'} · Team Amber: {', '.join(_amber) or '—'}"
    )
    print(f"score = {_score} of 5 friendships cross the team line")
    draw_graph(
        DEMO_NODES,
        DEMO_EDGES,
        bits=_bits,
        title=f"chart {_bits} · score {_score}/5",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Predict

    Commit to two answers before you build (say them out loud or jot
    them down — committing is the point):

    1. Can a chart ever score **5 of 5**? Look at the triangle
       `A–B–C`: two teams, three people — can all three of those
       friendships cross at once?
    2. So what's the **maximum score**, and how many of the 16 charts
       achieve it?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Build

    Complete `maxcut_cost` below — the scorekeeper. Given a chart
    bitstring and the friendship list, count the crossing friendships.

    - `bits[i]` is node `i`'s team as a character, `"0"` or `"1"` —
      in chart `0101`, `bits[0]` puts `A` on Teal and `bits[1]` puts
      `B` on Amber.
    - A friendship `(i, j)` **crosses** exactly when its endpoints
      disagree: `bits[i] != bits[j]`.

    Small as it looks, this function is the day's centre of gravity:
    the brute-force table below, the QAOA score ⟨C⟩ in Section 2, and
    the real seating chart in Section 3 are all judged by *your*
    `maxcut_cost`.
    """)
    return


@app.function
def maxcut_cost(bits: str, edges) -> int:
    """Score of a seating chart: how many friendships (i, j) have their
    endpoints on different teams (bits[i] != bits[j])."""
    score = 0
    for i, j in edges:
        # TODO: if nodes i and j are on different teams, add 1 to score.
        #   (Compare bits[i] with bits[j].)
        if bits[i] != bits[j]:
            score += 1
        pass
    return score


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Solution — the scorekeeper (open if stuck)": mo.md("""
    Replace the `TODO` comment (and the `pass`) with:

    ```python
    for i, j in edges:
        if bits[i] != bits[j]:
            score += 1
    ```

    That's the whole thing. One character per person, one comparison
    per friendship: endpoints disagree → the friendship crosses the
    line → +1. (The one-liner
    `sum(1 for i, j in edges if bits[i] != bits[j])` is the same
    function in a party dress.)
    """),
        }
    )
    return


@app.cell
def _(DEMO_EDGES, np, plt):
    # Brute force: score ALL 16 charts with YOUR maxcut_cost.
    _charts = [format(_k, "04b") for _k in range(16)]
    _scores = [maxcut_cost(_b, DEMO_EDGES) or 0 for _b in _charts]
    _best = max(_scores)
    _winners = [_b for _b, _s in zip(_charts, _scores) if _s == _best]

    if _best == 0:
        print(
            "⚠️  Every chart scores 0 → the TODO in `maxcut_cost` isn't filled"
        )
        print(
            "    in yet. A scorekeeper that gives everyone zero can't tell a"
        )
        print(
            "    great chart from a terrible one — fill in the comparison and"
        )
        print(
            "    rerun. (Once it works: best = 4, and exactly two charts tie.)"
        )
    else:
        print(
            f"best score = {_best} of 5 · achieved by: {', '.join(_winners)}"
        )
        print("Those two winners are mirror images — swap the team names and")
        print(
            "it's the same chart. And nothing reaches 5 (blame the triangles)."
        )

    _fig, _ax = plt.subplots(figsize=(8.6, 3.8))
    _x = np.arange(16)
    _cols = [
        "#D48F26" if (_s == _best and _best > 0) else "#0E7C86"
        for _s in _scores
    ]
    _ax.bar(_x, _scores, color=_cols)
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_charts, rotation=90)
    _ax.set_ylim(0, 5.3)
    _ax.set_yticks(range(6))
    _ax.axhline(5, color="#9a9a9a", ls=":", lw=1)
    _ax.annotate(
        "5 = impossible (triangles)", (0.1, 5.05), fontsize=9, color="#6B7280"
    )
    _ax.set_xlabel("seating chart (bitstring)")
    _ax.set_ylabel("score (your maxcut_cost)")
    _ax.set_title("all 16 charts, brute-forced · best in amber")
    _ax.grid(True, axis="y", alpha=0.3)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Baseline check** — before moving on:

    - Did `0101` and `1010` tie at **4**, matching the best chart you
      found with the slider?
    - Why is 5 impossible? (A triangle's three people only have two
      teams to sit at — two of them must share, so at most 2 of a
      triangle's 3 friendships can ever cross. This graph has two
      triangles, sharing the edge `A–C`.)
    - 16 charts brute-force in a blink. Every extra person **doubles**
      the count — hold that thought until Section 3's 1,024, and then
      imagine 60 people at 2⁶⁰ ≈ 10¹⁸.

    If your table matches the slider hunt: the scorekeeper works. Move
    to Section 2 and meet the machine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2 · The QAOA machine (stretch)

    Brute force checks charts *one at a time*. The quantum move — same
    as Grover yesterday — is to hold **all 16 charts in superposition**
    and make the good ones interfere their way to the top. QAOA does it
    with two alternating moves, each with a knob:

    1. **Cost stamp** (knob **γ**) — stamp every chart in the
       superposition with a phase that depends on *its own score*.
       Higher score, further-turned stamp.
    2. **Mixer** (knob **β**) — stir amplitude between charts so the
       stamps *interfere*: rotate every qubit with `RX(2β)`.

    One (stamp, stir) pair = one **layer**; `p` layers = `p` pairs of
    knobs. This is the morning's picture exactly: γ pushes toward good
    answers, β keeps exploring, and *finding the right knobs is the
    whole game* — that's what "variational" means.

    **The cost stamp is given as a black box.** Like yesterday's oracle:
    `make_cost_layer(n_nodes, edges, gamma)` hands you one opaque
    `cost(γ)` gate. You may *call* it, not open it (a "🔍 how is it
    built?" fold waits below for after your build works).
    """)
    return


@app.cell(hide_code=True)
def _(QuantumCircuit):
    # ── THE BLACK BOX · given ────────────────────────────────────────────────
    # The cost stamp, exactly as framed in the lecture: a gate you may CALL,
    # not open. `make_cost_layer(n_nodes, edges, gamma)` returns one opaque
    # box that phase-stamps every chart |x⟩ according to its own cut score.
    # How it works inside is in the "How is cost(γ) built?" fold below —
    # finish your build before opening it.
    def make_cost_layer(n_nodes: int, edges, gamma: float):
        """The black box: one opaque gate that stamps each basis state |x⟩
        with a phase set by its cut score C(x). Call it with
        `qc.append(make_cost_layer(n, edges, gamma), range(n))`."""
        _qc = QuantumCircuit(n_nodes, name="cost")
        for _i, _j in edges:
            _qc.rzz(2 * gamma, _i, _j)
        return _qc.to_gate(label="cost(γ)")

    return (make_cost_layer,)


@app.cell(hide_code=True)
def _(QuantumCircuit, make_cost_layer):
    # Reference engine for the EXPLORE widget only (same deal as Day 3's
    # grover_demo): a full working QAOA so you can explore before you build.
    # Your own `qaoa_circuit` below is what the checks and payoffs use.
    def qaoa_ref(n_nodes, edges, gammas, betas):
        qc = QuantumCircuit(n_nodes, n_nodes)
        qc.h(range(n_nodes))
        for g, b in zip(gammas, betas):
            qc.append(make_cost_layer(n_nodes, edges, g), range(n_nodes))
            for q in range(n_nodes):
                qc.rx(2 * b, q)
        qc.measure(range(n_nodes), range(n_nodes))
        return qc

    return (qaoa_ref,)


@app.cell(hide_code=True)
def _(DEMO_EDGES, np):
    # The p=1 (γ, β) terrain of ⟨C⟩ for the warm-up graph — the same landscape
    # the morning's hero animation crawled. Computed once, exactly (a 4-qubit
    # statevector is 16 numbers; no shots needed for a backdrop).
    _cv = np.array(
        [
            sum(1 for (i, j) in DEMO_EDGES if ((b >> i) & 1) != ((b >> j) & 1))
            for b in range(16)
        ],
        float,
    )
    _zz = len(DEMO_EDGES) - 2 * _cv  # Σ_edges z_i·z_j per basis state

    def _expect(g, bt):
        psi = np.full(16, 0.25, complex) * np.exp(-1j * g * _zz)
        rx = np.array(
            [[np.cos(bt), -1j * np.sin(bt)], [-1j * np.sin(bt), np.cos(bt)]]
        )
        for q in range(4):
            psi = psi.reshape(2 ** (3 - q), 2, 2**q)
            psi = np.einsum("ab,ibj->iaj", rx, psi).reshape(-1)
        return float(np.abs(psi) ** 2 @ _cv)

    LAND_G = np.linspace(0.0, np.pi, 61)
    LAND_B = np.linspace(0.0, np.pi, 61)
    LAND = np.array([[_expect(_g, _b) for _g in LAND_G] for _b in LAND_B])
    return LAND, LAND_B, LAND_G


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Explore

    Two knobs, two pictures, one state. **Left**: the measured
    distribution of *chart scores* (4,096 shots on the reference
    machine, p = 1), plus the running average **⟨C⟩** — the score you'd
    expect drawing one chart from the machine at random. **Right**: the
    morning's terrain — ⟨C⟩ over the whole (γ, β) plane — with a dot
    marking where your knobs sit.

    The sliders start parked on the terrain's **peak**: ⟨C⟩ ≈ 3.24,
    against 2.50 for blind guessing. Now wreck it, on purpose: drag γ
    to 0 and watch the machine forget the problem; find the dark valley
    (⟨C⟩ ≈ 0.9) where the knobs are actively *worse* than guessing.
    The terrain is real — you're crawling it by hand.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    gamma_s = mo.ui.slider(
        start=0.0,
        stop=3.14,
        value=2.85,
        step=0.01,
        label="γ · cost-stamp knob",
        show_value=True,
    )
    beta_s = mo.ui.slider(
        start=0.0,
        stop=3.14,
        value=1.89,
        step=0.01,
        label="β · mixer knob",
        show_value=True,
    )
    mo.vstack([gamma_s, beta_s])
    return beta_s, gamma_s


@app.cell(hide_code=True)
def _(
    DEMO_EDGES,
    LAND,
    LAND_B,
    LAND_G,
    backend,
    beta_s,
    expected_cut,
    gamma_s,
    np,
    plt,
    qaoa_ref,
    ref_cut,
    run_counts,
):
    _shots = 4096
    _counts = run_counts(
        qaoa_ref(4, DEMO_EDGES, [gamma_s.value], [beta_s.value]),
        backend,
        _shots,
    )
    _ev = expected_cut(_counts, DEMO_EDGES, ref_cut)

    # Score distribution over the FIXED domain 0..5 (5 stays visible at zero —
    # the triangles keep it empty, and that emptiness is information).
    _dist = np.zeros(6)
    for _b, _c in _counts.items():
        _dist[ref_cut(_b, DEMO_EDGES)] += _c
    _dist /= _shots

    print(
        f"(γ, β) = ({gamma_s.value:.2f}, {beta_s.value:.2f}) · "
        f"⟨C⟩ = {_ev:.2f} crossing friendships per draw · "
        f"blind guessing = 2.50"
    )

    _fig, (_ax1, _ax2) = plt.subplots(
        1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    _cols = ["#0E7C86"] * 4 + ["#D48F26", "#6B7280"]
    _ax1.bar(np.arange(6), _dist, color=_cols)
    _ax1.axvline(
        _ev, color="#1b2a4a", ls="--", lw=1.6, label=f"⟨C⟩ = {_ev:.2f}"
    )
    _ax1.axvline(
        2.5, color="#6B7280", ls=":", lw=1.4, label="blind guessing = 2.50"
    )
    _ax1.set_xticks(range(6))
    _ax1.set_xticklabels(["0", "1", "2", "3", "4\n★ best", "5\nimpossible"])
    _ax1.set_ylim(0, 1.0)
    _ax1.set_xlabel("chart score (crossing friendships)")
    _ax1.set_ylabel("probability")
    _ax1.set_title(f"{_shots} shots · score distribution")
    _ax1.legend(loc="upper left", fontsize=9)
    _ax1.grid(True, axis="y", alpha=0.3)

    _pc = _ax2.pcolormesh(LAND_G, LAND_B, LAND, cmap="viridis", shading="auto")
    _ax2.scatter(
        [gamma_s.value],
        [beta_s.value],
        s=110,
        facecolor="#B23B7B",
        edgecolor="white",
        lw=1.6,
        zorder=3,
    )
    _ax2.set_xlabel("γ")
    _ax2.set_ylabel("β")
    _ax2.set_title("the morning's terrain · p = 1 · peak ⟨C⟩ ≈ 3.24")
    _fig.colorbar(_pc, ax=_ax2, label="⟨C⟩")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Predict

    Commit before you touch the sliders again:

    1. Set **γ = 0** (stamp off). What will the score distribution look
       like, and what's ⟨C⟩? (What did the machine just forget?)
    2. Read the terrain's colorbar: the p = 1 peak is ⟨C⟩ ≈ 3.24, and
       the perfect chart scores 4. Can **one** layer ever make the best
       chart *certain* — or is that what more layers are for?
    3. At the peak, what fraction of draws land on one of the two best
       charts? Pick one: ~12% (blind), ~33%, ~66%, ~100%. (Check
       yourself after: it's the ★ bar's height at the default knobs.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build

    Complete `qaoa_circuit` below. The recipe, layer by layer:

    1. **`H` on every qubit** — all 16 charts at once, equal amplitude.
       (Given.)
    2. **Cost stamp** — *call the black box*:
       `qc.append(make_cost_layer(n_nodes, edges, gamma), range(n_nodes))`
       drops one opaque `cost(γ)` box into the circuit.
    3. **Mixer — yours**: `RX(2β)` on **every** qubit —
       `qc.rx(2 * beta, q)`.
    4. Repeat 2–3 once per layer (the `for` loop is given), then
       **measure**. (Given.)

    **Why the mixer is not optional** — the stamp writes each chart's
    score into its *phase*, and a phase changes **no** measurement
    probabilities. Sound familiar? Yesterday the oracle's flip was
    invisible until the diffuser turned it into amplitude. Same trick,
    new clothes: `RX(2β)` leans every qubit part-way between its two
    teams, letting neighbouring charts' amplitudes overlap and
    *interfere* — and only then do the stamped phases tilt the odds
    toward high scores. The stamp writes, the mixer cashes in. Full
    algebra in the folds below.
    """)
    return


@app.cell
def _(QuantumCircuit, make_cost_layer):
    def qaoa_circuit(n_nodes: int, edges, gammas, betas) -> QuantumCircuit:
        """p-layer QAOA for Max-Cut: H on all qubits, then per layer a cost
        stamp (black box, knob gammas[k]) and a mixer (RX(2·betas[k]) on
        every qubit), then measure. len(gammas) == len(betas) == p."""
        qc = QuantumCircuit(n_nodes, n_nodes)

        # (1) Every chart at once: uniform superposition. (Given.)
        qc.h(range(n_nodes))

        for gamma, beta in zip(gammas, betas):
            # (2) TODO — COST STAMP: call the black box (don't open it):
            qc.append(make_cost_layer(n_nodes, edges, gamma), range(n_nodes))
            pass

            # (3) TODO — MIXER: RX(2*beta) on EVERY qubit:
            qc.rx(2 * beta, list(range(n_nodes)))
            pass

        # (4) Measure. (Given — leave this so the cell runs while you build.)
        qc.measure(range(n_nodes), range(n_nodes))
        return qc

    return (qaoa_circuit,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Solution — one QAOA layer (open if stuck)": mo.md(r"""
    Replace the two `TODO` blocks (and both `pass` lines) with:

    ```python
    # (2) Cost stamp — the black box, called as a function.
    qc.append(make_cost_layer(n_nodes, edges, gamma), range(n_nodes))

    # (3) Mixer — RX(2β) on every qubit.
    for q in range(n_nodes):
        qc.rx(2 * beta, q)
    ```

    That's a whole layer: stamp (γ), stir (β). The
    `for gamma, beta in zip(gammas, betas)` loop around it repeats the
    pair once per layer — `p` layers means `p` γ's and `p` β's, which
    is exactly why deeper QAOA is *better and harder*: more knobs to
    get right.
    """),
            "🧠 Why does RZZ(2γ) per edge stamp the score?": mo.md(r"""
    The black box applies $R_{ZZ}(2\gamma) = e^{-i\gamma Z_i Z_j}$ once
    per friendship $(i,j)$. On a basis state (a definite chart),
    $Z_i Z_j$ reads $+1$ if $i, j$ sit on the **same** team and $-1$ if
    the friendship **crosses**. So each edge contributes a phase
    $e^{-i\gamma}$ (same team) or $e^{+i\gamma}$ (crossing), and a
    chart $x$ with score $C(x)$ collects

    $$|x\rangle \;\longmapsto\; e^{-i\gamma\,(E - 2C(x))}\,|x\rangle
      \;=\; \underbrace{e^{-i\gamma E}}_{\text{global, invisible}}
      \; e^{\,2i\gamma\,C(x)}\,|x\rangle,$$

    with $E$ the number of edges. Every chart gets **its own score
    written into its phase angle** — turned further the more
    friendships it cuts. That's the entire "stamp". And because it's
    diagonal (each $|x\rangle$ only picks up a phase), the edge stamps
    commute — the box is just one tiny stamp per friendship, in any
    order.

    One more echo of yesterday: measured *right now*, the stamped state
    is still exactly uniform. The mark is invisible until interference
    (the mixer) converts phase into probability.
    """),
            "🧠 Why score with the *average* ⟨C⟩ = Σ P(x)·C(x)?": mo.md(r"""
    Run the circuit, collect samples, score each sample with
    `maxcut_cost`, take the mean:

    $$\langle C\rangle \;=\; \sum_x P(x)\,C(x).$$

    Why hand the optimizer the *average* instead of the best sample?
    Because the average is a **smooth, honest signal**: nudge (γ, β) a
    little and ⟨C⟩ moves a little — that's what makes the terrain
    picture a *landscape* an optimizer can climb. Best-of-a-batch is a
    jumpy statistic (one lucky shot spikes it), and a lucky spike at
    bad knobs would teach the optimizer exactly the wrong lesson.

    So: **optimize the average, keep the best.** When the tuning is
    done we still hand you the best *sampled* chart as the answer —
    that's how Section 3 picks the seating chart.
    """),
            "🔍 How is the black box cost(γ) built? (open after your build works)": mo.md(r"""
    Smaller than you'd think — one two-qubit gate per friendship:

    ```python
    def make_cost_layer(n_nodes, edges, gamma):
        qc = QuantumCircuit(n_nodes, name="cost")
        for i, j in edges:
            qc.rzz(2 * gamma, i, j)          # e^{-iγ Z_i Z_j}
        return qc.to_gate(label="cost(γ)")   # ← packaged: one opaque box
    ```

    `rzz` is qiskit's ready-made $e^{-i\theta\, Z\otimes Z/2}$ (a
    CX·RZ·CX sandwich under the hood — library helper over gate
    gymnastics), and `to_gate(label="cost(γ)")` collapses the pile into
    the single closed box you saw in the circuit drawing — exactly how
    the lecture drew it. Note what *isn't* in there: nothing about
    which chart is best. The box only knows the edge list; the phases
    do the ranking.
    """),
        }
    )
    return


@app.cell(hide_code=True)
def _(
    DEMO_EDGES,
    backend,
    circuit_is_todo,
    cost_is_todo,
    expected_cut,
    qaoa_circuit,
    ref_cut,
    run_counts,
):
    # Check: YOUR circuit at the hand-tuned peak knobs (γ, β) = (2.85, 1.89),
    # scored by YOUR maxcut_cost. Drawn below at p = 2 so you can see the
    # stamp–stir rhythm repeat (cost(γ) appears as ONE closed box per layer).
    _cost = maxcut_cost if not cost_is_todo(maxcut_cost) else ref_cut
    if cost_is_todo(maxcut_cost):
        print(
            "ℹ️  Your maxcut_cost still scores everything 0 (Section 1 TODO) —"
        )
        print("    borrowing the reference scorer so this check can speak.")

    _qc1 = qaoa_circuit(4, DEMO_EDGES, [2.85], [1.89])
    _counts = run_counts(_qc1, backend, 4096)
    _ev = expected_cut(_counts, DEMO_EDGES, _cost)
    print(f"Your machine at (γ, β) = (2.85, 1.89): ⟨C⟩ = {_ev:.2f}")

    if circuit_is_todo(qaoa_circuit):
        _ops = _qc1.count_ops()
        if "cost" not in _ops:
            print(
                "⚠️  No cost(γ) box in your circuit → TODO (2) isn't filled in."
            )
            print("    Without the stamp no chart is marked — there's nothing")
            print("    for interference to amplify.")
        if "rx" not in _ops:
            print("⚠️  No RX mixer in your circuit → TODO (3) isn't filled in.")
            print(
                "    ⟨C⟩ sits at ~2.50 = blind guessing: the stamp only turns"
            )
            print(
                "    phases, and a phase changes NO probabilities (yesterday's"
            )
            print("    invisible mark!). Add the mixer to cash the stamp in.")
    else:
        print(
            "Reference machine at the same knobs: ⟨C⟩ ≈ 3.24. If yours reads"
        )
        print("~3.2, your machine matches the Explore engine — it's real.")

    qaoa_circuit(4, DEMO_EDGES, [2.85, 2.85], [1.89, 1.89]).draw("mpl")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The optimizer — turning the knobs automatically

    You just tuned (γ, β) by hand, on a terrain we precomputed for you.
    Real problems don't ship with a terrain map — there are 2p knobs
    and every ⟨C⟩ reading costs a batch of shots. So we hand the knobs
    to a **classical optimizer** (scipy's COBYLA): it proposes knob
    settings, *your quantum circuit* reports ⟨C⟩, and it proposes again
    — the variational loop from this morning, closed for real.

    One wrinkle: scipy only knows how to *minimize*, and we want the
    **biggest** ⟨C⟩ — so we feed it **−⟨C⟩** and let it dig. We give it
    two starting guesses on opposite sides of the terrain and keep the
    taller hill: watch it rediscover, in ~60 measured evaluations per
    start, the peak you found by hand — while the losing start usually
    parks on a real-but-lower ridge (⟨C⟩ ≈ 3.1). Local optima aren't a
    bug in this story; they *are* the story.
    """)
    return


@app.cell
def _(
    DEMO_EDGES,
    backend,
    circuit_is_todo,
    cost_is_todo,
    expected_cut,
    minimize,
    np,
    qaoa_circuit,
    ref_cut,
    run_counts,
):
    def optimize_qaoa(
        n_nodes, edges, p, inits, circuit_fn, cost_fn, shots=2048, maxiter=60
    ):
        """COBYLA-tune the 2p knobs to maximize ⟨C⟩ (by minimizing −⟨C⟩).
        Tries each starting point in `inits`, keeps the best; returns
        (best_params, best_⟨C⟩). Parameter layout: x = [γ₁..γₚ, β₁..βₚ]."""

        def _neg_expectation(x):
            qc = circuit_fn(n_nodes, edges, x[:p], x[p:])
            return -expected_cut(
                run_counts(qc, backend, shots), edges, cost_fn
            )

        best_x, best_val = None, -1.0
        for x0 in inits:
            res = minimize(
                _neg_expectation,
                np.asarray(x0, float),
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 0.4},
            )
            val = -_neg_expectation(res.x)  # fresh estimate at the optimum
            if val > best_val:
                best_x, best_val = res.x, val
        return best_x, best_val

    # ── run it: p = 1 on the warm-up graph, YOUR machine in the loop ──
    # (two starts on opposite sides of the terrain; the better hill wins)
    _cost = maxcut_cost if not cost_is_todo(maxcut_cost) else ref_cut
    _x, _v = optimize_qaoa(
        4, DEMO_EDGES, 1, [[0.35, 0.35], [2.5, 1.6]], qaoa_circuit, _cost
    )
    print(
        f"COBYLA's verdict: (γ*, β*) = ({_x[0]:.2f}, {_x[1]:.2f}) "
        f"with ⟨C⟩ = {_v:.2f}"
    )
    if circuit_is_todo(qaoa_circuit):
        print("⚠️  ⟨C⟩ is pinned near 2.50 and the 'optimum' wandered nowhere:")
        print("    with the TODOs empty your circuit is all H's — every knob")
        print("    setting gives the same blind-guess distribution, so the")
        print("    terrain is a flat pancake and COBYLA has nothing to climb.")
        print("    Finish the build above and rerun this cell.")
    else:
        print("Hand-found peak for comparison: (2.85, 1.89) with ⟨C⟩ ≈ 3.24.")
        print("(The terrain has mirror-image peaks — γ ↔ π−γ with β ↔ π−β —")
        print(
            " so landing on a twin of your peak still counts as finding it.)"
        )
    return (optimize_qaoa,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### More layers — the p-sweep

    One layer peaked at ⟨C⟩ ≈ 3.24 of 4. The QAOA promise is that
    **depth buys quality**: each extra layer adds a (γ, β) pair, and
    with the right knobs the machine funnels more amplitude onto the
    best charts. But each layer also makes the terrain *ridgier* —
    better solutions, harder optimization (this morning's crawl,
    remember). The sweep below runs p = 1…4 with **your** circuit,
    warm-starting each depth from the last (~20 s of honest shots —
    watch the printout tick).

    **Predict first**: measured on this graph, the approximation ratio
    ⟨C⟩/4 starts at ≈ 0.81 for p = 1 and reaches ≈ 0.95+ by p = 3–4
    (lucky runs get there at p = 2 already). Where does it stop — does
    p = 4 reach 1.00 exactly?
    """)
    return


@app.cell(hide_code=True)
def _(
    DEMO_EDGES,
    backend,
    circuit_is_todo,
    cost_is_todo,
    expected_cut,
    np,
    optimize_qaoa,
    plt,
    qaoa_circuit,
    ref_cut,
    run_counts,
):
    if circuit_is_todo(qaoa_circuit):
        print("⚠️  p-sweep skipped: your `qaoa_circuit` TODOs aren't filled in")
        print("    yet, so every depth would flatline at ratio 2.5/4 ≈ 0.63 —")
        print(
            "    the blind-guessing floor. There's nothing to sweep until the"
        )
        print(
            "    machine has its stamp and its mixer. Build them, then rerun."
        )
        _out = None
    else:
        _cost = maxcut_cost if not cost_is_todo(maxcut_cost) else ref_cut
        _ratios, _prev = [], None
        for _p in range(1, 5):
            _ramp = np.array([(_k + 0.5) / _p for _k in range(_p)])
            _inits = [np.concatenate([0.7 * _ramp, 0.7 * (1 - _ramp)])]
            if _prev is None:
                _inits.append(np.array([2.85, 1.89]))  # your hand-found peak
            else:
                _inits.append(  # warm start: stretch last depth's best knobs
                    np.concatenate(
                        [
                            _prev[: _p - 1],
                            _prev[_p - 2 : _p - 1],
                            _prev[_p - 1 :],
                            _prev[-1:],
                        ]
                    )
                )
            _x, _ = optimize_qaoa(
                4, DEMO_EDGES, _p, _inits, qaoa_circuit, _cost
            )
            _counts = run_counts(
                qaoa_circuit(4, DEMO_EDGES, _x[:_p], _x[_p:]), backend, 4096
            )
            _val = expected_cut(_counts, DEMO_EDGES, _cost)
            _ratios.append(_val / 4)
            _prev = _x
            print(f"p = {_p}: best ⟨C⟩ = {_val:.2f} → ratio {_val / 4:.2f}")

        print("Depth buys quality — but never a guarantee: the wiggles are")
        print("shot noise plus optimizer luck on an ever-ridgier terrain.")

        _fig, _ax = plt.subplots(figsize=(8, 4))
        _ax.plot(
            range(1, 5), _ratios, "o-", color="#0E7C86", lw=2, markersize=8
        )
        _ax.axhline(
            1.0, color="#D48F26", ls="--", label="perfect chart (ratio 1.0)"
        )
        _ax.axhline(
            2.5 / 4, color="#6B7280", ls=":", label="blind guessing (0.625)"
        )
        _ax.set_xticks(range(1, 5))
        _ax.set_xlabel("QAOA depth p (layers)")
        _ax.set_ylabel("approximation ratio  ⟨C⟩ / 4")
        _ax.set_ylim(0.5, 1.05)
        _ax.set_title("depth buys quality: your machine, p = 1…4")
        _ax.grid(True, alpha=0.3)
        _ax.legend()
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Stretch check** — before moving on:

    - Did the optimizer's p = 1 verdict match your hand-found peak (or
      a mirror twin of it)?
    - Ratio ≈ 0.81 at p = 1 climbing to ≈ 0.95+ by p = 3–4 — but not a
      clean staircase. Why can adding layers ever *fail* to help?
      (More knobs = ridgier terrain = the optimizer can park on a
      local hill. The ceiling rises; finding it gets harder.)
    - Say the loop out loud, once, to a peer: *quantum circuit reports
      ⟨C⟩, classical optimizer proposes knobs, repeat.* If you can say
      it, you own the word "variational".

    Machine built, knobs understood. Time to feed it the real graph.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🎁 Bonus — measure a real gradient (the parameter-shift rule)": mo.md(r"""
    COBYLA steered by *scores alone*. But the morning's loophole slide
    said a circuit can report its **own exact gradient**: the response to
    a rotation knob is a perfect sinusoid, and a sinusoid's slope is
    pinned down by two samples a quarter-turn apart:

    $$\frac{\partial \langle C\rangle}{\partial \theta} = \tfrac{1}{2}\big[\langle C\rangle(\theta + \tfrac{\pi}{2}) - \langle C\rangle(\theta - \tfrac{\pi}{2})\big]$$

    The cell below checks it on the cleanest possible case — Tuesday's
    one-qubit $R_y(\theta)$ circuit, where theory says
    $P(1) = \sin^2(\theta/2)$, so the true gradient is
    $\tfrac{1}{2}\sin\theta$. Two shifted runs, a subtraction — and the
    estimate should sit on the exact value (within shot noise).

    (Why not on the QAOA knobs directly? A knob that appears in *many*
    gates — γ presses one stamp per edge — needs one shifted pair per
    appearance, summed. Same rule, more bookkeeping; the one-qubit case
    shows the miracle without it.)
    """),
        }
    )
    return


@app.cell
def _(QuantumCircuit, backend, np, transpile):
    # Bonus (given): parameter-shift on one Ry knob — exact gradient from
    # two shifted runs. Compare against the analytic 0.5·sin(θ).
    _theta = 1.1
    _shots = 4096

    def _p1(angle):
        qc = QuantumCircuit(1, 1)
        qc.ry(angle, 0)
        qc.measure(0, 0)
        counts = (
            backend.run(transpile(qc, backend), shots=_shots)
            .result()
            .get_counts()
        )
        return counts.get("1", 0) / _shots

    _plus = _p1(_theta + np.pi / 2)
    _minus = _p1(_theta - np.pi / 2)
    _shift_grad = 0.5 * (_plus - _minus)
    _exact = 0.5 * np.sin(_theta)

    print(
        f"θ = {_theta}  ·  P(1) at θ+π/2 = {_plus:.4f}  ·  at θ−π/2 = {_minus:.4f}"
    )
    print(f"parameter-shift gradient : {_shift_grad:+.4f}")
    print(f"exact  0.5·sin(θ)        : {_exact:+.4f}")
    print(
        f"difference               : {abs(_shift_grad - _exact):.4f}  (shot noise only)"
    )
    print(
        "Two runs of the SAME circuit, one subtraction — an exact derivative,"
    )
    print("no calculus performed anywhere. The circuit differentiated itself.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3 · The class chart, for real (aspirational)

    No more warm-ups. The cell below holds **this cohort's friendship
    graph** — one node per person, one edge per pair that already knows
    each other. It ships with our best guess from the intake forms
    (shared universities, neighbouring majors); **it's an editable,
    plain-Python list** — when the class calls out its real
    friendships, type them in and rerun. Everything downstream
    recomputes.

    The output of this section is not a toy: it's **the seating chart
    for the rest of the afternoon**. Ten people, 2¹⁰ = 1,024 possible
    charts — the biggest Max-Cut in this course, and yours.
    """)
    return


@app.cell
def _():
    # ── THE CLASS GRAPH · EDIT ME LIVE ───────────────────────────────────────
    # Replace with the class's self-reported friendships and rerun — every
    # cell below recomputes. The default edges are guessed from the intake
    # forms (shared universities, neighbouring majors).
    NAMES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]

    FRIENDSHIPS = [
        ("S1", "S3"),  # the UIUC trio…
        ("S1", "S4"),
        ("S3", "S4"),
        ("S1", "S5"),  # …the CompE trio…
        ("S1", "S7"),
        ("S5", "S7"),
        ("S2", "S5"),  # EE ↔ CompE (circuits people find each other)
        ("S2", "S7"),
        ("S9", "S5"),  # CS ↔ CompE
        ("S8", "S4"),  # MechE ↔ MatSci (lab-bench neighbours)
        ("S6", "S3"),  # EnvSci ↔ ChemE
        ("S6", "S10"),
        ("S9", "S10"),
    ]

    # name pairs → index pairs (node i ↔ qubit i ↔ bit position i)
    CLASS_EDGES = [(NAMES.index(a), NAMES.index(b)) for a, b in FRIENDSHIPS]
    return CLASS_EDGES, FRIENDSHIPS, NAMES


@app.cell(hide_code=True)
def _(CLASS_EDGES, NAMES, draw_graph):
    draw_graph(
        NAMES,
        CLASS_EDGES,
        title=f"the class graph · {len(NAMES)} people · "
        f"{len(CLASS_EDGES)} friendships",
        figsize=(6.2, 6.2),
    )
    return


@app.cell(hide_code=True)
def _(FRIENDSHIPS, mo):
    mo.md(f"""
    ### Predict

    Before the machine runs, bound it by eye (with the default graph):

    1. Can all {len(FRIENDSHIPS)} friendships cross? Count the **triangles** —
       `S1–S3–S4` and `S1–S5–S7`. Each triangle keeps at least one
       friendship un-cut, so the ceiling is at most… what?
    2. Blind guessing averages exactly half the edges — 6.5. Where will
       QAOA's ⟨C⟩ land: below 6.5, around 8–9, or pinned at the
       maximum?
    3. 1,024 charts is still brute-forceable, so this time we get to
       *grade* the quantum machine against certain truth. What
       approximation ratio would impress you?
    """)
    return


@app.cell
def _(
    CLASS_EDGES,
    NAMES,
    backend,
    circuit_is_todo,
    cost_is_todo,
    np,
    optimize_qaoa,
    qaoa_circuit,
    qaoa_ref,
    ref_cut,
    run_counts,
):
    # ── The run: QAOA p = 3 on the class graph (~10 s of honest shots) ──────
    # Fallback guards: if a Section-1/2 TODO is still open, the run borrows
    # the reference engine so the seating chart ALWAYS lands — but it says
    # so, and finishing your build makes the chart genuinely yours.
    if circuit_is_todo(qaoa_circuit):
        class_engine = qaoa_ref
        print("⚠️  Running on the REFERENCE machine — your `qaoa_circuit`")
        print("    TODOs aren't in yet (an all-H circuit guesses blindly).")
        print("    Finish Section 2 and rerun to make the chart YOURS.")
    else:
        class_engine = qaoa_circuit
        print("Engine: YOUR qaoa_circuit. The chart below is genuinely yours.")
    if cost_is_todo(maxcut_cost):
        class_cost = ref_cut
        print("⚠️  Scoring with the reference cost — your `maxcut_cost` still")
        print("    scores everything 0 (Section 1 TODO).")
    else:
        class_cost = maxcut_cost

    _p = 3
    _ramp = np.array([(_k + 0.5) / _p for _k in range(_p)])
    _inits = [
        np.concatenate([0.7 * _ramp, 0.7 * (1 - _ramp)]),
        np.concatenate([0.5 * _ramp + 0.1, 0.6 * (1 - _ramp) + 0.1]),
    ]
    _x, _v = optimize_qaoa(
        len(NAMES), CLASS_EDGES, _p, _inits, class_engine, class_cost
    )
    class_counts = run_counts(
        class_engine(len(NAMES), CLASS_EDGES, _x[:_p], _x[_p:]), backend, 4096
    )
    class_best = max(
        class_counts,
        key=lambda b: (class_cost(b, CLASS_EDGES), class_counts[b]),
    )
    print(
        f"p = {_p}, knobs tuned: ⟨C⟩ = {_v:.2f} of {len(CLASS_EDGES)} · "
        f"best sampled chart scores {class_cost(class_best, CLASS_EDGES)} "
        f"— chart below."
    )
    return class_best, class_cost, class_counts


@app.cell(hide_code=True)
def _(CLASS_EDGES, NAMES, class_best, class_cost, draw_graph):
    draw_graph(
        NAMES,
        CLASS_EDGES,
        bits=class_best,
        title=f"the chart · {class_cost(class_best, CLASS_EDGES)} of "
        f"{len(CLASS_EDGES)} friendships cross",
        figsize=(6.2, 6.2),
    )
    return


@app.cell
def _(CLASS_EDGES, NAMES, class_best, class_cost, mo):
    _teal = [NAMES[_i] for _i in range(len(NAMES)) if class_best[_i] == "0"]
    _amber = [NAMES[_i] for _i in range(len(NAMES)) if class_best[_i] == "1"]
    _score = class_cost(class_best, CLASS_EDGES)
    mo.md(f"""
    # 🪑 The seating chart

    <span style="font-size:1.35em; color:#0E7C86; font-weight:bold">
    Team Teal &nbsp;·&nbsp; {" · ".join(_teal)}</span>

    <span style="font-size:1.35em; color:#D48F26; font-weight:bold">
    Team Amber &nbsp;·&nbsp; {" · ".join(_amber)}</span>

    **{_score} of {len(CLASS_EDGES)} friendships now bridge the two
    tables.** Computed by a quantum algorithm, on this cohort's own
    graph, and in force for the rest of the afternoon — go move your
    chairs. 🎉
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The honesty check — grading the machine

    Ten people is small enough that a `for` loop can check **all 1,024
    charts** and name the true optimum. So let's grade QAOA against
    certain truth, in public. (That we *can* is the point: at 60 people
    there are 2⁶⁰ ≈ 10¹⁸ charts — enumeration is dead, and the
    variational machine is one of the few games still running. Scale is
    where the classical crutch snaps.)
    """)
    return


@app.cell(hide_code=True)
def _(CLASS_EDGES, class_best, class_cost, class_counts, np, plt):
    # Brute force: every one of the 1,024 charts, scored by the same cost
    # function that scored the QAOA samples.
    _n = 10
    _all = [format(_k, f"0{_n}b") for _k in range(2**_n)]
    _scores = np.array([class_cost(_b, CLASS_EDGES) for _b in _all])
    _cmax = int(_scores.max())
    _n_opt = int((_scores == _cmax).sum())

    _shots = sum(class_counts.values())
    _ev = (
        sum(
            class_cost(_b, CLASS_EDGES) * _c for _b, _c in class_counts.items()
        )
        / _shots
    )
    _p_opt = (
        sum(
            _c
            for _b, _c in class_counts.items()
            if class_cost(_b, CLASS_EDGES) == _cmax
        )
        / _shots
    )
    _qaoa_best = class_cost(class_best, CLASS_EDGES)

    print(
        f"brute force (all 1,024): optimum = {_cmax} of {len(CLASS_EDGES)}"
        f" · {_n_opt} charts achieve it"
    )
    print(
        f"QAOA best sampled chart: {_qaoa_best}"
        f" → {_qaoa_best / _cmax:.0%} of optimal"
    )
    print(
        f"QAOA average sample ⟨C⟩: {_ev:.2f} → ratio {_ev / _cmax:.2f}"
        f"  (blind guessing: {_scores.mean() / _cmax:.2f})"
    )
    print(
        f"P(drawing an optimal chart): QAOA {_p_opt:.1%} vs blind "
        f"{_n_opt / 2**_n:.2%} — a ~{_p_opt / (_n_opt / 2**_n):.0f}× "
        f"concentration"
    )

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # left — score distribution: QAOA draws vs blind guessing, over the FULL
    # fixed domain 0..len(edges)
    _dom = np.arange(len(CLASS_EDGES) + 1)
    _qd = np.zeros(len(_dom))
    for _b, _c in class_counts.items():
        _qd[class_cost(_b, CLASS_EDGES)] += _c
    _qd /= _shots
    _bd = np.array([(_scores == _c).sum() / 2**_n for _c in _dom])
    _ax1.bar(_dom - 0.2, _qd, width=0.4, color="#0E7C86", label="QAOA draws")
    _ax1.bar(
        _dom + 0.2, _bd, width=0.4, color="#6B7280", label="blind guessing"
    )
    _ax1.axvline(
        _cmax, color="#D48F26", ls="--", lw=1.6, label=f"optimum = {_cmax}"
    )
    _ax1.set_xticks(_dom)
    _ax1.set_xlabel("chart score")
    _ax1.set_ylabel("probability")
    _ax1.set_title("where the samples land")
    _ax1.legend(fontsize=9)
    _ax1.grid(True, axis="y", alpha=0.3)

    # right — the leaderboard: top 20 charts of all 1,024, ranked by score
    # (a stated cap — the other 1,004 exist, they're just worse)
    _TOP = 20
    _order = np.argsort(-_scores, kind="stable")[:_TOP]
    _cols = ["#D48F26" if _scores[_i] == _cmax else "#0E7C86" for _i in _order]
    _ax2.bar(np.arange(_TOP), _scores[_order], color=_cols)
    _top_charts = [_all[_i] for _i in _order]
    if class_best in _top_charts:
        _rank = _top_charts.index(class_best)
        _ax2.annotate(
            "← your chart",
            (_rank + 0.3, _scores[_order[_rank]] - 0.3),
            fontsize=10,
            color="#B23B7B",
            fontweight="bold",
            rotation=90,
            va="top",
        )
    _ax2.axhline(
        _scores.mean(),
        color="#6B7280",
        ls=":",
        label=f"blind average = {_scores.mean():.1f}",
    )
    _ax2.set_xlabel(f"rank (top {_TOP} of all 1,024 charts)")
    _ax2.set_ylabel("score")
    _ax2.set_ylim(0, len(CLASS_EDGES))
    _ax2.set_title("the leaderboard · optimal charts in amber")
    _ax2.legend(fontsize=9)
    _ax2.grid(True, axis="y", alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Aspirational check** — if you're here:

    - Did your triangle bound hold? (Two triangles → at most 11 of 13
      can cross, and brute force confirms 11 *is* the optimum on the
      default graph.)
    - Read your own grade honestly: QAOA's best draw typically *finds*
      an optimal chart, its average draw sits around 70–80% of optimal,
      and it pulls optimal charts ~5–20× more often than blind
      guessing. Not magic — a **bias machine**, tilted toward good
      answers. (Rerun the cell and watch the grade move: COBYLA
      sometimes parks on a lesser hill. That wobble *is* the morning's
      click line — finding the knobs is the whole game.)
    - The two honest caveats, out loud: (1) at this size brute force is
      *faster* — quantum buys nothing at n = 10; the bet is n = 60+,
      where 2ⁿ dies. (2) QAOA ships no optimality certificate — we only
      knew the ratio because we *could* brute-force. Living with that
      trade is what "heuristic" means.
    - **Bonus**: the class's real friendships are in by now — edit the
      `FRIENDSHIPS` cell, rerun, and hand the new chart to the
      instructor. Try p = 4 in the run cell while you're at it: does
      the extra layer earn its knobs on this graph?

    If your machine seated the class, you've run the full variational
    loop — problem in, circuit up, optimizer around it, answer out.
    That's the working skeleton of most near-term quantum computing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Wrap

    You built:
    - **A scorekeeper** — Max-Cut as seating charts, and `maxcut_cost`
      as the judge every later cell trusted.
    - **The QAOA machine** — a black-box cost stamp you could call but
      not open, your own mixer to cash the stamped phases into
      probabilities, and knobs an optimizer learned to turn: quantum
      circuit reports, classical optimizer proposes, repeat. *That's
      what "variational" means.*
    - **A real seating chart** — the cohort's own graph through your
      machine, graded honestly against brute force: roughly
      three-quarters of optimal on the average draw, an optimal chart
      in hand, and the class actually moved chairs.

    Yesterday interference found a needle; today it *tilted a
    haystack*. And the honest grade is the takeaway: heuristic, no
    certificate, pointless at n = 10 — but still standing at the
    scales where enumeration dies.

    Tomorrow: **where this runs for real** — VQE, QAOA's chemistry
    twin, with molecules instead of seating charts; real hardware over
    the cloud instead of a simulator; and the ticking clock that says
    when all of this stops being a classroom exercise.
    """)
    return


if __name__ == "__main__":
    app.run()
