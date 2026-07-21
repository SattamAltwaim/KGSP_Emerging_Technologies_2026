# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo","numpy","matplotlib","qiskit","qiskit-aer","pylatexenc"]
# ///
"""Day 3 lab · Marimo notebook · Grover search → iteration sweep → quantum walk

Tapered-tier structure per PEDAGOGY.md § Difficulty polarity:
    Baseline (everyone with TA support): Section 1 — Grover on N=4
    Stretch (most students):             Section 2 — Grover on N=16, find the peak
    Aspirational (top ~30%):             Section 3 — discrete-time quantum walk

Each tier runs the `explore → predict → build` micro-structure (decision #23):
an `mo.ui` widget to explore, a predict-then-run commitment, then a scaffolded
build. TAs cover mechanical snags (uv, imports, molab hiccups).

SINGLE NOTEBOOK (2026-07-21, replaces the lab.py/lab-student.py split):
solutions live HERE, collapsed in `mo.accordion` blocks under each build cell —
hidden by default, one click to open. The Explore widgets run on hidden
reference engines (`grover_demo`, `walk_demo`, hide_code cells) so exploration
works BEFORE the student builds anything; the student's own `grover_circuit` /
`walk_step` drive the check cells. The Grover oracle is a BLACK BOX: students
call `make_oracle(n, marked)` and get an opaque U_f gate (one closed box in
circuit drawings, matching the morning's framing); its internals are in an
accordion. The PEP 723 header lets molab / `marimo edit --sandbox` auto-install
qiskit + qiskit-aer.
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
    import matplotlib.pyplot as plt

    return AerSimulator, QuantumCircuit, mo, np, plt, transpile


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Day 3 lab · Grover + walks

    This morning you saw *two views* of Grover — the amplitude bars
    growing on the marked item, and the same story as a rotation in the
    |s⟩ / |s⊥⟩ plane. √N was not a lucky trick: it fell out of composing
    two reflections. This afternoon you'll build it, in **Qiskit**, and
    watch the marked amplitude concentrate one iteration at a time.

    **How to use this notebook**:
    - Cells run top-to-bottom. Try each in order.
    - Each section follows the same rhythm: **explore** (move a slider,
      watch the pictures), **predict** (commit an answer *before* you
      run), **build** (write the circuit from the recipe).
    - Stuck? Every build has a **💡 Solution** fold right under it —
      closed by default. Predict first, peek last.
    - No take-home. Anything you don't finish now, we don't chase later.

    **Three sections**:
    1. **Grover on 4 items** — baseline. Everyone finishes.
    2. **Grover on 16 items** — stretch. Find the iteration peak.
    3. **Quantum walk on a ring** — aspirational. Reach it and you've
       done the hardest thing in the room.
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
def _(QuantumCircuit):
    # ── THE BLACK BOX · given ────────────────────────────────────────────────
    # The oracle U_f, exactly as framed in the morning: a gate you may CALL,
    # not open. `make_oracle(n, marked)` returns one opaque box that flips the
    # PHASE of |marked⟩ and does nothing else. How it works inside is in the
    # "How is U_f built?" fold below — finish your build before opening it.
    def make_oracle(n_qubits: int, marked: str):
        """The black box U_f: flips the phase of |marked⟩, nothing else.
        Returned as an opaque gate — call it with
        `qc.append(make_oracle(n, marked), range(n))`."""
        assert len(marked) == n_qubits, "marked bitstring must match n_qubits"
        _qc = QuantumCircuit(n_qubits, name="U_f")
        _last = n_qubits - 1
        for _q, _bit in enumerate(reversed(marked)):
            if _bit == "0":
                _qc.x(_q)
        _qc.h(_last)
        _qc.mcx(list(range(_last)), _last)
        _qc.h(_last)
        for _q, _bit in enumerate(reversed(marked)):
            if _bit == "0":
                _qc.x(_q)
        return _qc.to_gate(label="U_f")

    return (make_oracle,)


@app.cell(hide_code=True)
def _(QuantumCircuit, make_oracle):
    # Reference engine for the EXPLORE widgets only — it lets you explore real
    # Grover behaviour BEFORE you've built your own. Your `grover_circuit` in
    # the Build step is the one the check cells use. No peeking needed.
    def grover_demo(
        n_qubits: int, marked: str, iterations: int
    ) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits, n_qubits)
        last = n_qubits - 1
        for q in range(n_qubits):
            qc.h(q)
        for _ in range(iterations):
            qc.append(make_oracle(n_qubits, marked), range(n_qubits))
            for q in range(n_qubits):
                qc.h(q)
                qc.x(q)
            qc.h(last)
            qc.mcx(list(range(last)), last)
            qc.h(last)
            for q in range(n_qubits):
                qc.x(q)
                qc.h(q)
        qc.measure(range(n_qubits), range(n_qubits))
        return qc

    return (grover_demo,)


@app.cell(hide_code=True)
def _(np, plt):
    # Side-by-side view: measured histogram + the morning's subspace picture.
    def grover_view(counts, n_qubits, marked, iterations, shots):
        """Left: probability per basis state (marked in amber).
        Right: the |s⟩/|s⊥⟩ plane — |ψ⟩ at angle (2k+1)θ, θ = arcsin(1/√N)."""
        N = 2**n_qubits
        theta = np.arcsin(1 / np.sqrt(N))
        ang = (2 * iterations + 1) * theta

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.4, 1]}
        )

        # left — the histogram
        keys = [format(i, f"0{n_qubits}b") for i in range(N)]
        vals = [counts.get(k, 0) / shots for k in keys]
        cols = ["#D48F26" if k == marked else "#0E7C86" for k in keys]
        ax1.bar(keys, vals, color=cols)
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel("measured probability")
        ax1.set_title(f"{shots} shots · marked = |{marked}⟩")
        if N > 8:
            ax1.tick_params(axis="x", rotation=90)
        ax1.grid(True, axis="y", alpha=0.3)

        # right — the subspace picture
        c = np.linspace(0, 2 * np.pi, 200)
        ax2.plot(np.cos(c), np.sin(c), color="#9a9a9a", lw=1, alpha=0.6)
        ax2.axhline(0, color="#1b2a4a", lw=1)
        ax2.axvline(0, color="#1b2a4a", lw=1)
        ax2.annotate(
            r"$|s^\perp\rangle$", (1.02, 0.08), fontsize=13, color="#1b2a4a"
        )
        ax2.annotate(
            r"$|s\rangle$", (0.06, 1.04), fontsize=13, color="#D48F26"
        )
        # start direction (dashed) + θ arc + state arrow
        ax2.plot(
            [0, 1.12 * np.cos(theta)],
            [0, 1.12 * np.sin(theta)],
            ls="--",
            color="#0E7C86",
            lw=1,
            alpha=0.5,
        )
        arc = np.linspace(0, ang, 60)
        ax2.plot(
            0.35 * np.cos(arc), 0.35 * np.sin(arc), color="#D48F26", lw=1.5
        )
        ax2.arrow(
            0,
            0,
            0.92 * np.cos(ang),
            0.92 * np.sin(ang),
            head_width=0.06,
            length_includes_head=True,
            color="#0E7C86",
            lw=2.6,
        )
        ax2.annotate(
            r"$|\psi\rangle$",
            (1.14 * np.cos(ang + 0.16), 1.14 * np.sin(ang + 0.16)),
            fontsize=13,
            color="#0E7C86",
            ha="center",
            va="center",
        )
        ax2.set_title(
            f"$(2k+1)\\theta$ = (2·{iterations}+1)·{np.degrees(theta):.1f}° "
            f"= {np.degrees(ang):.0f}°\n"
            f"$\\sin^2$ of that = {np.sin(ang) ** 2:.0%} on the marked item"
        )
        ax2.set_xlim(-1.35, 1.35)
        ax2.set_ylim(-1.35, 1.35)
        ax2.set_aspect("equal")
        ax2.axis("off")

        fig.tight_layout()
        return fig

    return (grover_view,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1 · Grover on 4 items (baseline)

    Two qubits give **4** basis states — `00`, `01`, `10`, `11`. One of
    them is *marked*. A classical search checks them one at a time (up to
    4 peeks). Grover finds it with a single pass.

    The recipe is two reflections, repeated:
    1. **Hadamards on both qubits** → a uniform superposition over all 4.
    2. **Oracle** — the black box $U_f$: flip the *phase* of the marked
       state (nothing else).
    3. **Diffuser** — reflect every amplitude about their mean.
    4. **Measure.**

    For N=4 the optimal count is $\lfloor\frac{\pi}{4}\sqrt{4}\rfloor = 1$. At
    exactly **one** iteration Grover finds the N=4 marked item with
    *certainty* — 100%, not "usually". That is the payoff to watch for.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Explore

    The slider sets how many Grover iterations run on the N=4 circuit
    (marked item = `11`). **Two pictures, one state**: the histogram is
    what you'd measure; the plane on the right is the morning's rotation
    view — |ψ⟩ starts at θ from |s⊥⟩ and turns 2θ per iteration.

    Watch them move *together*: the marked bar peaks exactly when |ψ⟩
    points at |s⟩ — and slide past it to watch the overshoot in both
    pictures at once.
    """)
    return


@app.cell
def _(mo):
    iters4 = mo.ui.slider(
        start=0,
        stop=5,
        value=1,
        step=1,
        label="Grover iterations (N=4)",
        show_value=True,
    )
    iters4
    return (iters4,)


@app.cell
def _(backend, grover_demo, grover_view, iters4, transpile):
    _shots = 2048
    _qc = grover_demo(n_qubits=2, marked="11", iterations=iters4.value)
    _counts = (
        backend.run(transpile(_qc, backend), shots=_shots)
        .result()
        .get_counts()
    )

    _p = _counts.get("11", 0) / _shots
    print(
        f"Marked = |11⟩ · iterations = {iters4.value} · P(measure 11) = {_p:.1%}"
    )
    grover_view(_counts, 2, "11", iters4.value, _shots)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Predict

    Before you build anything, commit to two answers (say them out loud
    or jot them down — committing is the point):

    1. Once the oracle + diffuser are in place, **which** of the four bars
       (`00`, `01`, `10`, `11`) should tower over the others at 1
       iteration?
    2. **Roughly what probability** will that bar reach? Pick one: ~25%,
       ~50%, ~75%, ~100%.

    (N=4 is the special case where Grover is *exact*. Hold that guess.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build

    Complete `grover_circuit` below.

    **The oracle is given** — it's the black box from the lecture. You
    *call* it, you don't open it:
    `qc.append(make_oracle(n_qubits, marked), range(n_qubits))` drops one
    opaque $U_f$ box into the circuit (you'll see it drawn as exactly
    that — one closed box).

    **You build the diffuser** (reflect about the mean): `H` on all
    qubits, `X` on all, a multi-controlled-Z, `X` on all, `H` on all.
    A controlled-Z on the last qubit is `H · mcx(others → last) · H`.

    **Why that sandwich works** — you already own a gate that phase-flips
    *one basis state* (the MCZ). Nobody sells a "reflect about the mean"
    gate. But the mean direction is just the uniform state $|u\rangle$,
    and `H` on every qubit *rotates* $|u\rangle$ onto the single state
    $|0\ldots0\rangle$. So: rotate (`H`s), phase-flip that one state (the
    `X`-wrap turns the MCZ's $|1\ldots1\rangle$ into $|0\ldots0\rangle$),
    rotate back (`H`s again). A reflection you can't do directly, done in
    a basis where it's one gate — the full $a \to 2m - a$ algebra is in
    the fold below the code.
    """)
    return


@app.cell
def _(QuantumCircuit, make_oracle):
    def grover_circuit(
        n_qubits: int, marked: str, iterations: int
    ) -> QuantumCircuit:
        """Grover search on `n_qubits` qubits, marking bitstring `marked`,
        running `iterations` Grover steps, then measuring all qubits."""
        assert len(marked) == n_qubits, "marked bitstring must match n_qubits"
        qc = QuantumCircuit(n_qubits, n_qubits)
        last = n_qubits - 1

        # (1) Uniform superposition over all 2**n basis states. (Given.)
        for q in range(n_qubits):
            qc.h(q)

        for _ in range(iterations):
            # (2) Oracle — the black box, called as a function. (Given.)
            qc.append(make_oracle(n_qubits, marked), range(n_qubits))

            # (3) TODO — DIFFUSER: reflect all amplitudes about their mean.
            #   Recipe: qc.h + qc.x on every qubit; the multi-controlled-Z
            #   (qc.h(last); qc.mcx(list(range(last)), last); qc.h(last)); then
            #   qc.x + qc.h on every qubit (to undo the wrap).
            pass

        # (4) Measure. (Given — leave this so the cell runs while you build.)
        qc.measure(range(n_qubits), range(n_qubits))
        return qc

    return (grover_circuit,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Solution — the diffuser (open if stuck)": mo.md(r"""
    Replace the `TODO` block (and the `pass`) with:

    ```python
    for q in range(n_qubits):
        qc.h(q)
        qc.x(q)
    qc.h(last)
    qc.mcx(list(range(last)), last)
    qc.h(last)
    for q in range(n_qubits):
        qc.x(q)
        qc.h(q)
    ```

    `H`-then-`X` on every qubit maps the *uniform* state to `|1…1⟩`; the
    multi-controlled-Z flips the phase of exactly that state; undoing the
    wrap turns the whole thing into "flip the phase of the mean
    direction" — the reflection $a \to 2m - a$ from the paper quiz.
    """),
            "🧠 Why does that circuit compute a → 2m − a? (the algebra)": mo.md(r"""
    **Reflecting about the mean = reflecting about the uniform state.**
    Let $|u\rangle = \tfrac{1}{\sqrt N}\sum_x |x\rangle$. The diffuser is
    the operator

    $$D \;=\; 2|u\rangle\langle u| - I.$$

    **Check it really does $a \to 2m - a$:** write your state as
    $|\psi\rangle = \sum_x a_x |x\rangle$. Then
    $\langle u|\psi\rangle = \tfrac{1}{\sqrt N}\sum_x a_x = \sqrt N\, m$
    (that's where the *mean* $m$ enters — it's the overlap with the
    uniform direction). So

    $$D|\psi\rangle = 2\sqrt N m\,|u\rangle - |\psi\rangle
      \quad\Rightarrow\quad \text{amplitude on } |x\rangle
      \;=\; 2m - a_x. \checkmark$$

    **Why the gate sandwich implements $D$:** Hadamards rotate the uniform
    state onto a basis state — $H^{\otimes n}|u\rangle = |0\ldots0\rangle$
    — and $H^{\otimes n}$ is its own inverse, so

    $$D \;=\; H^{\otimes n}\,\big(2|0\ldots0\rangle\langle 0\ldots0| - I\big)\,H^{\otimes n}.$$

    That's *conjugation*: rotate to a basis where the reflection is easy,
    reflect, rotate back. The middle factor phase-flips every state
    *except* $|0\ldots0\rangle$ — which is the same as flipping *only*
    $|0\ldots0\rangle$, up to a global sign no measurement can see. And
    flipping one basis state is exactly what you own: the `X`-wrap swaps
    $|0\ldots0\rangle \leftrightarrow |1\ldots1\rangle$, the MCZ flips
    $|1\ldots1\rangle$, the `X`-wrap swaps back.

    (Same trick as the oracle, aimed at $|u\rangle$ instead of the marked
    item — Grover is two phase-flips in two different bases.)
    """),
            "🔍 How is the black box U_f built? (open after your build works)": mo.md(r"""
    The oracle has the same shape as the diffuser's core — an X-wrap
    around a multi-controlled-Z, aimed at `marked` instead of `|1…1⟩`:

    ```python
    def make_oracle(n_qubits, marked):
        qc = QuantumCircuit(n_qubits, name="U_f")
        last = n_qubits - 1
        # X-wrap the 0-bits so `marked` becomes |1…1⟩
        # (Qiskit is little-endian: qubit q holds bit reversed(marked)[q])
        for q, bit in enumerate(reversed(marked)):
            if bit == "0":
                qc.x(q)
        # multi-controlled-Z = H · mcx · H on the last qubit
        qc.h(last)
        qc.mcx(list(range(last)), last)
        qc.h(last)
        # unwrap
        for q, bit in enumerate(reversed(marked)):
            if bit == "0":
                qc.x(q)
        return qc.to_gate(label="U_f")   # ← packaged: one opaque box
    ```

    `to_gate(label="U_f")` is what makes it *look* like a black box in
    circuit drawings — Qiskit collapses the internals into a single named
    block, exactly how the morning slides drew it.
    """),
        }
    )
    return


@app.cell
def _(backend, grover_circuit, transpile):
    # Draw the N=4, 1-iteration circuit — the oracle appears as ONE closed
    # U_f box (the black box), the diffuser as the gates you wrote.
    _qc = grover_circuit(n_qubits=2, marked="11", iterations=1)
    _tc = transpile(_qc, backend)
    _counts = backend.run(_tc, shots=2048).result().get_counts()
    print(f"N=4, 1 iteration, marked |11⟩ → counts: {_counts}")
    if _counts.get("11", 0) / 2048 < 0.5:
        print(
            "⚠️  Still ~25% on every bar → your diffuser TODO isn't filled in yet."
        )
        print(
            "    The oracle alone flips a sign, and a sign changes NO probabilities"
        )
        print("    (the invisible mark!). Add the diffuser and rerun.")
    _qc.draw("mpl")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Baseline check** — before moving on:

    - Did the `11` bar hit ~100% at 1 iteration? (On the simulator it
      should be essentially certain — this is the N=4 magic.)
    - In the drawing, can you point at the two reflections? (The closed
      `U_f` box, and your diffuser gates after it.)
    - Change `marked` to `"01"` in the draw cell and rerun. Same
      behaviour on a different target?

    If the marked bar peaked: you've built the smallest complete Grover
    search there is. Move to Section 2.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2 · Grover on 16 items · find the peak (stretch)

    Four qubits = **16** basis states. Same two-reflection engine — you
    already built it, `grover_circuit` handles any `n_qubits`. What
    changes is the *timing*.

    Grover rotates the state a fixed angle per iteration. Too few and the
    marked amplitude hasn't arrived; too many and you swing *past* it
    (the overshoot you computed by hand this morning). The optimum is
    $\lfloor(\pi/4)\sqrt{N}\rfloor$.

    For N=16 that's $\lfloor(\pi/4)\cdot 4\rfloor = \lfloor\pi\rfloor = 3$.
    You'll sweep the iteration count and watch success probability rise to
    a peak and fall.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Explore

    Marked item is `1011`. Same two pictures: at N=16 the starting angle
    θ is *smaller* (sin θ = 1/4, so θ ≈ 14.5°) — watch |ψ⟩ creep toward
    |s⟩ in smaller turns, and note **where the marked bar is tallest**.
    """)
    return


@app.cell
def _(mo):
    iters16 = mo.ui.slider(
        start=0,
        stop=12,
        value=3,
        step=1,
        label="Grover iterations (N=16)",
        show_value=True,
    )
    iters16
    return (iters16,)


@app.cell
def _(backend, grover_demo, grover_view, iters16, transpile):
    _shots = 4096
    _marked = "1011"
    _qc = grover_demo(n_qubits=4, marked=_marked, iterations=iters16.value)
    _counts = (
        backend.run(transpile(_qc, backend), shots=_shots)
        .result()
        .get_counts()
    )

    _p = _counts.get(_marked, 0) / _shots
    print(
        f"Marked = |{_marked}⟩ · iterations = {iters16.value} · P = {_p:.1%}"
    )
    print(
        "Try 3 iterations, then 6, then 9. Watch bar and arrow move together."
    )
    grover_view(_counts, 4, _marked, iters16.value, _shots)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Predict

    Commit *before* you look at the plot:

    1. Compute $\lfloor(\pi/4)\sqrt{16}\rfloor$ by hand. That's your
       predicted **optimal iteration count**. Write it down.
    2. Sketch the shape of "success probability vs. iteration count" from
       0 to 12. Rising then falling? A single peak, or does it come back?

    The next cell draws the real curve — using **your** `grover_circuit`.
    See if your peak lands where you said.
    """)
    return


@app.cell
def _(backend, grover_circuit, np, plt, transpile):
    _marked = "1011"
    _max_iters = 12
    _shots = 4096

    _probs = []
    for _it in range(_max_iters + 1):
        _qc = grover_circuit(n_qubits=4, marked=_marked, iterations=_it)
        _tc = transpile(_qc, backend)
        _counts = backend.run(_tc, shots=_shots).result().get_counts()
        _probs.append(_counts.get(_marked, 0) / _shots)

    _opt = int(np.floor(np.pi / 4 * np.sqrt(16)))
    _best = int(
        np.argmax(_probs[:7])
    )  # first peak (past ~8 the curve revives)
    print(f"Predicted optimum floor(pi/4 * sqrt(16)) = {_opt}")
    print(
        f"Measured first peak at iteration = {_best}  (P = {_probs[_best]:.1%})"
    )
    print(f"Success probability by iteration: {[f'{p:.2f}' for p in _probs]}")
    if max(_probs) < 0.2:
        print(
            "⚠️  Flat at ~1/16 = 6% for every iteration count? That means your"
        )
        print(
            "    diffuser TODO in `grover_circuit` isn't filled in yet — the oracle"
        )
        print(
            "    alone only flips a sign, which no measurement can see. Once the"
        )
        print(
            "    diffuser is in, this curve peaks above 95% at 3 iterations."
        )

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(
        range(_max_iters + 1),
        _probs,
        "o-",
        color="#0E7C86",
        linewidth=2,
        markersize=7,
    )
    _ax.axvline(
        _opt,
        color="#D48F26",
        linestyle="--",
        label=f"theoretical optimum = {_opt}",
    )
    _ax.set_xlabel("Grover iterations")
    _ax.set_ylabel(f"P(measure {_marked})")
    _ax.set_ylim(0, 1.05)
    _ax.set_title("Grover overshoots: success probability peaks, then decays")
    _ax.grid(True, alpha=0.3)
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Stretch check** — before moving on:

    - Where's your peak? (It should sit at **3**, matching the formula.)
    - What's the success probability at the peak? (>95% on the simulator.)
    - By iteration ~6 the curve has crashed, then it climbs again near
      ~9–10. Why? (|ψ⟩ is *rotating around the circle* in the explore
      view. It swings toward |s⟩, past it, then back around — periodic,
      not monotonic.)
    - So "more iterations = better" is **false**. Explain that to a peer
      using the rotation picture.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3 · The treasure hunt — a quantum walk on a ring (aspirational)

    Grover found a marked item when *any* item could be checked at any
    moment. Now play the honest version: a **treasure hunt on a graph**.
    The treasure is buried at one node, and a walker can only move along
    *edges* — no teleporting between candidates. Mazes, timetable states,
    board positions: most real search spaces look like this.

    How fast you find the treasure is set by how fast you can *move
    through the graph*. That engine is the **quantum walk**: it spreads
    quadratically faster than a classical random walker — distance ~T
    instead of ~√T. (The full search algorithm — walk steps alternated
    with a treasure-checking oracle, Grover's graph cousin — is beyond
    today. You're building its engine, and racing it to the treasure.)

    **This section is genuinely hard.** Reaching a working walk means
    you're doing quantum algorithm design past the textbook toys.

    We'll build a discrete-time quantum walk (DTQW) on a small **ring of 8
    nodes**, with the treasure buried at **node 4 = |100⟩ — the farthest
    node from the start**. The walker has:
    - a **position** register — 3 qubits encode which of the 8 nodes,
    - a **coin** qubit — decides direction each step.

    One step = **coin flip** (`H` on the coin) then a **shift**: if the
    coin reads 1, step to the next node (`+1 mod 8`); if 0, step to the
    previous node (`-1 mod 8`). Repeat, then measure position.

    A classical walker's position after T steps is a binomial bump around
    the start. The quantum walker interferes with itself and spreads
    *ballistically* — two lobes racing outward.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The graph you're walking on

    Before any circuits: *this* is the problem. Eight nodes in a ring,
    each edge a legal move. Every node number is also a 3-qubit basis
    state — that's the entire encoding: **node ↔ bitstring ↔ position
    register**. The walker starts at node 0 = |000⟩ (amber); the treasure
    is buried at node 4 = |100⟩ (magenta), the far side of the ring.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):
    # The 8-node ring, with the node ↔ qubit-basis-state mapping drawn in.
    _fig, _ax = plt.subplots(figsize=(5.6, 5.6))
    _ang = (
        np.pi / 2 - 2 * np.pi * np.arange(8) / 8
    )  # node 0 at the top, clockwise
    _x, _y = np.cos(_ang), np.sin(_ang)
    for _i in range(8):  # edges i — (i+1) mod 8
        _j = (_i + 1) % 8
        _ax.plot(
            [_x[_i], _x[_j]],
            [_y[_i], _y[_j]],
            color="#9a9a9a",
            lw=1.6,
            zorder=1,
        )
    _ax.scatter(
        _x, _y, s=1500,
        c=["#D48F26"] + ["#0E7C86"] * 3 + ["#B23B7B"] + ["#0E7C86"] * 3,
        zorder=2,
    )
    for _i in range(8):
        _ax.annotate(
            f"{_i}\n|{_i:03b}⟩",
            (_x[_i], _y[_i]),
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
            zorder=3,
        )
    _ax.annotate(
        "start",
        (_x[0], _y[0] + 0.22),
        ha="center",
        fontsize=11,
        color="#D48F26",
        fontweight="bold",
    )
    _ax.annotate(
        "★ treasure",
        (_x[4], _y[4] - 0.24),
        ha="center",
        fontsize=11,
        color="#B23B7B",
        fontweight="bold",
    )
    _ax.set_xlim(-1.45, 1.45)
    _ax.set_ylim(-1.5, 1.5)
    _ax.set_aspect("equal")
    _ax.axis("off")
    _ax.set_title("8-node ring · node number ↔ 3-qubit basis state")
    _fig
    return


@app.cell(hide_code=True)
def _(QuantumCircuit):
    # Reference engine for the walk EXPLORE widget (same deal as grover_demo):
    # a full working walk so you can explore before you build. Your own
    # `walk_step` below is what the payoff plot uses.
    def walk_demo(t_steps: int) -> QuantumCircuit:
        inc = QuantumCircuit(3, name="+1")
        inc.ccx(0, 1, 2)
        inc.cx(0, 1)
        inc.x(0)
        plus = inc.to_gate().control(1)
        minus = inc.to_gate().inverse().control(1, ctrl_state=0)
        qc = QuantumCircuit(4, 4)
        p0, p1, p2, c = 0, 1, 2, 3
        for _ in range(t_steps):
            qc.h(c)
            qc.append(plus, [c, p0, p1, p2])
            qc.append(minus, [c, p0, p1, p2])
            qc.barrier()
        qc.measure(range(3), range(3))
        return qc

    return (walk_demo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Explore

    The slider sets the number of walk steps T. Watch how the position
    distribution spreads over the ring's nodes as T grows — and watch the
    **treasure node (magenta bar)**: at what T does real probability first
    arrive on it?
    """)
    return


@app.cell
def _(mo):
    steps = mo.ui.slider(
        start=0,
        stop=8,
        value=4,
        step=1,
        label="Quantum-walk steps T",
        show_value=True,
    )
    steps
    return (steps,)


@app.cell
def _(backend, np, plt, steps, transpile, walk_demo):
    _shots = 4096
    _qc = walk_demo(steps.value)
    _tc = transpile(_qc, backend)
    _counts = backend.run(_tc, shots=_shots).result().get_counts()

    # Marginalise onto the 3 position bits — ALL 8 nodes, zeros included, so
    # the histogram keeps the same bars in the same places at every T.
    _pos = np.zeros(8)
    for _bits, _c in _counts.items():
        _pos[int(_bits[-3:], 2)] += _c
    _pos /= _shots

    print(f"Quantum walk, T = {steps.value} steps, start at node 0.")
    print(f"P(standing on the treasure, node 4) = {_pos[4]:.1%}")
    print("Not a single bump around 0 — the quantum walk spreads into lobes.")
    print(
        "(Only every-other node is reachable at a given T — each step moves ±1,"
    )
    print(" so the walker's node parity flips every step.)")
    _fig, _ax = plt.subplots(figsize=(8, 3.8))
    _x = np.arange(8)
    _cols = ["#0E7C86"] * 8
    _cols[4] = "#B23B7B"
    _ax.bar(_x, _pos, color=_cols)
    _ax.set_xticks(_x)
    _ax.set_xticklabels(
        [f"{_i}\n|{_i:03b}⟩" + ("\n★" if _i == 4 else "") for _i in range(8)]
    )
    _ax.set_ylim(0, 1.0)
    _ax.set_ylabel("probability")
    _ax.set_title(f"position after T = {steps.value} steps · ★ = treasure")
    _ax.grid(True, axis="y", alpha=0.3)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Predict

    Before you build the shift operator, predict:

    1. A *classical* walker on this ring, starting at node 0 after 4 coin
       flips, is most likely found **where**? (Look at the ring drawing.)
    2. The *quantum* walker's distribution — one central bump, or
       something else? Commit to a sketch.
    3. The treasure sits 4 edges away. **Which walker is standing on it
       first, and by roughly what margin?** Commit to a guess — the race
       plot at the end settles it.

    Then build the shift and see which prediction the histogram matches.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build

    The hard part is the **shift**: `+1 mod 8` when the coin reads 1,
    `-1 mod 8` when it reads 0. On the ring drawing, `+1` is one step
    clockwise.

    You only have to build **one small thing** — a plain `+1 mod 8` on the
    3 position qubits, no coin anywhere. It's a ripple of flips, MSB
    first: `inc.ccx(0, 1, 2)`, `inc.cx(0, 1)`, `inc.x(0)` ("flip the top
    bit when both lower bits carry, then the middle, then the bottom").

    Qiskit does the rest *for* you:

    - `inc.to_gate()` packages your circuit as a gate;
    - `gate.inverse()` **is** the `-1 mod 8` — no hand-reversing gates;
    - `gate.control(1)` returns a version that fires when a control qubit
      is **1**, and `gate.control(1, ctrl_state=0)` one that fires when
      it's **0** — no `X`-wrap needed.

    One walk step = coin flip, then append both controlled shifts. When
    appending, the **control qubit goes first** in the qubit list:
    `qc.append(plus, [c, p0, p1, p2])`.
    """)
    return


@app.cell
def _(QuantumCircuit):
    N_POS = 3  # position qubits → ring of 2**3 = 8 nodes
    COIN = 3  # coin qubit index (after the 3 position qubits)

    def walk_step(qc: QuantumCircuit) -> None:
        """One discrete-time quantum-walk step on the 8-node ring.
        Coin flip, then coin-controlled +1/-1 (mod 8) on the position."""
        p0, p1, p2 = 0, 1, 2
        c = COIN

        # (1) Coin flip. (Given.)
        qc.h(c)

        # (2) TODO — build the plain +1 mod 8 as its own tiny circuit:
        inc = QuantumCircuit(3, name="+1")
        #   …the MSB-first ripple: ccx(0,1,2), cx(0,1), x(0)…
        inc_gate = inc.to_gate()

        # (3) TODO — let qiskit add the coin control, and append both shifts
        #   (control qubit first in the list!):
        #   coin==1 → +1:  inc_gate.control(1)
        #   coin==0 → −1:  inc_gate.inverse().control(1, ctrl_state=0)
        pass

    def walk_circuit(t_steps: int) -> QuantumCircuit:
        """Start at node 0 (all position qubits |0⟩), coin |0⟩, run t_steps,
        measure the 3 position qubits into the low 3 classical bits."""
        qc = QuantumCircuit(N_POS + 1, N_POS + 1)
        for _ in range(t_steps):
            walk_step(qc)
            qc.barrier()
        qc.measure(
            range(N_POS), range(N_POS)
        )  # position → classical bits 0..2
        return qc

    return (walk_circuit,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "💡 Solution — the coined shift (open if stuck)": mo.md(r"""
    Replace the two `TODO` blocks (and the `pass`) with:

    ```python
    # (2) A plain +1 mod 8 on the 3 position qubits (MSB-first ripple).
    inc = QuantumCircuit(3, name="+1")
    inc.ccx(0, 1, 2)   # p0 & p1 → flip p2   (the carry into the top bit)
    inc.cx(0, 1)       # p0      → flip p1
    inc.x(0)           #           flip p0
    inc_gate = inc.to_gate()

    # (3) Qiskit adds the coin control; the control qubit goes first.
    qc.append(inc_gate.control(1), [c, p0, p1, p2])                        # coin==1 → +1
    qc.append(inc_gate.inverse().control(1, ctrl_state=0), [c, p0, p1, p2])  # coin==0 → −1
    ```

    The increment is a ripple-carry: flip the MSB only when both lower
    bits are 1 (that's the carry), then the middle bit, then the LSB.
    From there qiskit's helpers carry the load: `.inverse()` runs the same
    gates backwards (a decrement — every gate here is its own inverse),
    `.control(1)` bolts a control qubit onto the whole gate, and
    `ctrl_state=0` makes it fire on coin = 0 instead of 1 (the `X`-wrap
    you'd otherwise write by hand, done for you).
    """),
        }
    )
    return


@app.cell
def _(mo):
    t_compare = mo.ui.slider(
        start=0,
        stop=8,
        value=6,
        step=1,
        label="steps T · the race to the treasure (quantum = yours)",
        show_value=True,
    )
    t_compare
    return (t_compare,)


@app.cell
def _(backend, np, plt, t_compare, transpile, walk_circuit):
    # Quantum walk vs. classical walk after T steps — the payoff plot.
    # (Uses YOUR walk_step — flat until your shift works.) Scrub T and watch
    # the two walkers race: classical diffuses (~√T), quantum runs (~T).
    _t = t_compare.value
    _shots = 8192

    _qc = walk_circuit(_t)
    _tc = transpile(_qc, backend)
    _counts = backend.run(_tc, shots=_shots).result().get_counts()
    _q_dist = np.zeros(8)
    for _bits, _cnt in _counts.items():
        _q_dist[int(_bits[-3:], 2)] += _cnt
    _q_dist /= _shots

    # Classical lazy walk on the same ring: +/-1 each step with prob 1/2.
    _c_dist = np.zeros(8)
    _c_dist[0] = 1.0
    for _ in range(_t):
        _c_dist = 0.5 * np.roll(_c_dist, 1) + 0.5 * np.roll(_c_dist, -1)

    print(f"After T={_t} steps from node 0 — who's standing on the treasure (node 4)?")
    print(f"  quantum: {_q_dist[4]:.1%}   classical: {_c_dist[4]:.1%}")
    if _q_dist[0] > 0.95 and _t > 0:
        print(
            "⚠️  All mass still on node 0 → the shift TODOs in `walk_step` aren't"
        )
        print("    filled in yet. The walker has a coin but no legs.")
    elif _t >= 4:
        print(
            "  The quantum walker runs (distance ~T); the classical one dawdles (~√T)."
        )

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _x = np.arange(8)
    _ax.axvspan(3.55, 4.45, color="#B23B7B", alpha=0.10, zorder=0)
    _ax.bar(
        _x - 0.18,
        _q_dist,
        width=0.36,
        color="#0E7C86",
        label="quantum walk (yours)",
    )
    _ax.bar(
        _x + 0.18, _c_dist, width=0.36, color="#D48F26", label="classical walk"
    )
    _ax.annotate("★ treasure", (4, 0.93), ha="center", fontsize=11,
                 color="#B23B7B", fontweight="bold")
    _ax.set_xlabel("node")
    _ax.set_ylabel("probability")
    _ax.set_ylim(0, 1.0)
    _ax.set_title(f"The race to the treasure · T={_t} steps")
    _ax.set_xticks(_x)
    _ax.set_xticklabels([f"{i}\n|{i:03b}⟩" for i in range(8)])
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Aspirational check** — if you're here:

    - Does your `+1 mod 8` shift wrap cleanly (node 7 → node 0)? Test it
      alone: one step with the coin forced to `1` should move all mass
      from node 0 to node 1.
    - After several steps, is the quantum distribution *bimodal* (two
      lobes) while the classical one stays a single bump at the start?
    - At T=6, who's standing on the treasure — and by what margin? (Your
      walker should be at ~60%+ on node 4 while the classical one is
      still under ~20%.)
    - Can you explain to a peer why the quantum walker spreads faster?
      (Amplitudes interfere; the coin keeps the walker's "momentum"
      instead of re-randomising it each step.)

    If your shift works and your walker wins the race, you've built a
    working discrete-time quantum walk — genuinely grad-school material.
    That's the ceiling for today. The real *search* algorithm on a graph
    alternates walk steps with a treasure-checking oracle (exactly the
    U_f idea again) and finds the marked node quadratically faster —
    Grover's cousin, with your walk as its engine.

    **Bonus**: replace the `H` coin with a different coin (try `Ry` at
    some angle) and watch the lobes go asymmetric.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Wrap

    You built:
    - **Grover on N=4** — a black-box oracle you could call but not open,
      your own diffuser, and the marked item with certainty in one
      iteration.
    - **Grover on N=16** — the same engine, and the peak-then-decay curve
      that shows why √N iterations is the *right* number, not a minimum.
    - **A quantum walk** — Grover's interference trick carried onto a
      graph you could actually see, racing a classical walker to buried
      treasure and getting there quadratically sooner.

    Interference was the engine the whole time. √N is the geometry of a
    rotation.

    Tomorrow: **variational algorithms** — where quantum circuits meet the
    modern-ML playbook: parameters, expectation values, a classical
    optimizer in the loop.
    """)
    return


if __name__ == "__main__":
    app.run()
