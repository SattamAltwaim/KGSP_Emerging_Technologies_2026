# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "qiskit",
#     "qiskit-aer",
# ]
# ///
"""Day 2 lab · STUDENT version (no solutions) · quantum coin → GHZ → teleportation

Section 1 (quantum coin) is worked as the example. Sections 2 (GHZ) and 3
(teleportation) are scaffolded — you write the circuit from the recipe in each
section's text. The full solutions live in lab.py (TA copy).

Tapered-tier structure per PEDAGOGY.md § Difficulty polarity:
    Baseline (everyone with TA support): Section 1 — quantum coin
    Stretch (most students):             Section 2 — GHZ state
    Aspirational (top ~30%):             Section 3 — teleportation

The PEP 723 header above lets molab / `marimo edit --sandbox` auto-install
qiskit + qiskit-aer. See PEDAGOGY.md for the TA/Wael split.
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
    from qiskit.visualization import plot_histogram
    import matplotlib.pyplot as plt

    return AerSimulator, QuantumCircuit, mo, np, plot_histogram, transpile


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Day 2 lab · Qubits in code

    This morning you met ψ and φ, watched the collapse animation, and
    saw teleportation move a state without ever copying it. This
    afternoon you'll build all three — in **Qiskit**, in **Python**,
    one line at a time.

    **How to use this notebook**:
    - Cells run top-to-bottom. Try each in order.
    - Green button = "safe to click." Amber button = "uses real quantum
      hardware, one shot only, wait for the queue."
    - No take-home. Anything you don't finish now, we don't chase later.

    **Three sections**:
    1. **Quantum coin** — baseline. Everyone finishes.
    2. **GHZ state** — stretch. Most finish.
    3. **Teleportation** — aspirational. Reach it and you've done the
       hardest thing in the room.
    """)
    return


@app.cell
def _():
    # Backend selector. Leave this False on Colab — the local simulator is fast,
    # deterministic, and needs no queue. Flip to True only if you've linked an
    # IBM Quantum account and filled in the service-load in the next cell.
    USE_REAL_HW = False
    return (USE_REAL_HW,)


@app.cell
def _(AerSimulator, USE_REAL_HW):
    if not USE_REAL_HW:
        backend = AerSimulator()
        print("Using local simulator (AerSimulator). Fast, deterministic, no queue.")
    else:
        # To use real hardware, link your IBM Quantum account and uncomment:
        #   from qiskit_ibm_runtime import QiskitRuntimeService
        #   service = QiskitRuntimeService()
        #   backend = service.least_busy(operational=True, simulator=False)
        print("Real hardware selected, but no service configured — using simulator.")
        backend = AerSimulator()  # fallback
    return (backend,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 1 · Quantum coin (baseline)

    One qubit. One Hadamard gate. One measurement. Run it many times,
    count how often you get 0 vs. 1.

    **Expected**: ~50/50, ± shot noise. This is ψ in a **fair
    superposition** — the split-pose you saw this morning.

    **Story of the cell below**: build the circuit, print it, look at
    the diagram. `H` is the Hadamard gate. `measure` collapses the
    state.
    """)
    return


@app.cell
def _(QuantumCircuit):
    coin = QuantumCircuit(1, 1)
    coin.h(0)
    coin.measure(0, 0)
    coin.draw("mpl")
    return (coin,)


@app.cell
def _(backend, coin, plot_histogram, transpile):
    _tc = transpile(coin, backend)
    _result = backend.run(_tc, shots=1024).result()
    _counts = _result.get_counts()

    print(f"Counts (1024 shots): {_counts}")
    print("Ratio should be close to 50/50 — that's ψ in superposition.")
    plot_histogram(_counts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Baseline check** — before moving on, ask yourself:

    - Is your split within a few percent of 50/50?
    - What would it mean if it wasn't? (Answer: hardware noise, if you
      ran on real hardware; a bug in your circuit, if you ran on the
      simulator.)

    If yes to the first: you've just built the simplest interesting
    quantum program that exists. Move to Section 2.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2 · GHZ state (stretch)

    Three qubits, all entangled. When you measure them, you should
    see **all three agree** — either `000` or `111`. Never a
    mixed outcome like `010`.

    This is a bigger version of yesterday's Bell state. The Bell state
    entangles two qubits; GHZ (Greenberger–Horne–Zeilinger) entangles
    three or more.

    **Task**: build the circuit yourself. The recipe is
    `H` on qubit 0, then `CNOT` from qubit 0 to qubit 1, then `CNOT`
    from qubit 0 to qubit 2, then measure all three.

    **Expected**: histogram should show two big peaks — `000` and
    `111` — with everything else near zero.
    """)
    return


@app.cell
def _(QuantumCircuit):
    ghz = QuantumCircuit(3, 3)

    # TODO — entangle the three qubits into a GHZ state (recipe is in the text
    # above). The gate methods you need:
    #     ghz.h(<qubit>)          — Hadamard
    #     ghz.cx(<control>, <target>)  — CNOT
    # Write those lines here, then the measure below stays as-is.

    ghz.measure([0, 1, 2], [0, 1, 2])
    ghz.draw("mpl")
    return (ghz,)


@app.cell
def _(backend, ghz, plot_histogram, transpile):
    _tc = transpile(ghz, backend)
    _result = backend.run(_tc, shots=1024).result()
    _counts = _result.get_counts()

    print(f"Counts (1024 shots): {_counts}")
    print("Two big peaks (000 and 111) + a few small ones "
          "(noise, if real hardware; near-zero if simulator).")
    plot_histogram(_counts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Stretch check** — before moving on:

    - Did you see `000` and `111` together account for >90% of shots
      (simulator) or >85% (real hardware)?
    - What would happen if you replaced one CNOT with a different
      gate? Try it in a new cell and see.
    - The state you built is
      $\ket{\Phi_{GHZ}} = \tfrac{1}{\sqrt{2}}(\ket{000} + \ket{111})$.
      Three qubits, one shared thing.

    **TA prompt** if a student is fast: ask them to explain WHY the
    histogram has no `001` or `100` peaks. The answer connects back
    to entanglement — the qubits share one thing, so they cannot
    disagree.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3 · Teleportation (aspirational)

    The showstopper. What you'll build:

    1. **Alice** has a qubit ψ in some state (we'll pick a specific one).
    2. **Alice and Bob share an entangled Bell pair** — a resource
       prepared earlier.
    3. Alice does a **Bell measurement** on ψ and her half of the pair.
    4. Alice sends **2 classical bits** to Bob (the measurement outcome).
    5. Bob applies a **Pauli correction** based on the 2 bits.
    6. Bob's qubit is now in the state ψ was in. Alice's is gone.

    No copying. No signalling faster than light. Just the
    arithmetic-of-4 you saw this morning.

    **This is the hardest circuit you'll build today. Reaching a
    working teleportation circuit means you've done more than most
    undergraduates ever do.**

    **Structure of the cells below**:
    - Cell 13: prepare Alice's ψ (a specific non-trivial state so
      we can tell if it teleported correctly).
    - Cell 14: build the full teleportation circuit.
    - Cell 15: run it, verify Bob's qubit matches Alice's initial ψ.
    """)
    return


@app.cell
def _(QuantumCircuit, np):
    # Pick a specific non-trivial state for ψ — Ry(π/3) on |0⟩ gives
    # ψ = cos(π/6)|0⟩ + sin(π/6)|1⟩ = √3/2 |0⟩ + 1/2 |1⟩. If teleportation
    # works, Bob should measure |0⟩ ~75% of the time and |1⟩ ~25%.

    theta_alice = np.pi / 3

    prep_alice = QuantumCircuit(1)
    prep_alice.ry(theta_alice, 0)
    print("Alice's |ψ⟩ = Ry(π/3)|0⟩.  Expected measurement: "
          "|0⟩ with probability cos²(π/6) ≈ 75%, |1⟩ ≈ 25%.")
    prep_alice.draw("mpl")
    return (theta_alice,)


@app.cell
def _(QuantumCircuit, theta_alice):
    # 3 qubits: q0 = Alice's ψ, q1 = Alice's half of Bell pair, q2 = Bob's half.
    # 3 classical bits: c0 and c1 = Alice's Bell-measurement outcome,
    #                   c2 = Bob's final measurement (to verify).

    tp = QuantumCircuit(3, 3)

    # Build the teleportation circuit step by step. Fill in each TODO; the final
    # verify-measure (Step 5) is left in so the cell runs while you work.

    # Step 1 — prepare Alice's ψ on q0.
    #   TODO: tp.ry(theta_alice, 0)

    # Step 2 — create the entangled Bell pair between q1 (Alice) and q2 (Bob).
    #   TODO: Hadamard on q1, then CNOT q1 → q2
    tp.barrier()

    # Step 3 — Alice's Bell measurement on q0 + q1.
    #   TODO: CNOT q0 → q1, Hadamard on q0,
    #         then tp.measure(0, 0) (→ c0) and tp.measure(1, 1) (→ c1)
    tp.barrier()

    # Step 4 — Bob's Pauli correction, conditioned on Alice's two classical bits.
    #   Qiskit 2.x conditions on a classical bit with `if_test`. Template:
    #       with tp.if_test((<classical_bit>, 1)):   # runs the block if that bit == 1
    #           tp.<gate>(2)
    #   TODO: add the TWO corrections Bob needs (each on qubit 2). Which classical
    #         bit triggers which Pauli? Recall the morning's "every count is 4".

    # Step 5 — Bob measures his qubit q2 to verify (leave this in).
    tp.measure(2, 2)

    tp.draw("mpl")
    return (tp,)


@app.cell
def _(backend, plot_histogram, tp, transpile):
    _tc = transpile(tp, backend)
    _result = backend.run(_tc, shots=1024).result()
    _counts = _result.get_counts()

    # Bob's outcome is bit c2 (the leftmost bit in Qiskit's ordering).
    # Marginalise: count how often Bob measured 0 vs 1, regardless of
    # Alice's outcomes.
    _bob_0 = sum(v for k, v in _counts.items() if k[0] == "0")
    _bob_1 = sum(v for k, v in _counts.items() if k[0] == "1")
    _total = _bob_0 + _bob_1
    _p0 = _bob_0 / _total
    _p1 = _bob_1 / _total

    print("Bob's marginal outcome (what teleportation delivered):")
    print(f"  |0⟩ measured {_p0:.1%} of the time (expected ~75%)")
    print(f"  |1⟩ measured {_p1:.1%} of the time (expected ~25%)")
    print("If those match, teleportation worked — Alice's ψ arrived at Bob "
          "without ever being copied.")
    plot_histogram(_counts)
    return


@app.cell
def _(mo):
    mo.md("""
    **Aspirational check** — if you're here:

    - Did Bob's marginals match Alice's expected distribution
      (75/25)?
    - Can you explain to a peer WHY the `if_test` blocks are
      conditioned on the two classical bits? (Hint: log₂4 = 2.)
    - What would happen if you skipped Bob's `X` correction?
      (Answer: Bob's state would be **rotated** — you'd measure
      the wrong distribution about half the time.)

    If you can answer those and your marginals match, you have
    built and understood quantum teleportation. That's the ceiling
    for today.

    **Bonus** (if you're still hungry): change `theta_alice` to
    a different angle, re-run the whole thing, and check that
    Bob's marginals still match Alice's predicted distribution.
    The teleportation should work for *any* ψ.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Wrap

    You built:
    - **Quantum coin** — ψ in superposition, measured.
    - **GHZ state** — three qubits sharing one thing.
    - **Teleportation** — moving ψ without copying her, using
      entanglement + 2 classical bits + a Pauli correction.

    Every count was 4. The arithmetic wrote itself.

    Tomorrow: **algorithms**. Where the quantum speedup comes from,
    and why √N is not a lucky trick.
    """)
    return


if __name__ == "__main__":
    app.run()
