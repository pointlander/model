# Quantum vacuum model

A **relativistic quantum vacuum simulator** for a free (weakly interacting) massive scalar field in **1+1D Minkowski spacetime**. Virtual quanta live in two conjugate sectors and are optimized with **Adam gradient descent** (via [gradient](https://github.com/pointlander/gradient)) so the configuration approaches a low-energy vacuum.

```bash
go build -o model . && ./model
```

Produces `model.gif` (spacetime diagrams of the optimizing vacuum).

---

## Theory

### Relativistic scalar vacuum

In relativistic quantum field theory, the vacuum \(\lvert 0\rangle\) is the ground state of a field theory on Minkowski spacetime. For a free real scalar field of mass \(m\), mode frequencies obey the relativistic dispersion relation

\[
\omega_{\mathbf{p}} = \sqrt{\mathbf{p}^{2}c^{2} + m^{2}c^{4}}\,/\hbar .
\]

Each mode contributes a **zero-point energy** \(\tfrac12\hbar\omega_{\mathbf{p}}\). Vacuum two-point correlations (Wightman / Hadamard structure) depend on the **Minkowski interval**

\[
s^{2} = c^{2}\Delta t^{2} - \Delta x^{2},
\]

and fall off at large spacelike separation on the scale of the Compton length \(\hbar/(mc)\). Four-momenta of on-shell quanta satisfy the **mass-shell constraint**

\[
E^{2} - p^{2}c^{2} = m^{2}c^{4}.
\]

Virtual particle–antiparticle pairs can fluctuate into existence provided total four-momentum and energy accounting are consistent with the uncertainty principle; their mutual separation is organized by the causal structure (light cones at \(\lvert\Delta x\rvert = c\lvert\Delta t\rvert\)).

### Variational model

This simulator does not discretize a lattice Hamiltonian directly. Instead it represents the vacuum as two conjugate ensembles of virtual quanta:

| Symbol | Role |
|--------|------|
| \(\varphi\) | Particle-sector spacetime events \((t,x)\) |
| \(\pi\) | Antiparticle / conjugate-sector events \((t,x)\) |
| \(k_\varphi,\,k_\pi\) | Four-momenta \((E,p)\) for each quantum |

A differentiable **vacuum energy** is minimized with Adam:

\[
E_{\mathrm{vac}}
=
E_{\mathrm{correlator}}
+ E_{\mathrm{shell}}
+ E_{\mathrm{zeropoint}}
+ E_{\mathrm{pair}}
+ E_{\mathrm{mom}}
+ E_{\mathrm{int}}
+ E_{\mathrm{spacelike}}.
\]

| Term | Meaning |
|------|---------|
| \(E_{\mathrm{correlator}}\) | Conjugate sectors share the same Minkowski **propagator** structure \(G(s^{2})\approx e^{-m\rho}/\rho\), \(\rho=\sqrt{\lvert s^{2}\rvert+\varepsilon}\) |
| \(E_{\mathrm{shell}}\) | Soft on-shell penalty \(\langle(E^{2}-p^{2}c^{2}-m^{2}c^{4})^{2}\rangle\) |
| \(E_{\mathrm{zeropoint}}\) | \(\tfrac12\hbar\langle\omega_{p}\rangle\) with relativistic \(\omega_{p}\) |
| \(E_{\mathrm{pair}}\) | \(\varphi\)–\(\pi\) events prefer small invariant separation (pair vertices) |
| \(E_{\mathrm{mom}}\) | Matching Gram structure of the two sectors’ four-momenta |
| \(E_{\mathrm{int}}\) | Weak \(\lambda\) self-coupling on spacetime coordinates |
| \(E_{\mathrm{spacelike}}\) | Soft penalty on **timelike** self-separations (vacuum correlations are spacelike) |

Stochastic **dropout** on correlator weights acts as vacuum sampling noise during optimization. Units are natural with \(c=1\), \(\hbar=1\) by default (`C`, `Hbar` in `main.go`).

### Causal structure in the figure

Spacetime plots use \(t\) horizontal and \(x\) vertical (equal scale so light cones appear at \(45^\circ\) when \(c=1\)):

- **Green** diagonals — light cones through the sector center  
- **Purple** links — spacelike nearest-neighbor intervals (\(s^{2}<0\))  
- **Red** links — timelike intervals (\(s^{2}>0\))  
- **Yellow** links — near lightlike  
- Marker size / momentum ticks — spatial momentum \(\lvert p\rvert\)

Left panel: \(\varphi\) sector. Right panel: \(\pi\) sector. Bottom strip: vacuum energy during Adam steps.

---

## Simulation results

Default run parameters:

| Parameter | Value |
|-----------|-------|
| \(c\) | \(1.0\) |
| \(m\) | \(0.4\) |
| \(\lambda\) | \(0.02\) |
| \(\hbar\) | \(1.0\) |
| Quanta per sector | \(33\) |
| Frames | \(512\) |
| Adam steps / frame | \(2\) (1024 total) |
| Learning rate \(\eta\) | \(0.025\) |

### Optimization trajectory

Representative diagnostics from a full run:

| Frame | \(E_{\mathrm{vac}}\) | \(\langle\mathrm{shell}^{2}\rangle\) | \(\langle\omega\rangle\) |
|------:|---------------------:|-------------------------------------:|-------------------------:|
| 1 | 4153.56 | 0.0015 | 0.4908 |
| 65 | 553.78 | \(\sim 0\) | 0.4000 |
| 129 | 339.36 | \(\sim 0\) | 0.4000 |
| 257 | 4.78 | \(\sim 0\) | 0.4000 |
| 512 | 0.46 | \(\sim 0\) | 0.4000 |

Interpretation:

1. **Mass shell** is enforced quickly — four-momenta sit on \(E^{2}-p^{2}c^{2}=m^{2}c^{4}\).
2. **Mode frequencies** settle at \(\langle\omega\rangle \approx m c^{2}/\hbar = m\) (rest-frame zero-point of the massive IR vacuum).
3. **Vacuum energy** falls by orders of magnitude as correlator, pair, and causal terms organize the spacetime diagram.

### Animation

![Relativistic quantum vacuum simulation](model.gif)

The animation shows Adam relaxation of the variational vacuum: early frames are disordered off-shell fluctuations; later frames show structured \(\varphi\)/\(\pi\) sectors, light-cone geometry, and a flattened energy history (gold strip at the bottom).

### Reproducing

```bash
go build -o model . && ./model
```

Requires Go 1.25+ and `github.com/pointlander/gradient`. Constants at the top of `main.go` control mass, coupling, particle count, and GIF length.
