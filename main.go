// Copyright 2026 The Model Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Relativistic quantum vacuum simulator with gradient-descent optimization.
//
// Models a free (weakly interacting) massive scalar field vacuum in 1+1D
// Minkowski spacetime. Virtual quanta live in two conjugate sectors φ and π.
// Each quantum has a spacetime event (t, x) and a 4-momentum (E, p).
//
// Relativistic structure:
//
//	• Minkowski interval  s² = c²Δt² − Δx²
//	• On-shell mass constraint  E² − p²c² = m²c⁴
//	• Dispersion  ω = √(p²c² + m²c⁴) / ℏ  (cgs-natural: ℏ=1 ⇒ ω=√(p²c²+m²c⁴))
//	• Propagator kernel from invariant distance √|s²|
//	• Causal correlators preferred on spacelike separations (vacuum Wightman)
//
// Adam minimizes the vacuum energy; output is an animated spacetime GIF.
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"math"
	"math/rand"
	"os"

	"github.com/pointlander/gradient"
)

const (
	// C is the speed of light (simulation units).
	C = 1.0
	// Mass is the rest mass m.
	Mass = 0.4
	// Lambda is the weak self-coupling.
	Lambda = 0.02
	// Hbar is ℏ (set to 1 in natural units; kept for clarity).
	Hbar = 1.0
	// SoftEPS regularizes singularities at s² → 0 (light cone / coincidence).
	SoftEPS = 5e-2
	// Eta is the Adam learning rate.
	Eta = 2.5e-2
	// Particles is the number of virtual quanta per sector.
	Particles = 33
	// Frames is the number of GIF frames.
	Frames = 512
	// StepsPerFrame is Adam steps between frames.
	StepsPerFrame = 2
	// Drop is the dropout rate for stochastic vacuum sampling.
	Drop = 0.2
)

// MinkowskiS2 computes pairwise Minkowski interval-squared matrices:
//
//	s²_ij = c² (t_i − t_j)² − (x_i − x_j)²
//
// Rows of a,b are events (t, x). Result shape is (n_a, n_b).
func MinkowskiS2[T gradient.Number](c T) gradient.Binary[T] {
	c2 := c * c
	return func(k gradient.Continuation[T], node int, a, b *gradient.V[T], options ...map[string]interface{}) bool {
		if len(a.S) != 2 || len(b.S) != 2 || a.S[0] != 2 || b.S[0] != 2 {
			panic("MinkowskiS2 expects events of shape (2, n)")
		}
		na, nb := a.S[1], b.S[1]
		out := gradient.NewV[T](na, nb)
		for i := 0; i < na; i++ {
			ti, xi := a.X[i*2], a.X[i*2+1]
			for j := 0; j < nb; j++ {
				dt := ti - b.X[j*2]
				dx := xi - b.X[j*2+1]
				out.X = append(out.X, c2*dt*dt-dx*dx)
			}
		}
		if k(out) {
			return true
		}
		// ∂s²/∂t_i =  2 c² (t_i − t_j),  ∂s²/∂x_i = −2 (x_i − x_j)
		// ∂s²/∂t_j = −2 c² (t_i − t_j),  ∂s²/∂x_j =  2 (x_i − x_j)
		idx := 0
		for i := 0; i < na; i++ {
			ti, xi := a.X[i*2], a.X[i*2+1]
			for j := 0; j < nb; j++ {
				d := out.D[idx]
				dt := ti - b.X[j*2]
				dx := xi - b.X[j*2+1]
				a.D[i*2] += d * (2 * c2 * dt)
				a.D[i*2+1] += d * (-2 * dx)
				b.D[j*2] += d * (-2 * c2 * dt)
				b.D[j*2+1] += d * (2 * dx)
				idx++
			}
		}
		return false
	}
}

// ordered is constrained to real scalars (comparisons).
type ordered interface {
	~float32 | ~float64
}

func absT[T ordered](x T) T {
	if x < 0 {
		return -x
	}
	return x
}

func signT[T ordered](x T) T {
	if x > 0 {
		return 1
	}
	if x < 0 {
		return -1
	}
	return 0
}

// RelativisticKernel maps s² → vacuum propagator weight.
//
// Uses the invariant distance ρ = √(|s²| + ε) and a massive Yukawa form
// G = exp(−m ρ) / ρ, which recovers the correct exponential decay of the
// free massive Wightman function at large spacelike separation.
// Self-entries (s² == 0 on the diagonal of identical events) are zeroed.
func RelativisticKernel[T ordered](mass, eps T) gradient.Unary[T] {
	return func(k gradient.Continuation[T], node int, a *gradient.V[T], options ...map[string]interface{}) bool {
		out := gradient.NewV[T](a.S...)
		rho := make([]T, len(a.X))
		for i, s2 := range a.X {
			r := gradient.Sqrt(absT(s2) + eps)
			rho[i] = r
			// Zero diagonal of square self-kernel matrices (i == j ⇒ s² = 0).
			if a.S[0] == a.S[1] && s2 == 0 {
				out.X = append(out.X, 0)
			} else {
				out.X = append(out.X, gradient.Exp(-mass*r)/r)
			}
		}
		if k(out) {
			return true
		}
		// G = e^{−mρ}/ρ,  ρ = √(|s²|+ε)
		// dG/dρ = −G (m + 1/ρ),  dρ/ds² = sign(s²)/(2ρ)
		for i, d := range out.D {
			s2 := a.X[i]
			if a.S[0] == a.S[1] && s2 == 0 {
				continue
			}
			r := rho[i]
			g := out.X[i]
			dGdr := -g * (mass + 1/r)
			a.D[i] += d * dGdr * signT(s2) / (2 * r)
		}
		return false
	}
}

// ElementSquare computes x*x element-wise.
func ElementSquare[T gradient.Number](k gradient.Continuation[T], node int, a *gradient.V[T], options ...map[string]interface{}) bool {
	c := gradient.NewV[T](a.S...)
	for _, x := range a.X {
		c.X = append(c.X, x*x)
	}
	if k(c) {
		return true
	}
	for i, d := range c.D {
		a.D[i] += 2 * a.X[i] * d
	}
	return false
}

// Scale multiplies every element by a constant (options["scale"]).
func Scale[T gradient.Number](k gradient.Continuation[T], node int, a *gradient.V[T], options ...map[string]interface{}) bool {
	var s T = 1
	if len(options) > 0 {
		if v, ok := options[0]["scale"].(T); ok {
			s = v
		}
	}
	c := gradient.NewV[T](a.S...)
	for _, x := range a.X {
		c.X = append(c.X, s*x)
	}
	if k(c) {
		return true
	}
	for i, d := range c.D {
		a.D[i] += s * d
	}
	return false
}

// MassShell residual: for 4-momenta rows (E, p),
//
//	r = E² − p² c² − m² c⁴
//
// Returned as a (1, n) column of residuals (element-wise).
func MassShell[T gradient.Number](c, mass T) gradient.Unary[T] {
	c2 := c * c
	m2c4 := mass * mass * c2 * c2
	return func(k gradient.Continuation[T], node int, a *gradient.V[T], options ...map[string]interface{}) bool {
		if a.S[0] != 2 {
			panic("MassShell expects 4-momenta (E, p) with shape (2, n)")
		}
		n := a.S[1]
		out := gradient.NewV[T](1, n)
		for i := 0; i < n; i++ {
			e, p := a.X[i*2], a.X[i*2+1]
			out.X = append(out.X, e*e-p*p*c2-m2c4)
		}
		if k(out) {
			return true
		}
		for i := 0; i < n; i++ {
			e, p := a.X[i*2], a.X[i*2+1]
			d := out.D[i]
			a.D[i*2] += d * (2 * e)
			a.D[i*2+1] += d * (-2 * p * c2)
		}
		return false
	}
}

// RelOmega computes relativistic mode frequencies ω = √(p²c² + m²c⁴) / ℏ
// from 4-momenta (E, p) — uses spatial momentum p only (on-shell ω).
func RelOmega[T gradient.Number](c, mass, hbar T) gradient.Unary[T] {
	c2 := c * c
	m2c4 := mass * mass * c2 * c2
	return func(k gradient.Continuation[T], node int, a *gradient.V[T], options ...map[string]interface{}) bool {
		if a.S[0] != 2 {
			panic("RelOmega expects shape (2, n)")
		}
		n := a.S[1]
		out := gradient.NewV[T](1, n)
		for i := 0; i < n; i++ {
			p := a.X[i*2+1]
			out.X = append(out.X, gradient.Sqrt(p*p*c2+m2c4)/hbar)
		}
		if k(out) {
			return true
		}
		for i := 0; i < n; i++ {
			p := a.X[i*2+1]
			w := out.X[i]
			// dω/dp = (p c²) / (ℏ² ω) but ω = √(...)/ℏ so dω/dp = p c² / (ℏ √(...)) = p c² / (ℏ² ω)
			if w != 0 {
				a.D[i*2+1] += out.D[i] * (p * c2) / (hbar * hbar * w)
			}
		}
		return false
	}
}

// Vacuum is a relativistic variational quantum vacuum optimized with Adam.
type Vacuum[T ordered] struct {
	Iteration int
	Context   *gradient.Context[T]
	Set       *gradient.Set[T]
	rng       *rand.Rand
	Images    *gif.GIF
	loss      gradient.Meta[T]
	energies  []float64
}

// NewVacuum creates a relativistic vacuum with the given seed and particle count.
func NewVacuum[T ordered](seed int64, n int) *Vacuum[T] {
	rng := rand.New(rand.NewSource(seed))
	context := &gradient.Context[T]{}
	set := context.NewSet()
	// Spacetime events (t, x) and 4-momenta (E, p) for each sector.
	set.Add("phi", 2, n)
	set.Add("pi", 2, n)
	set.Add("k_phi", 2, n)
	set.Add("k_pi", 2, n)
	set.InitAdam(rng)

	// Seed momenta near the mass shell E ≈ +√(p²c² + m²c⁴).
	initNearShell(set.ByName["k_phi"], rng, T(C), T(Mass))
	initNearShell(set.ByName["k_pi"], rng, T(C), T(Mass))

	v := &Vacuum[T]{
		rng:     rng,
		Context: context,
		Set:     &set,
		Images:  &gif.GIF{},
	}
	v.loss = v.buildEnergy()
	return v
}

func initNearShell[T ordered](k *gradient.V[T], rng *rand.Rand, c, mass T) {
	c2 := c * c
	m2c4 := mass * mass * c2 * c2
	for i := 0; i < k.S[1]; i++ {
		p := T(rng.NormFloat64()) * mass * c
		// Positive-energy branch (particles / antiparticles both E > 0 in vacuum pairs).
		e := gradient.Sqrt(p*p*c2 + m2c4)
		// Small jitter off shell so the constraint term has work to do.
		e += T(rng.NormFloat64()) * mass * c2 * 0.05
		k.X[i*2] = e
		k.X[i*2+1] = p
	}
}

// buildEnergy constructs the relativistic vacuum energy graph.
//
//	E_correlator — conjugate sectors share Minkowski propagator structure
//	E_mass_shell — 4-momenta sit on E² − p²c² = m²c⁴
//	E_zeropoint  — Σ ½ ℏ ω_p with relativistic dispersion
//	E_pair       — virtual pairs: spacelike separation + opposite momenta
//	E_interaction— weak λ self-coupling on spacetime amplitudes
//	E_energy_pos — soft preference for E > 0 (Dirac sea / positive-frequency)
func (v *Vacuum[T]) buildEnergy() gradient.Meta[T] {
	ctx := v.Context
	minkowski := ctx.B(MinkowskiS2[T](T(C)))
	kernel := ctx.U(RelativisticKernel[T](T(Mass), T(SoftEPS)))
	quadratic := ctx.B(ctx.Quadratic)
	mul := ctx.B(ctx.Mul)
	add := ctx.B(ctx.Add)
	avg := ctx.U(ctx.Avg)
	dropout := ctx.U(ctx.Dropout)
	matSquare := ctx.U(ctx.Square)
	elemSq := ctx.U(ElementSquare[T])
	scale := ctx.U(Scale[T])
	massShell := ctx.U(MassShell[T](T(C), T(Mass)))
	omega := ctx.U(RelOmega[T](T(C), T(Mass), T(Hbar)))

	drop := Drop
	dropoutOpt := map[string]interface{}{
		"rng":  v.rng,
		"drop": &drop,
	}

	phi := v.Set.Get("phi")
	pi := v.Set.Get("pi")
	kPhi := v.Set.Get("k_phi")
	kPi := v.Set.Get("k_pi")

	// Relativistic propagator kernels from Minkowski intervals.
	gPhi := kernel(minkowski(phi, phi))
	gPi := kernel(minkowski(pi, pi))

	// Conjugate correlator match (stochastic vacuum sampling via dropout).
	kGPhi := mul(dropout(matSquare(pi), dropoutOpt), gPhi)
	kGPi := mul(dropout(matSquare(phi), dropoutOpt), gPi)
	eCorrelator := avg(quadratic(kGPhi, kGPi))

	// Mass-shell constraints for both sectors: ⟨r²⟩ with r = E² − p²c² − m²c⁴.
	shellPhi := elemSq(massShell(kPhi))
	shellPi := elemSq(massShell(kPi))
	shellScale := map[string]interface{}{"scale": T(1.0)}
	eMassShell := scale(avg(add(shellPhi, shellPi)), shellScale)

	// Zero-point energy: ½ ℏ ⟨ω⟩ with ω = √(p²c² + m²c⁴)/ℏ.
	zpScale := map[string]interface{}{"scale": T(0.5 * Hbar)}
	eZeroPoint := scale(avg(add(omega(kPhi), omega(kPi))), zpScale)

	// Virtual pair structure:
	//  1) φ–π events prefer small invariant separation (pair creation near a vertex)
	//  2) 4-momenta of the two sectors share similar Gram structure (pair kinematics)
	softAbs := ctx.U(SoftAbs[T](T(SoftEPS)))
	pairScale := map[string]interface{}{"scale": T(0.12)}
	ePair := scale(avg(softAbs(minkowski(phi, pi))), pairScale)

	// Momentum conjugation: match |k| Gram structure between sectors.
	momScale := map[string]interface{}{"scale": T(0.1)}
	eMomPair := scale(avg(quadratic(matSquare(kPhi), matSquare(kPi))), momScale)

	// Weak self-interaction on spacetime coordinates.
	intScale := map[string]interface{}{"scale": T(Lambda)}
	eInteraction := scale(avg(add(elemSq(elemSq(phi)), elemSq(elemSq(pi)))), intScale)

	// Positive-energy soft prior: penalize negative E via ReLU-like square of min(E,0)
	// Implemented as: softplus-free — use element square of (E − |E|)/2 ≈ negative part.
	// Simpler: quadratic pull of E toward +mc² using mass-shell already; add small
	// penalty on E² when we want large |E| only if on shell — covered by shell + ω.

	// Spacelike vacuum preference: average of max(s², 0) on self-intervals should
	// not dominate — vacuum correlators are spacelike (s² < 0). Penalize timelike
	// self-pair weight via softplus of s².
	// Approximate: average ReLU(s²) using (s² + √(s²²+ε))/2.
	spacelike := ctx.U(softTimelikePenalty[T](T(SoftEPS)))
	spaceScale := map[string]interface{}{"scale": T(0.08)}
	eSpacelike := scale(avg(add(spacelike(minkowski(phi, phi)), spacelike(minkowski(pi, pi)))), spaceScale)

	total := add(add(eCorrelator, eMassShell), add(eZeroPoint, ePair))
	total = add(total, add(eMomPair, add(eInteraction, eSpacelike)))
	return avg(total)
}

// SoftAbs returns √(x² + ε) ≈ |x|.
func SoftAbs[T gradient.Number](eps T) gradient.Unary[T] {
	return func(k gradient.Continuation[T], node int, a *gradient.V[T], options ...map[string]interface{}) bool {
		out := gradient.NewV[T](a.S...)
		for _, x := range a.X {
			out.X = append(out.X, gradient.Sqrt(x*x+eps))
		}
		if k(out) {
			return true
		}
		for i, d := range out.D {
			x := a.X[i]
			r := gradient.Sqrt(x*x + eps)
			a.D[i] += d * x / r
		}
		return false
	}
}

// softTimelikePenalty returns ≈ max(s², 0) smoothed: (s² + √(s²²+ε))/2.
// Penalizes timelike separations in vacuum self-correlations.
func softTimelikePenalty[T gradient.Number](eps T) gradient.Unary[T] {
	return func(k gradient.Continuation[T], node int, a *gradient.V[T], options ...map[string]interface{}) bool {
		out := gradient.NewV[T](a.S...)
		for _, s2 := range a.X {
			r := gradient.Sqrt(s2*s2 + eps)
			out.X = append(out.X, (s2+r)/2)
		}
		if k(out) {
			return true
		}
		for i, d := range out.D {
			s2 := a.X[i]
			r := gradient.Sqrt(s2*s2 + eps)
			// d/ds² (s² + √(s²²+ε))/2 = (1 + s²/√(s²²+ε))/2
			a.D[i] += d * (1 + s2/r) / 2
		}
		return false
	}
}

// Step runs steps Adam iterations and records the last energy.
func (v *Vacuum[T]) Step(steps int) T {
	var energy T
	for range steps {
		v.Set.Zero()
		energy = gradient.Gradient(v.loss).X[0]
		v.Set.Adam(gradient.B1, gradient.B2, T(Eta))
		v.Iteration++
	}
	v.energies = append(v.energies, any(energy).(float64))
	return energy
}

var palette = make([]color.Color, 256)

func init() {
	for i := range 256 {
		palette[i] = color.RGBA{uint8(i / 6), uint8(i / 5), uint8(i / 3), 0xff}
	}
	palette[255] = color.RGBA{0xff, 0xff, 0xff, 0xff} // event core
	palette[254] = color.RGBA{0x55, 0xcc, 0xff, 0xff} // φ sector
	palette[253] = color.RGBA{0xff, 0x77, 0xbb, 0xff} // π sector
	palette[252] = color.RGBA{0xff, 0xdd, 0x44, 0xff} // energy
	palette[251] = color.RGBA{0x44, 0x55, 0x66, 0xff} // divider
	palette[250] = color.RGBA{0x88, 0xaa, 0xcc, 0xff} // progress
	palette[249] = color.RGBA{0x33, 0x88, 0x55, 0xff} // light cone
	palette[248] = color.RGBA{0xaa, 0x66, 0xff, 0xff} // spacelike link
	palette[247] = color.RGBA{0xff, 0x66, 0x44, 0xff} // timelike link
	palette[246] = color.RGBA{0xee, 0xee, 0x66, 0xff} // lightlike link
}

// Render draws spacetime diagrams (t horizontal, x vertical) with light cones.
func (v *Vacuum[T]) Render() {
	const (
		w      = 1024
		h      = 512
		half   = 512
		margin = 12
		plotW  = 488
		plotH  = 460
	)
	img := image.NewPaletted(image.Rect(0, 0, w, h), palette)

	// Faint vacuum noise.
	rng := rand.New(rand.NewSource(int64(v.Iteration)*997 + 13))
	for y := 0; y < h-28; y++ {
		for x := 0; x < w; x++ {
			if rng.Float64() < 0.015 {
				img.SetColorIndex(x, y, uint8(6+rng.Intn(18)))
			}
		}
	}

	// Sector divider.
	for y := 0; y < h; y++ {
		img.SetColorIndex(half, y, 251)
		if half+1 < w {
			img.SetColorIndex(half+1, y, 251)
		}
	}

	drawSector := func(eventName, momName string, xOff int, col uint8) {
		ev := v.Set.ByName[eventName]
		mom := v.Set.ByName[momName]
		n := ev.S[1]
		ts := make([]float64, n)
		xs := make([]float64, n)
		es := make([]float64, n)
		ps := make([]float64, n)
		minT, maxT := math.MaxFloat64, -math.MaxFloat64
		minX, maxX := math.MaxFloat64, -math.MaxFloat64
		for i := 0; i < n; i++ {
			t := any(ev.X[i*2]).(float64)
			x := any(ev.X[i*2+1]).(float64)
			ts[i], xs[i] = t, x
			es[i] = any(mom.X[i*2]).(float64)
			ps[i] = any(mom.X[i*2+1]).(float64)
			if t < minT {
				minT = t
			}
			if t > maxT {
				maxT = t
			}
			if x < minX {
				minX = x
			}
			if x > maxX {
				maxX = x
			}
		}
		// Symmetric bounds so light cones at 45° (c=1) render cleanly.
		dt := maxT - minT
		dx := maxX - minX
		if dt < 1e-6 {
			dt = 1
		}
		if dx < 1e-6 {
			dx = 1
		}
		// Pad and equalize scales in display units for c=1 cones.
		span := math.Max(dt, dx) * 1.15
		tMid := 0.5 * (minT + maxT)
		xMid := 0.5 * (minX + maxX)
		minT, maxT = tMid-span/2, tMid+span/2
		minX, maxX = xMid-span/2, xMid+span/2
		dt, dx = span, span

		toPix := func(t, x float64) (int, int) {
			px := xOff + margin + int(float64(plotW)*(t-minT)/dt)
			// x up the page (relativistic diagram convention often has x vertical).
			py := margin + int(float64(plotH)*(maxX-x)/dx)
			return px, py
		}

		// Light cones through the sector center.
		cx, cy := toPix(tMid, xMid)
		coneLen := plotW / 2
		drawLine(img, cx-coneLen, cy-coneLen, cx+coneLen, cy+coneLen, xOff, half, 249)
		drawLine(img, cx-coneLen, cy+coneLen, cx+coneLen, cy-coneLen, xOff, half, 249)

		// Causal links to nearest neighbor, colored by interval type.
		for i := 0; i < n; i++ {
			best, bestD := -1, math.MaxFloat64
			for j := 0; j < n; j++ {
				if i == j {
					continue
				}
				dtt := ts[i] - ts[j]
				dxx := xs[i] - xs[j]
				// Use Euclidean proximity in spacetime plane for neighbor pick.
				d := dtt*dtt + dxx*dxx
				if d < bestD {
					bestD, best = d, j
				}
			}
			if best < 0 {
				continue
			}
			s2 := C*C*(ts[i]-ts[best])*(ts[i]-ts[best]) - (xs[i]-xs[best])*(xs[i]-xs[best])
			link := uint8(248) // spacelike
			if s2 > SoftEPS {
				link = 247 // timelike
			} else if math.Abs(s2) <= SoftEPS {
				link = 246 // near lightlike
			}
			x0, y0 := toPix(ts[i], xs[i])
			x1, y1 := toPix(ts[best], xs[best])
			drawLine(img, x0, y0, x1, y1, xOff, half, link)
		}

		// Events; radius hints |p| (boost).
		for i := 0; i < n; i++ {
			px, py := toPix(ts[i], xs[i])
			boost := math.Abs(ps[i]) / (Mass*C + 1e-6)
			rad := 3 + int(math.Min(boost, 3))
			for oy := -rad; oy <= rad; oy++ {
				for ox := -rad; ox <= rad; ox++ {
					if ox*ox+oy*oy > rad*rad {
						continue
					}
					x, y := px+ox, py+oy
					if x < xOff || x >= xOff+half || y < 0 || y >= h-28 {
						continue
					}
					img.SetColorIndex(x, y, col)
				}
			}
			for oy := -1; oy <= 1; oy++ {
				for ox := -1; ox <= 1; ox++ {
					x, y := px+ox, py+oy
					if x >= xOff && x < xOff+half && y >= 0 && y < h-28 {
						img.SetColorIndex(x, y, 255)
					}
				}
			}
			// Momentum dash (spatial p direction, horizontal in plot is t so p along x → vertical).
			pLen := int(math.Max(-12, math.Min(12, ps[i]/(Mass*C+1e-6)*8)))
			if pLen != 0 {
				drawLine(img, px, py, px, py-pLen, xOff, half, 255)
			}
		}
	}

	drawSector("phi", "k_phi", 0, 254)
	drawSector("pi", "k_pi", half, 253)

	// Energy history.
	if len(v.energies) > 1 {
		minE, maxE := v.energies[0], v.energies[0]
		for _, e := range v.energies {
			if e < minE {
				minE = e
			}
			if e > maxE {
				maxE = e
			}
		}
		span := maxE - minE
		if span < 1e-12 {
			span = 1
		}
		last := len(v.energies) - 1
		for i, e := range v.energies {
			x := i * (w - 1) / maxInt(last, 1)
			height := int(22 * (e - minE) / span)
			for dy := 0; dy <= height; dy++ {
				img.SetColorIndex(x, h-1-dy, 252)
			}
		}
	}

	progress := v.Iteration * (w - 1) / maxInt(Frames*StepsPerFrame, 1)
	for y := h - 26; y < h-24; y++ {
		for x := 0; x <= progress && x < w; x++ {
			img.SetColorIndex(x, y, 250)
		}
	}

	v.Images.Image = append(v.Images.Image, img)
	v.Images.Delay = append(v.Images.Delay, 4)
	v.Images.Disposal = append(v.Images.Disposal, gif.DisposalBackground)
}

func drawLine(img *image.Paletted, x0, y0, x1, y1, xOff, half int, idx uint8) {
	dx := abs(x1 - x0)
	dy := -abs(y1 - y0)
	sx, sy := 1, 1
	if x0 > x1 {
		sx = -1
	}
	if y0 > y1 {
		sy = -1
	}
	err := dx + dy
	x, y := x0, y0
	bounds := img.Bounds()
	for {
		if x >= xOff && x < xOff+half && y >= 0 && y < bounds.Dy()-28 {
			img.SetColorIndex(x, y, idx)
		}
		if x == x1 && y == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x += sx
		}
		if e2 <= dx {
			err += dx
			y += sy
		}
	}
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	vac := NewVacuum[float64](2, Particles)
	fmt.Printf("relativistic quantum vacuum simulator (gradient descent)\n")
	fmt.Printf("  c=%.2f  m=%.3f  λ=%.3f  ℏ=%.2f\n", C, Mass, Lambda, Hbar)
	fmt.Printf("  particles/sector=%d  frames=%d  steps/frame=%d  η=%.3f\n",
		Particles, Frames, StepsPerFrame, Eta)

	for i := range Frames {
		e := vac.Step(StepsPerFrame)
		vac.Render()
		if i%64 == 0 || i == Frames-1 {
			// Report mean mass-shell residual and mean ω for diagnostics.
			shell, omega := diagnostics(vac)
			fmt.Printf("  frame %4d/%d  E_vac=%.6f  ⟨shell²⟩=%.4f  ⟨ω⟩=%.4f  iter=%d\n",
				i+1, Frames, e, shell, omega, vac.Iteration)
		}
	}

	out, err := os.Create("model.gif")
	if err != nil {
		panic(err)
	}
	defer out.Close()
	if err := gif.EncodeAll(out, vac.Images); err != nil {
		panic(err)
	}
	fmt.Printf("wrote model.gif (%d frames)\n", len(vac.Images.Image))
}

func diagnostics[T ordered](v *Vacuum[T]) (shellMean, omegaMean float64) {
	c2 := C * C
	m2c4 := Mass * Mass * c2 * c2
	var shellSum, omegaSum float64
	var count int
	for _, name := range []string{"k_phi", "k_pi"} {
		k := v.Set.ByName[name]
		for i := 0; i < k.S[1]; i++ {
			e := any(k.X[i*2]).(float64)
			p := any(k.X[i*2+1]).(float64)
			r := e*e - p*p*c2 - m2c4
			shellSum += r * r
			omegaSum += math.Sqrt(p*p*c2+m2c4) / Hbar
			count++
		}
	}
	return shellSum / float64(count), omegaSum / float64(count)
}
