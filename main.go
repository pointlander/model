// Copyright 2026 The Model Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Quantum vacuum simulator with gradient-descent optimization.
//
// Models a free (weakly interacting) scalar field vacuum in 2D as two
// conjugate sectors of virtual quanta — φ (particles) and π (antiparticles /
// conjugate samples). Adam minimizes a vacuum energy functional whose
// stationary configurations reproduce vacuum correlator structure, mass gap,
// virtual-pair binding, and zero-point fluctuations.
//
// Energy:
//
//	E = E_correlator + E_mass + E_pair + E_zeropoint + E_interaction
//
// Output is an animated GIF of the optimizing vacuum.
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
	// Mass is the scalar mass gap m (IR scale).
	Mass = 0.4
	// Lambda is the weak self-coupling strength.
	Lambda = 0.02
	// Hbar scales the zero-point contribution.
	Hbar = 1.0
	// SoftEPS regularizes 1/r singularities (lattice UV cutoff).
	SoftEPS = 5e-2
	// Eta is the Adam learning rate.
	Eta = 3e-2
	// Particles is the number of virtual quanta per sector.
	Particles = 33
	// Frames is the number of GIF frames.
	Frames = 512
	// StepsPerFrame is Adam steps between frames.
	StepsPerFrame = 2
	// Drop is the dropout rate for stochastic vacuum sampling.
	Drop = 0.2
)

// SoftInvDist computes a UV-regularized Green kernel from distances:
// diagonal (self) → 0; off-diagonal → 1/(d + ε).
func SoftInvDist[T gradient.Number](eps T) gradient.Unary[T] {
	return func(k gradient.Continuation[T], node int, a *gradient.V[T], options ...map[string]interface{}) bool {
		c := gradient.NewV[T](a.S...)
		for _, x := range a.X {
			if x == 0 {
				c.X = append(c.X, 0)
			} else {
				c.X = append(c.X, 1.0/(x+eps))
			}
		}
		if k(c) {
			return true
		}
		for i, d := range c.D {
			x := a.X[i]
			if x == 0 {
				continue
			}
			denom := x + eps
			a.D[i] += -d / (denom * denom)
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

// Vacuum is a variational quantum vacuum optimized with Adam.
type Vacuum[T gradient.Number] struct {
	Iteration int
	Context   *gradient.Context[T]
	Set       *gradient.Set[T]
	rng       *rand.Rand
	Images    *gif.GIF
	loss      gradient.Meta[T]
	energies  []float64
}

// NewVacuum creates a vacuum with the given RNG seed and particle count.
func NewVacuum[T gradient.Number](seed int64, n int) *Vacuum[T] {
	rng := rand.New(rand.NewSource(seed))
	context := &gradient.Context[T]{}
	set := context.NewSet()
	// φ and π sectors: each quantum is a point in R².
	set.Add("phi", 2, n)
	set.Add("pi", 2, n)
	set.InitAdam(rng)

	v := &Vacuum[T]{
		rng:     rng,
		Context: context,
		Set:     &set,
		Images:  &gif.GIF{},
	}
	v.loss = v.buildEnergy()
	return v
}

// buildEnergy constructs the vacuum energy computational graph.
//
//	E_correlator  — conjugate sectors share inverse-distance (propagator) structure
//	E_mass        — m²/2 ⟨|r|²⟩ mass gap / IR scale
//	E_pair        — virtual pair binding via mean φ–π distance
//	E_zeropoint   — ℏ-weighted UV zero-point from soft 1/r self-kernels
//	E_interaction — weak self-coupling via squared amplitudes
func (v *Vacuum[T]) buildEnergy() gradient.Meta[T] {
	ctx := v.Context
	euclidean := ctx.B(ctx.Euclidean)
	quadratic := ctx.B(ctx.Quadratic)
	mul := ctx.B(ctx.Mul)
	add := ctx.B(ctx.Add)
	avg := ctx.U(ctx.Avg)
	dropout := ctx.U(ctx.Dropout)
	matSquare := ctx.U(ctx.Square)
	softInv := ctx.U(SoftInvDist[T](T(SoftEPS)))
	elemSq := ctx.U(ElementSquare[T])
	scale := ctx.U(Scale[T])

	drop := Drop
	dropoutOpt := map[string]interface{}{
		"rng":  v.rng,
		"drop": &drop,
	}

	phi := v.Set.Get("phi")
	pi := v.Set.Get("pi")

	// Propagator-like kernels G = 1/(d + ε) of each sector.
	gPhi := softInv(euclidean(phi, phi))
	gPi := softInv(euclidean(pi, pi))

	// Conjugate correlator match with stochastic vacuum sampling (dropout):
	// K_φ ~ (ππᵀ) G_φ  should equal  K_π ~ (φφᵀ) G_π  in the vacuum.
	kPhi := mul(dropout(matSquare(pi), dropoutOpt), gPhi)
	kPi := mul(dropout(matSquare(phi), dropoutOpt), gPi)
	eCorrelator := avg(quadratic(kPhi, kPi))

	// Mass gap: m²/2 mean squared coordinate amplitude.
	massScale := map[string]interface{}{"scale": T(0.5 * Mass * Mass)}
	eMass := scale(avg(add(elemSq(phi), elemSq(pi))), massScale)

	// Virtual pair binding: soft attraction between φ and π sectors.
	pairScale := map[string]interface{}{"scale": T(0.2)}
	ePair := scale(avg(euclidean(phi, pi)), pairScale)

	// Zero-point energy: ℏ ⟨G_offdiag⟩ — finite UV mode energy.
	zpScale := map[string]interface{}{"scale": T(Hbar * 0.1)}
	eZeroPoint := scale(avg(add(gPhi, gPi)), zpScale)

	// Weak self-interaction ~ λ ⟨|r|⁴⟩ (element-wise).
	intScale := map[string]interface{}{"scale": T(Lambda)}
	phi4 := elemSq(elemSq(phi))
	pi4 := elemSq(elemSq(pi))
	eInteraction := scale(avg(add(phi4, pi4)), intScale)

	// Total vacuum energy (average of scalar terms keeps graph shape consistent).
	return avg(add(add(eCorrelator, eMass), add(add(ePair, eZeroPoint), eInteraction)))
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
		// Cool dark vacuum background ramp.
		palette[i] = color.RGBA{uint8(i / 5), uint8(i / 4), uint8(i / 3), 0xff}
	}
	palette[255] = color.RGBA{0xff, 0xff, 0xff, 0xff} // core
	palette[254] = color.RGBA{0x66, 0xcc, 0xff, 0xff} // φ sector
	palette[253] = color.RGBA{0xff, 0x88, 0xcc, 0xff} // π sector
	palette[252] = color.RGBA{0xff, 0xdd, 0x44, 0xff} // energy
	palette[251] = color.RGBA{0x44, 0x55, 0x66, 0xff} // divider
	palette[250] = color.RGBA{0x88, 0xaa, 0xcc, 0xff} // progress
}

// Render appends a frame: left = φ sector, right = π sector, bottom = energy.
func (v *Vacuum[T]) Render() {
	const (
		w      = 1024
		h      = 512
		half   = 512
		margin = 10
		plot   = 492
	)
	img := image.NewPaletted(image.Rect(0, 0, w, h), palette)

	// Faint vacuum noise background (deterministic from iteration).
	rng := rand.New(rand.NewSource(int64(v.Iteration)*997 + 13))
	for y := 0; y < h-28; y++ {
		for x := 0; x < w; x++ {
			if rng.Float64() < 0.02 {
				img.SetColorIndex(x, y, uint8(8+rng.Intn(24)))
			}
		}
	}

	// Vertical divider between sectors.
	for y := 0; y < h; y++ {
		img.SetColorIndex(half, y, 251)
		if half+1 < w {
			img.SetColorIndex(half+1, y, 251)
		}
	}

	drawSector := func(name string, xOff int, col uint8) {
		sec := v.Set.ByName[name]
		minX, maxX := math.MaxFloat64, -math.MaxFloat64
		minY, maxY := math.MaxFloat64, -math.MaxFloat64
		n := sec.S[1]
		xs := make([]float64, n)
		ys := make([]float64, n)
		for i := 0; i < n; i++ {
			x := any(sec.X[i*2]).(float64)
			y := any(sec.X[i*2+1]).(float64)
			xs[i], ys[i] = x, y
			if x < minX {
				minX = x
			}
			if x > maxX {
				maxX = x
			}
			if y < minY {
				minY = y
			}
			if y > maxY {
				maxY = y
			}
		}
		dx, dy := maxX-minX, maxY-minY
		if dx < 1e-9 {
			dx = 1
		}
		if dy < 1e-9 {
			dy = 1
		}

		// Pair lines within sector (nearest neighbor) — vacuum correlator hints.
		for i := 0; i < n; i++ {
			best, bestD := -1, math.MaxFloat64
			for j := 0; j < n; j++ {
				if i == j {
					continue
				}
				ddx, ddy := xs[i]-xs[j], ys[i]-ys[j]
				d := ddx*ddx + ddy*ddy
				if d < bestD {
					bestD, best = d, j
				}
			}
			if best < 0 {
				continue
			}
			x0 := xOff + margin + int(float64(plot)*(xs[i]-minX)/dx)
			y0 := margin + int(float64(plot)*(ys[i]-minY)/dy)
			x1 := xOff + margin + int(float64(plot)*(xs[best]-minX)/dx)
			y1 := margin + int(float64(plot)*(ys[best]-minY)/dy)
			drawLine(img, x0, y0, x1, y1, xOff, half, 20)
		}

		for i := 0; i < n; i++ {
			px := xOff + margin + int(float64(plot)*(xs[i]-minX)/dx)
			py := margin + int(float64(plot)*(ys[i]-minY)/dy)
			// Halo.
			for oy := -4; oy <= 4; oy++ {
				for ox := -4; ox <= 4; ox++ {
					if ox*ox+oy*oy > 18 {
						continue
					}
					x, y := px+ox, py+oy
					if x < xOff || x >= xOff+half || y < 0 || y >= h-28 {
						continue
					}
					img.SetColorIndex(x, y, col)
				}
			}
			// Bright core.
			for oy := -1; oy <= 1; oy++ {
				for ox := -1; ox <= 1; ox++ {
					x, y := px+ox, py+oy
					if x >= xOff && x < xOff+half && y >= 0 && y < h-28 {
						img.SetColorIndex(x, y, 255)
					}
				}
			}
		}
	}

	drawSector("phi", 0, 254)
	drawSector("pi", half, 253)

	// Energy history along the bottom.
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

	// Iteration progress.
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
	for {
		if x >= xOff && x < xOff+half && y >= 0 && y < img.Bounds().Dy()-28 {
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
	fmt.Printf("quantum vacuum simulator (gradient descent)\n")
	fmt.Printf("  particles/sector=%d  mass=%.3f  λ=%.3f  ℏ=%.2f\n", Particles, Mass, Lambda, Hbar)
	fmt.Printf("  frames=%d  steps/frame=%d  η=%.3f\n", Frames, StepsPerFrame, Eta)

	for i := range Frames {
		e := vac.Step(StepsPerFrame)
		vac.Render()
		if i%64 == 0 || i == Frames-1 {
			fmt.Printf("  frame %4d/%d  E_vac=%.6f  iter=%d\n", i+1, Frames, e, vac.Iteration)
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
