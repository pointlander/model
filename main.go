// Copyright 2026 The Model Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

// Euclidean computes the euclidean distance between all row vectors and all row vectors
func Euclidean[T gradient.Number](k gradient.Continuation[T], node int, a, b *gradient.V[T], options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c, sizeA, sizeB := gradient.NewV[T](a.S[1], b.S[1]), len(a.X), len(b.X)
	for i := 0; i < sizeA; i += width {
		for ii := 0; ii < sizeB; ii += width {
			av, bv, sum := a.X[i:i+width], b.X[ii:ii+width], T(0.0)
			for j, ax := range av {
				diff := (ax - bv[j])
				sum += diff * diff
			}
			c.X = append(c.X, gradient.Sqrt(sum))
		}
	}
	if k(c) {
		return true
	}
	for _, x := range a.D {
		if gradient.IsInf(x) || gradient.IsNaN(x) {
			fmt.Println("euclidean", a.D)
			panic(x)
		}
	}
	index := 0
	for i := 0; i < sizeA; i += width {
		for ii := 0; ii < sizeB; ii += width {
			av, bv, cx, ad, bd, d := a.X[i:i+width], b.X[ii:ii+width], c.X[index], a.D[i:i+width], b.D[ii:ii+width], c.D[index]
			for j, ax := range av {
				if cx == 0 {
					continue
				}
				if gradient.IsNaN((ax-bv[j])*d/cx) || gradient.IsInf((ax-bv[j])*d/cx) {
					panic("blah")
				}
				if gradient.IsNaN((bv[j]-ax)*d/cx) || gradient.IsInf((bv[j]-ax)*d/cx) {
					panic("gah")
				}
				ad[j] += (ax - bv[j]) * d / cx
				bd[j] += (bv[j] - ax) * d / cx
			}
			index++
		}
	}
	for _, x := range a.D {
		if gradient.IsInf(x) || gradient.IsNaN(x) {
			fmt.Println("euclidean 2", a.D)
			panic(x)
		}
	}
	return false
}

// Neuron is a neuromorphic neuron
type Neuron[T gradient.Number] struct {
	Iteration int
	Context   *gradient.Context[T]
	Set       *gradient.Set[T]
	rng       *rand.Rand
	Images    *gif.GIF
}

// NewNeuron creates a new neuron
func NewNeuron[T gradient.Number](seed int64, rows int) Neuron[T] {
	rng := rand.New(rand.NewSource(seed))
	context := gradient.Context[T]{}
	set := context.NewSet()
	set.Add("x", 2, rows)
	set.Add("y", 2, rows)
	set.InitAdam(rng)

	return Neuron[T]{
		rng:     rng,
		Context: &context,
		Set:     &set,
		Images:  &gif.GIF{},
	}
}

const (
	// Eta is the learning rate
	Eta = 1.0e-1
)

var palette = []color.Color{}

func init() {
	for i := range 256 {
		g := byte(i)
		palette = append(palette, color.RGBA{g, g, g, 0xff})
	}
}

// Iterate iterates the neuron
func (n *Neuron[T]) Iterate(iterations int) {
	drop := .3
	dropout := map[string]interface{}{
		"rng":  n.rng,
		"drop": &drop,
	}

	euclidean := n.Context.B(Euclidean)
	Mul := n.Context.B(n.Context.Mul)
	Dropout := n.Context.U(n.Context.Dropout)
	Square := n.Context.U(n.Context.Square)
	Inv := n.Context.U(n.Context.Inv)
	Avg := n.Context.U(n.Context.Avg)
	Quadratic := n.Context.B(n.Context.Quadratic)
	l0 := Mul(Dropout(Square(n.Set.Get("y")), dropout),
		Inv(euclidean(n.Set.Get("x"), n.Set.Get("x"))))
	loss := Avg(Quadratic(Mul(Dropout(Square(n.Set.Get("x")), dropout),
		Inv(euclidean(n.Set.Get("y"), n.Set.Get("y")))), l0))

	var l T
	for range iterations {
		n.Set.Zero()
		l = gradient.Gradient(loss).X[0]
		_ = l
		//fmt.Println(n.Iteration, l)
		n.Set.Adam(gradient.B1, gradient.B2, Eta)
		n.Iteration++
	}
	count := 0
	image := image.NewPaletted(image.Rect(0, 0, 1024, 512), palette)
	{
		x := n.Set.ByName["x"]
		minX, maxX, minY, maxY := math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
		for i := range x.S[1] {
			x, y := any(x.X[i*x.S[0]]).(float64), any(x.X[i*x.S[0]+1]).(float64)
			if x > -1 {
				count++
			}
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
		for i := range x.S[1] {
			xx, yy := any(x.X[i*x.S[0]]).(float64), any(x.X[i*x.S[0]+1]).(float64)
			x := 500*(xx-minX)/(maxX-minX) + 6
			y := 500*(yy-minY)/(maxY-minY) + 6
			for n := -1; n < 2; n++ {
				for m := -1; m < 2; m++ {
					image.Set(n+int(x), m+int(y), color.RGBA{0xff, 0xff, 0xff, 0xff})
				}
			}
		}
	}
	{
		for i := range 512 {
			image.Set(512, int(i), color.RGBA{0xff, 0xff, 0xff, 0xff})
		}
	}
	{
		x := n.Set.ByName["y"]
		minX, maxX, minY, maxY := math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
		for i := range x.S[1] {
			x, y := any(x.X[i*x.S[0]]).(float64), any(x.X[i*x.S[0]+1]).(float64)
			if x > -1 {
				count++
			}
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
		for i := range x.S[1] {
			xx, yy := any(x.X[i*x.S[0]]).(float64), any(x.X[i*x.S[0]+1]).(float64)
			x := 500*(xx-minX)/(maxX-minX) + 6
			y := 500*(yy-minY)/(maxY-minY) + 6
			for n := -1; n < 2; n++ {
				for m := -1; m < 2; m++ {
					image.Set(n+int(x)+512, m+int(y), color.RGBA{0xff, 0xff, 0xff, 0xff})
				}
			}
		}
	}
	{
		for i := range 512 {
			for ii := range 4 {
				image.Set(int(float64(n.Iteration*i)/float64(512)), 511-ii, color.RGBA{0xff, 0xff, 0xff, 0xff})
			}
		}
	}
	n.Images.Image = append(n.Images.Image, image)
	n.Images.Delay = append(n.Images.Delay, 10)
}

func main() {
	neuron := NewNeuron[float64](2, 33)
	for range 1024 {
		neuron.Iterate(1)
	}
	out, err := os.Create("model.gif")
	if err != nil {
		panic(err)
	}
	defer out.Close()
	err = gif.EncodeAll(out, neuron.Images)
	if err != nil {
		panic(err)
	}
}
