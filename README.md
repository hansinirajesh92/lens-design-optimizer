# lens design optimizer

a camera lens is a few pieces of curved glass. it takes light from one point in the scene and brings it back to one point on the sensor. if the rays land at slightly different places, the image blurs (aberration).

designing a lens means choosing the parameters that control this: how curved and how thick each piece is, how far apart the pieces sit, and which glass to use. rays behave differently at each point on the lens, so designers simulate where light lands and adjust the numbers until the blur is as small as possible. this project is a starting point for automating that whole process, built from scratch in python and numpy.

## the plan

1. simulate: trace rays of light through curved glass using snell's law (done)
2. measure: turn the blur into one number, the rms spot size (done)
3. optimize: adjust the lens shape using derivatives (gradient descent)
4. tolerance: simulate thousands of slightly imperfect lenses and see how many still come out sharp (monte carlo)
5. source: choose from a catalog of real optical glasses
6. speed up: train a model that predicts sharpness without a full simulation, so many designs can be tried quickly

## status

steps 1 and 2 work. the ray tracer follows light through a plano-convex lens (curved front, flat back) with exact snell's law at both surfaces. rays near the axis focus where the lensmaker's equation predicts (51.67 mm for r = 25 mm, n = 1.5, t = 5 mm), and rays near the edge focus early, which is spherical aberration coming straight out of the geometry.

because of that aberration, the sharpest place for the sensor isn't the textbook focus. scanning 151 sensor positions with 1001 rays, the best focus is at 50.2 mm with an rms spot size of 0.077 mm, about 2.5x sharper than at the textbook focus (0.19 mm).

## running it

```
git clone https://github.com/hansinirajesh92/lens-design-optimizer.git
cd lens-design-optimizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```python
from lens.raytrace import trace_lens, cross_axis_back

y2, m2 = trace_lens(6, 25, 1.5, 5)   # ray at 6 mm through a lens: r = 25 mm, n = 1.5, t = 5 mm
cross_axis_back(y2, m2, 5)           # 50.83 mm
```

## layout

```
lens/raytrace.py    ray tracing functions
notebooks/          experiments and plots, one per step
```