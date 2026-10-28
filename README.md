# lens design optimizer

a camera lens is a few pieces of curved glass. it takes light from one point in the scene and brings it back to one point on the sensor. if the rays land at slightly different places, the image blurs (aberration).

designing a lens means choosing the parameters that control this: how curved and how thick each piece is, how far apart the pieces sit, and which glass to use. rays behave differently at each point on the lens, so designers simulate where light lands and adjust the numbers until the blur is as small as possible. this project is a starting point for automating that whole process, built from scratch in python and numpy.

## the plan

1. simulate: trace rays of light through curved glass using snell's law
2. measure: turn the blur into one number, the spot size
3. optimize: adjust the lens shape using derivatives (gradient descent)
4. tolerance: simulate thousands of slightly imperfect lenses and see how many still come out sharp (monte carlo)
5. source: choose from a catalog of real optical glasses
6. speed up: train a model that predicts sharpness without a full simulation, so many designs can be tried quickly

## status

step 1 works for a single spherical surface. rays near the axis focus within 0.05% of the textbook focal length, and rays near the edge focus early, which is spherical aberration coming straight out of the geometry. a full lens (front and back surface) is in progress.

## running it

```
git clone https://github.com/hansinirajesh92/lens-design-optimizer.git
cd lens-design-optimizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```python
from lens.raytrace import crossing

crossing(6, 25, 1.0, 1.5)   # ray at height 6 mm, surface radius 25 mm, air into glass
# 74.03 mm (the textbook focus for near-axis rays is 75 mm)
```

## layout

```
lens/raytrace.py    ray tracing functions
notebooks/          experiments and plots
```