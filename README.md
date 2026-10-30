# lens design optimizer

a camera lens is a few pieces of curved glass. it takes light from one point in the scene and brings it back to one point on the sensor. if the rays land at slightly different places, the image blurs (aberration).

designing a lens means choosing the parameters that control this: how curved and how thick each piece is, how far apart the pieces sit, and which glass to use. rays behave differently at each point on the lens, so designers simulate where light lands and adjust the numbers until the blur is as small as possible. this project is a starting point for automating that whole process, built from scratch in python and numpy.

## the plan

1. simulate: trace rays of light through curved glass using snell's law (done)
2. measure: turn the blur into one number, the rms spot size (done)
3. optimize: adjust the lens shape using derivatives (gradient descent), with design rules like a fixed focal length (done)
4. tolerance: simulate thousands of slightly imperfect lenses and see how many still come out sharp (monte carlo)
5. source: choose from a catalog of real optical glasses
6. speed up: train a model that predicts sharpness without a full simulation, so many designs can be tried quickly

## status

steps 1 to 3 work. the ray tracer follows light through a plano-convex lens (curved front, flat back) with exact snell's law at both surfaces. rays near the axis focus where the lensmaker's equation predicts (51.67 mm for r = 25 mm, n = 1.5, t = 5 mm), and rays near the edge focus early, which is spherical aberration coming straight out of the geometry. because of that, the sharpest sensor position isn't the textbook focus: the best focus is at 50.2 mm with an rms spot size of 0.077 mm, about 2.5x sharper than at the textbook focus (0.19 mm).

the optimizer uses gradient descent to reshape the lens. with the sensor fixed at 50 mm and the front radius starting at 30 mm, it curves the surface down to r = 24.91 mm and shrinks the rms spot from 1.04 mm to 0.077 mm (about 13x sharper). that's the floor for a plano-convex lens: one curved surface can move the focus but not remove the aberration.

to go further, the tracer now handles any lens described as a list of surfaces (curved or flat, either way round), and the optimizer gets more knobs and rules. it turns the front radius and the thickness, the back radius is calculated so the focal length stays exactly 50 mm, and the thickness is kept between 2 and 6 mm. it lands on r1 = 29.5 mm, t = 6 mm (the thickness limit is active), r2 = -152.9 mm, with an rms spot of 0.068 mm, about 10% sharper than the plano-convex lens at the same focal length. that shape, curved strongly at the front and gently at the back, is what optical designers call a best-form lens.

## running it

```
git clone https://github.com/hansinirajesh92/lens-design-optimizer.git
cd lens-design-optimizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```python
import numpy as np
from lens.raytrace import best_spot
from lens.design import R2_for

lens = [(0, 29.5, 1.5), (6, R2_for(29.5, 6, 50), 1.0)]   # best-form lens: front, then back
print(best_spot(lens))                                     # best focus (mm), rms spot (mm)
```

```python
import numpy as np
from lens.raytrace import rms_spot
from lens.optimize import gradient_descent

heights = np.linspace(-10, 10, 1001)        # 1001 rays

def spot_at_50(R):                          # sensor fixed at 50 mm, n = 1.5, t = 5 mm
    return rms_spot(50, heights, R, 1.5, 5)

R, loss = gradient_descent(spot_at_50, 30, 2, 60)
print(R, loss[-1])                          # about 24.91 mm and 0.077 mm
```

## layout

```
lens/raytrace.py    ray tracing (any lens as a list of surfaces) and spot size
lens/optimize.py    derivatives and gradient descent
lens/design.py      design rules, like keeping the focal length fixed
notebooks/          experiments and plots, one per step
```