# lens design optimizer

a camera lens is a few pieces of curved glass. it takes light from one point in the scene and brings it back to one point on the sensor. if the rays land at slightly different places, the image blurs (aberration).

designing a lens means choosing the parameters that control this: how curved and how thick each piece is, how far apart the pieces sit, and which glass to use. rays behave differently at each point on the lens, so designers simulate where light lands and adjust the numbers until the blur is as small as possible. this project is a starting point for automating that whole process, built from scratch in python and numpy.

## the plan

1. simulate: trace rays of light through curved glass using snell's law (done)
2. measure: turn the blur into one number, the rms spot size (done)
3. optimize: adjust the lens shape using derivatives (gradient descent), with design rules like a fixed focal length (done)
4. tolerance: simulate thousands of slightly imperfect lenses and see how many still come out sharp (monte carlo) (done)
5. off-axis: check the whole image, not just the center, by tracing tilted bundles of rays (done)
6. diagnose: small detectors that measure each aberration (spherical, off-axis, colour) as one number
7. source: choose from a catalog of real optical glasses, including how each one splits colours
8. speed up: train a model that predicts sharpness without a full simulation, so many designs can be tried quickly

## status

steps 1 to 5 work. the ray tracer follows light through a plano-convex lens (curved front, flat back) with exact snell's law at both surfaces. rays near the axis focus where the lensmaker's equation predicts (51.67 mm for r = 25 mm, n = 1.5, t = 5 mm), and rays near the edge focus early, which is spherical aberration coming straight out of the geometry. because of that, the sharpest sensor position isn't the textbook focus: the best focus is at 50.2 mm with an rms spot size of 0.077 mm, about 2.5x sharper than at the textbook focus (0.19 mm).

the optimizer uses gradient descent to reshape the lens. with the sensor fixed at 50 mm and the front radius starting at 30 mm, it curves the surface down to r = 24.91 mm and shrinks the rms spot from 1.04 mm to 0.077 mm (about 13x sharper). that's the floor for a plano-convex lens: one curved surface can move the focus but not remove the aberration.

to go further, the tracer now handles any lens described as a list of surfaces (curved or flat, either way round), and the optimizer gets more knobs and rules. it turns the front radius and the thickness, the back radius is calculated so the focal length stays exactly 50 mm, and the thickness is kept between 2 and 6 mm. it lands on r1 = 29.5 mm, t = 6 mm (the thickness limit is active), r2 = -152.9 mm, with an rms spot of 0.068 mm, about 10% sharper than the plano-convex lens at the same focal length. that shape, curved strongly at the front and gently at the back, is what optical designers call a best-form lens.

no factory makes a lens exactly to the numbers, so step 4 checks what happens when it's built. each of the four values (r1, r2, t and the glass index n) comes out slightly off, following a normal distribution (σ = 0.2 mm, 1.0 mm, 0.1 mm and 0.002). simulating 1000 of these lenses with the sensor fixed at the design's best position (51.3 mm), about 69% (± 3%) meet a 0.08 mm spot spec. the errors don't matter equally: r1 alone fails about 24% of lenses and n about 7%, while r2 and t fail none, because the strongly curved front surface does most of the bending. so the cheap fix is to tighten only those two: r1 to σ = 0.075 mm and n to 0.001 gets the yield to 97.7%. the better fix is to let the camera refocus. the errors mostly move the focus rather than blur it, so moving the sensor to each lens's own best position makes 100% of lenses pass, even with three times the errors. that needs about ±0.7 mm of focus travel to cover 95% of lenses.

a photo isn't just its center, so step 5 checks the edges too. light from a point off to the side arrives tilted (the field angle), lands away from the middle of the sensor, and its blur is measured around where it lands. at f/2.5 the best-form lens goes from a 0.068 mm spot in the center to 0.33 mm at 10° and 0.75 mm at 15°. each angle can still focus sharply, just closer to the lens the further out it is (51.3 mm at 0°, 45.8 mm at 15°): the sharp image lies on a curved bowl while the sensor is flat (field curvature). refocusing can't fix that, because the whole photo shares one sensor. at f/5, a grid search over the lens shape and the position of the aperture stop (the opening that lets light in) finds a lens bent the other way (r1 = 100 mm, r2 = -32.7 mm) with the stop 15 mm in front of it. that makes the bowl about 3x flatter. on one flat sensor its spots stay within 0.022 to 0.038 mm from the center out to 10°, against 0.008 to 0.117 mm for the best-form lens, which cuts the average blur across the photo by 46%. it gives up some sharpness at perfect focus to get an image that's sharp everywhere. (measured in 2d; a full 3d trace would also show astigmatism, so the real gain would be smaller.)

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

```python
import numpy as np
from lens.raytrace import spot_at
from lens.design import R2_for

R1, t = 29.5, 6.0
R2 = R2_for(R1, t, 50)
rng = np.random.default_rng(42)
spots = []
for i in range(1000):                       # 1000 lenses, each slightly off
    R1_made = rng.normal(R1, 0.2)
    R2_made = rng.normal(R2, 1.0)
    t_made = rng.normal(t, 0.1)
    n_made = rng.normal(1.5, 0.002)
    lens = [(0, R1_made, n_made), (t_made, R2_made, 1.0)]
    spots.append(spot_at(lens, 51.3))       # sensor fixed at the design's best position
print(np.mean(np.array(spots) < 0.08))      # yield: about 0.69
```

```python
import numpy as np
from lens.raytrace import spot_at
from lens.design import R2_for

f5 = np.linspace(-5, 5, 1001)                            # rays through a 10 mm opening (f/5)
lens = [(0, 100, 1.5), (6, R2_for(100, 6, 50), 1.0)]     # bent for a flat field
for deg in [0, 5, 10]:                                   # center, middle, edge of the photo
    print(deg, spot_at(lens, 53.85, np.radians(deg), f5, z_stop = -15))   # about 0.028, 0.022, 0.038 mm
```

## layout

```
lens/raytrace.py    ray tracing (any lens as a list of surfaces, at any field angle and stop position) and spot size
lens/optimize.py    derivatives and gradient descent
lens/design.py      design rules, like keeping the focal length fixed
notebooks/          experiments and plots, one per step
```