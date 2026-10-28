# Lens Design Optimizer

The problem: A camera lens is a few pieces of curved galss - it takes light from one point in the scene and brings its back to one point in the sensor. If the rays land at slightly different places - there is a blur (aberration)

So, the idea: is to help choose the metrics that control the designing of lens - the curvature and the thickness of lenses, how far apart they sit, which glass to use, etc.,

Rays behave differently at each point on the lens. So, the designers simulate where light lands and adjusts numbers to achieve minimal blur. This is project is a starting point in automating this whole ordeal.

Step-by-Step:

1) Simulate rays of light through curved glass
2) Measure blurriness (spot size)
3) Optimize to adjust lens shape using derivatives (gradient descent)
4) Check manufacturability by simulating 1000s of slightly imperfect lenses and see how many come out with sharp images (Monte Carlo)
5) Speed process up by training a model, without full simulations, to try designs quickly.
