def deriv(func, x, step=1e-4):
    """slope of func at x, nudging a tiny step to each side (central difference)

    func: a function of one number, x: where to measure the slope
    """

    return (func(x + step) - func(x - step)) / (2 * step)


def gradient_descent(func, x0, lr, steps):
    """walk downhill on func starting from x0, to find where it's smallest

    each step moves x opposite to the slope: x = x - lr * slope
    lr: learning rate (how big each step is), steps: how many steps to take
    returns the final x and a list of the loss (func(x)) after every step
    """

    x = x0
    loss = []
    for i in range(steps):
        x = x - lr * deriv(func, x)
        loss.append(func(x))
    return x, loss