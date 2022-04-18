import autograd.numpy as np
from autograd import grad, hessian
import wolf


# squares the output of func1
def sq_func1(x):
	out = wolf.func1(x)
	return out**2


x = np.array([2., 2.])

