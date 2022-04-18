import autograd.numpy as np
from autograd import grad, hessian

def func1(x):
	return x[0]**2 + x[1]**2

def func2(x):
	return 2*x[0]**3 - 5*np.sin(x[1])

def func3(x):
	return x[0]**(-2) / 3. + x[0]*np.sin(x[1]) - 8*np.cos(x[0]) / x[1]

def grad_func1(x):
	return np.array([2*x[0], 2*x[1]])

def grad_func2(x):
	return np.array([6*x[0]**2, -5*np.cos(x[1])])

def grad_func3(x):
	return np.array([-2/3.*x[0]**(-3) + np.sin(x[1]) + 8*np.sin(x[0])/x[1], x[0]*np.cos(x[1]) + 8*np.cos(x[0])/(x[1]**2)])

def hess_func1(x):
	hess_r1 = np.array([2, 0])
	hess_r2 = np.array([0, 2])
	return np.array([hess_r1, hess_r2])

def hess_func2(x):
	hess_r1 = np.array([12*x[0], 0])
	hess_r2 = np.array([0, 5*np.sin(x[1])])
	return np.array([hess_r1, hess_r2])

def hess_func3(x):
	hess_r1 = np.array([2*x[0]**(-4) + 8*np.cos(x[0])/x[1], np.cos(x[1]) - 8*np.sin(x[0])/(x[1]**2)])
	hess_r2 = np.array([np.cos(x[1]) - 8*np.sin(x[0])/(x[1]**2), -x[0]*np.sin(x[1]) - 16*np.cos(x[0])/(x[1]**3)])
	return np.array([hess_r1, hess_r2])


test_pairs = [np.array([1., 2.]), np.array([19.76, np.pi]), np.array([-9., 0.045])]
n = len(test_pairs)

print('Gradient tests\n----------------------\n')

grad1 = grad(func1)
for k in range(n):
	assert (grad1(test_pairs[k]) == grad_func1(test_pairs[k])).all(), ('func1 grad test%d: failed'%(k+1))
	print('func1 grad test%d: passed'%(k+1))

print('\n')

grad2 = grad(func2)
for k in range(n):
	assert (grad2(test_pairs[k]) == grad_func2(test_pairs[k])).all(), ('func2 grad test%d: failed'%(k+1))
	print('func2 grad test%d: passed'%(k+1))

print('\n')

grad3 = grad(func3)
for k in range(n):
	assert (np.abs(grad3(test_pairs[k]) - grad_func3(test_pairs[k])) < 1e-10).all(), ('func3 grad test%d: failed'%(k+1))
	print('func3 grad test%d: passed'%(k+1))


print('\nHessian tests\n----------------------\n')

hess1 = hessian(func1)
for k in range(n):
	assert (np.abs(hess1(test_pairs[k]) - hess_func1(test_pairs[k])) < 1e-10).all(), ('func1 hess test%d: failed'%(k+1))
	print('func1 hess test%d: passed'%(k+1))

print('\n')

hess2 = hessian(func2)
for k in range(n):
	assert (np.abs(hess2(test_pairs[k]) - hess_func2(test_pairs[k])) < 1e-10).all(), ('func2 hess test%d: failed'%(k+1))
	print('func2 hess test%d: passed'%(k+1))

print('\n')

hess3 = hessian(func3)
for k in range(n):
	assert (np.abs(hess3(test_pairs[k]) - hess_func3(test_pairs[k])) < 1e-10).all(), ('func3 hess test%d: failed'%(k+1))
	print('func3 hess test%d: passed'%(k+1))



