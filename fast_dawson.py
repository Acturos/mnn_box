#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020 Yang Qi, ISTBI, Fudan University China


import numpy as np
import mpmath as mpm
from scipy.special import erfcx, gamma, erfi, erfc, dawsn
from scipy.integrate import quad
import matplotlib.pyplot as plt
import time


class Coeffs():
    def __init__(self):
        self.asym_neginf = 0
        self.asym_posinf = 0
        self.taylor = 0
        self.int_asym_neginf = 0

    def contfrac(self, A):
        '''Convert summation coefficients to continued fraction'''
        B = np.append(A[0], A[1:] / A[:-1])
        return B


class Dawson1:
    def __init__(self):
        '''1st order Dawson function and its integral'''
        self.coef = Coeffs()
        self.coef.cheb = self.chebyshev()
        return

    def dawson1(self, x):
        '''Compute Dawson function with existing library'''
        y = erfcx(-x) * np.sqrt(np.pi) / 2
        return y

    def int_fast(self, x):
        '''fast approximation'''
        region1 = x <= -2.5
        region3 = x > 2.5
        region2 = ~(region1 | region3)
        y = np.zeros(x.size)
        y[region1] = self.int_asym_neginf(x[region1])
        y[region3] = self.int_asym_posinf(x[region3])
        y[region2] = np.polynomial.chebyshev.chebval(x[region2], self.coef.cheb)
        return y

    def int_exact(self, x):
        '''Integral of dawson function: analytical solution'''
        # Evaluated with arbitrary precision arithmetic
        # 50 times faster than direct integration; can still suffer from numeric overflow
        y = np.zeros(x.size)
        i = 0
        for t in x:  # run a loop since mpm doesn't support vectorization
            y[i] = 0.5 * t * t * float(mpm.hyp2f2(1, 1, 3 / 2, 2, t * t)) + 0.25 * np.pi * erfi(t)
            i += 1
        return y

    def int_asym_posinf(self, x, N=7):
        '''Compute asymptotic expansion of the indefinite integral of g(x) for x>>1'''
        h = 0
        for k in range(N):
            h += gamma(0.5 + k) * np.power(x, -2 * k - 1)
        h = 0.5 * np.exp(x * x) * h
        h += 0.25 * (-np.euler_gamma - np.log(4))
        return h

    def int_asym_neginf(self, x):
        '''Compute asymptotic expansion of the indefinite integral of g(x) for x<<-1'''
        A = [-1 / 8, 3 / 32, -5 / 32]
        h = 0.25 * (-np.euler_gamma - np.log(4)) - 0.5 * np.log(
            -x)  # - 0.25*np.real(np.log(-x*x+0j)), just keep the real part
        k = 2
        for a in A:
            h += a * np.power(x, -k)
            k += 2
        return h

    def chebyshev(self, d=20):
        '''Fit with Chebyshev polynomial'''
        x = np.arange(-2.5, 2.5, 1e-2)
        y = self.int_exact(x)
        c = np.polynomial.chebyshev.chebfit(x, y, d)

        return c

    def asym_neginf(self, x, N=7):
        '''Evaluate Dawson function with asymptotic expansion. Works well for x<<-3 '''
        k = np.arange(N)
        h = 0
        for k in range(N):
            h += np.power(x, -1 - 2 * k) * self.coef.asym_neginf[k]
        return h

    def int_brute_force(self, X):
        '''2nd order Dawson function (direct integration)'''
        q = np.zeros(X.size)
        i = 0
        for x in X:
            q[i], _ = quad(lambda x: erfcx(-x), 0, x)
            i += 1
        q = q * np.sqrt(np.pi) / 2
        return q

    def diagnosis(self):
        # x = np.arange(-10,-3.5,1e-2)
        x = np.arange(-3, 3, 1e-3)

        tic = time.time()
        # z = self.asym_neginf(x)
        # z = self.int(x)
        z = self.int_asym_posinf(x[x > 1.5])
        w = self.int_asym_neginf(x[x < -0.5])
        y = np.polynomial.chebyshev.chebval(x, self.coef.cheb)
        print('Time for evaluating approximation: {:.2E}'.format(time.time() - tic))

        tic = time.time()
        # q = self.dawson1(x)
        q = self.int_exact(x)
        # q  = self.int_brute_force(x)
        print('Time for evaluating integral: {:.2E}'.format(time.time() - tic))

        plt.plot(x, q)
        plt.plot(x[x > 1.5], z, '--')
        plt.plot(x[x < -0.5], w, '--')
        plt.plot(x, y, '--')
        # plt.semilogy(x,q)
        # plt.semilogy(x,z,'--')

        plt.legend(['Analytical', 'Asymptote at +inf', 'Asymptote at -inf', 'Chebyshev'])
        plt.ylim([-5, 50])
        plt.xlabel('x')
        plt.ylabel('G(x)')
        plt.title('Integral of g(x)')

        plt.show()

    def speed_test(self):
        '''Over all speed test'''
        N = int(2e3)
        xmax = 5
        x = xmax * (np.random.rand(N) - 0.5)

        T = {}

        tic = time.perf_counter()
        erfcx(x)
        T['Benchmark (erfcx)'] = time.perf_counter() - tic

        tic = time.perf_counter()
        self.int_brute_force(x)
        T['Brute force integration'] = time.perf_counter() - tic

        tic = time.perf_counter()
        self.int_exact(x)
        T['Simplified integration'] = time.perf_counter() - tic

        tic = time.perf_counter()
        self.int_fast(x)
        T['Fast approximation'] = time.perf_counter() - tic

        rep = ['Speed Test Result', 'Number of samples: {}'.format(N), 'Sample Range: [-{},{}]'.format(xmax, xmax)]
        rep += ['Time Elapsed | Relative to benchmark']
        for k in T:
            rep.append('{}: {:.1e} | {:.1e}'.format(k, T[k], T[k] / T['Benchmark (erfcx)']))

        print('\n'.join(rep))

    def precision_test(self):
        """Over all precision test"""
        x = np.arange(-5, 5, 0.1)
        x = x[np.abs(x) > 1e-4]
        G0 = self.int_exact(x)
        G = self.int_fast(x)
        plt.plot(x, (G - G0) / G0)
        plt.show()


class Dawson2:
    def __init__(self, N=30):
        '''Provide 2nd order Dawson function and their integrals'''
        self.dawson1 = Dawson1().dawson1
        self.N = N  # truncation
        # pre-computed asymptotic expansion coefficients
        self.coef = Coeffs()
        self.coef.asym_neginf = np.array(
            [-1 / 8, 5 / 16, -1, 65 / 16, -2589 / 128, 30669 / 256, -52779 / 64, 414585 / 64,
             -117193185 / 2048, 2300964525 / 4096, -6214740525 / 1024, 293158982025 / 4096,
             -29981616403725 / 32768, 826063833097125 / 65536, -1525071991827825 / 8192,
             12020398467838425 / 4096, -25784897051958192225 / 524288, 915566919983318035125 / 1048576,
             -2145833659489095662625 / 131072, 338972721447561521945625 / 1048576])
        # self.coef.asym_posinf = 1# leading term is exp(x^2)
        # self.coef.taylor = self.taylor(N)#
        self.coef.cheb = self.chebyshev(self.brute_force, -2.5, 2, 25)
        self.coef.cheb_int = self.chebyshev(self.int_exact, -3, 2, 25)

    def dawson2(self, x):
        """
		2nd order Dawson function (fast approximation)
		"""
        region1 = x <= -2.5
        region3 = x > 2
        region2 = ~(region1 | region3)
        y = np.zeros(x.size)
        y[region1] = self.asym_neginf(x[region1])
        y[region3] = self.asym_posinf(x[region3])
        y[region2] = np.polynomial.chebyshev.chebval(x[region2], self.coef.cheb)
        return y

    def int_fast(self, x):
        """2nd order Dawson function (fast approximation)"""
        region1 = x <= -3
        region3 = x > 2
        region2 = ~(region1 | region3)
        y = np.zeros(x.size)
        y[region1] = self.int_asym_neginf(x[region1])
        y[region3] = self.int_asym_posinf(x[region3])
        y[region2] = np.polynomial.chebyshev.chebval(x[region2], self.coef.cheb_int)
        return y

    def int_brute_force(self, X):
        '''Integral of the 2nd order Dawson function (direct integration)'''
        q = np.zeros(X.size)
        i = 0
        fun = lambda x: quad(lambda y: np.exp((x + y) * (x - y)) * (self.dawson1(y) ** 2), -np.inf, x)[0]
        for x in X:
            q[i], _ = quad(fun, -np.inf, x)
            i += 1
        return q

    def int_exact(self, X):
        q = np.zeros(X.size)
        i = 0
        fun1 = lambda x: np.power(erfcx(-x), 2) * dawsn(x)
        fun2 = lambda x: np.exp(-x * x) * np.power(erfcx(-x), 2)
        for x in X:
            y1, _ = quad(fun1, -np.inf, x)
            y2, _ = quad(fun2, -np.inf, x)
            q[i] = -np.pi / 4 * y1 + np.power(np.sqrt(np.pi) / 2, 3) * erfi(x) * y2
            i += 1
        return q

    def brute_force(self, X):
        '''2nd order Dawson function (direct integration)'''
        q = np.zeros(X.size)
        i = 0
        for x in X:
            q[i], _ = quad(lambda y: np.exp((x + y) * (x - y)) * (self.dawson1(y) ** 2), -np.inf, x)
            i += 1
        return q

    def dawson2_taylor(self, x, N=10):  #
        y = 0
        for i in range(N):
            y += self.coef.taylor[i] * np.power(x, i)
        return y

    def taylor(self, N):
        '''Compute coefficients of Taylor expansion near 0. Not useful in practice.'''
        G = np.zeros(N)
        G2 = G.copy()  # g^2
        H = G.copy()
        G[0] = np.sqrt(np.pi) / 2
        G2[0] = np.pi / 4
        H[0] = np.sqrt(np.pi) * np.log(2) / 4

        G[1] = 1
        G2[1] = 2 * G[0]
        H[1] = np.pi / 4

        for i in range(N - 2):
            G[i + 2] = 2 * (i + 1) * G[i]
            G2[i + 2] = 4 * (i + 1) * G2[i] + 2 * G[i + 1]
            H[i + 2] = 2 * (i + 1) * H[i] + G2[i + 1]
        p = np.arange(N)  # power = 0,1,2,3...
        H = np.array(H) / gamma(p + 1)

        return H

    def chebyshev(self, fun, xmin, xmax, d):
        '''Fit a function with Chebyshev polynomial'''
        # x = np.arange(-3,2,1e-2)
        x = np.arange(xmin, xmax, 1e-2)
        # y = self.brute_force(x)
        # y = self.int_exact(x) #fast direct integration
        y = fun(x)
        c = np.polynomial.chebyshev.chebfit(x, y, d)
        return c

    def asym_neginf(self, x, N=7):
        '''Asymptotic expansion of H(x) as x-->-Inf. Works well for x<<-3'''
        # WARNING: truncate the expansion at N=7 is good. Larger truncation inreases error so don't change it.
        # Continued fraction doesn't seem to make a big difference on modern hardware.
        h = 0
        for k in range(N):
            h += np.power(x, -3 - 2 * k) * self.coef.asym_neginf[k]
        return h

    def asym_posinf(self, x):
        '''Asymptotic expansion of H(x) as x-->+Inf.'''
        h = np.power(np.sqrt(np.pi) / 2, 3) * np.exp(x * x)
        h *= np.power(erfc(-x), 2) * erfi(x)
        return h

    def int_asym_neginf(self, x, N=7):
        '''Evaluate integral of the 2nd order Dawson function with asymptotic expansion. Works well for x<<-3 '''
        h = 0
        for k in range(N):
            h += np.power(x, -2 - 2 * k) * self.coef.asym_neginf[k] / (-2 - 2 * k)
        return h

    def int_asym_posinf(self, x):

        E1 = erfi(x)
        E2 = np.power(erfc(-x), 2)
        a = np.pi ** 2 / 32
        H = a * (E1 - 1) * E1 * E2

        return H

    def contfrac(self, R):
        '''Evaluate continued fraction using the naive method.'''
        # There are better ways, e.g. Lenz's method but for now the naive method suffices.
        # INPUT: terms in the continued fraction
        n = len(R)
        cf = 0  # initialize continued fraction (innermost term)
        for r in reversed(R[1:]):  # work outwards
            cf = r / (1 + r - cf)
        cf = R[0] / (1 - cf)  # outermost term
        return cf

    def diagnosis(self):
        x = np.arange(-3.5, 2.5, 1e-2)
        tic = time.time()
        z = self.int_asym_neginf(x[x < -1])
        y = self.int_asym_posinf(x[x > 0.8])

        print('Time for evaluating asymptote: {:.2E}'.format(time.time() - tic))
        tic = time.time()
        # q = self.int_brute_force(x)
        q = self.int_exact(x)
        # q = self.brute_force(x)
        print('Time for evaluating integral: {:.2E}'.format(time.time() - tic))
        tic = time.time()
        cheb = np.polynomial.chebyshev.chebval(x, self.coef.cheb_int)
        print('Time for evaluating chebyshev approximation: {:.2E}'.format(time.time() - tic))

        plt.semilogy(x, q)
        plt.semilogy(x[x > 0.8], y, '--')
        plt.semilogy(x[x < -1], z, '--')
        plt.semilogy(x, cheb, '--')

        # plt.plot(x,q)
        # plt.plot(x[x>1],y,'--')
        # plt.plot(x[x<-1],z,'--')
        # plt.plot(x, cheb,'--')

        plt.legend(['Analytical', 'Asymptote at +inf', 'Asymptote at -inf', 'Chebyshev'])
        plt.xlabel('x')
        plt.ylabel('H(x)')
        plt.show()

    def speed_test(self):
        '''Over all speed test'''
        N = int(2e3)
        xmax = 5
        x = xmax * (np.random.rand(N) - 0.5)

        T = {}

        tic = time.perf_counter()
        erfcx(x)
        T['Benchmark (erfcx)'] = time.perf_counter() - tic

        tic = time.perf_counter()
        self.int_brute_force(x)
        T['Brute force integration'] = time.perf_counter() - tic

        tic = time.perf_counter()
        self.int_exact(x)
        T['Simplified integration'] = time.perf_counter() - tic

        tic = time.perf_counter()
        self.int_fast(x)
        T['Fast approximation'] = time.perf_counter() - tic

        rep = ['Speed Test Result', 'Number of samples: {}'.format(N), 'Sample Range: [-{},{}]'.format(xmax, xmax)]
        rep += ['Time Elapsed | Relative to benchmark']
        for k in T:
            rep.append('{}: {:.1e} | {:.1e}'.format(k, T[k], T[k] / T['Benchmark (erfcx)']))

        print('\n'.join(rep))

        return

    def precision_test(self):
        '''Over all precision test'''
        x = np.arange(-5, 5, 0.1)
        # H1 = self.int_brute_force(x)
        H0 = self.int_exact(x)
        H = self.int_fast(x)
        h0 = self.brute_force(x)
        h = self.dawson2(x)

        plt.plot(x, (H - H0) / H0, x, (h - h0) / h0)
        plt.show()

        return


if __name__ == "__main__":
    # demo
    ds1 = Dawson1()
    ds2 = Dawson2()
    x = np.arange(-3, 3, 0.01)
    H = ds2.int_fast(x)  # h(x)
    h = ds2.dawson2(x)  # H(x)
    G = ds1.int_fast(x)
    g = ds1.dawson1(x)

    plt.semilogy(x, g, x, G, x, h, x, H)
    plt.xlabel('x')
    plt.legend(['g(x)', 'G(x)', 'h(x)', 'H(x)'])
    plt.show()

# ds = Dawson1()
# ds.speed_test()
# ds.precision_test()
# ds.diagnosis()

# ds = Dawson2()

# ds.diagnosis()
# ds.speed_test()
# ds.precision_test()
