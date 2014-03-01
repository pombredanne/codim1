import numpy as np
import copy
import gaussian_quad

"""
Some of the manual input of quadrature rules in this file is super ugly. But,
it works. Maybe one day, it would be nice to export to some files.
"""
class QuadGauss(object):
    """
    Simple wrapper of gaussian quadrature. Returns points and weights for
    quadrature on [0, 1].
    """
    def __init__(self, N):
        self.N = N
        self.x, self.w = gaussian_quad.gaussxwab(self.N, 0.0, 1.0)

class QuadGaussLog(object):
    """
    Quadrature points and weights for \int_0^1 \log(x) * f(x) dx
    Note that the weights include the log(x) term.
    """
    def __init__(self, N, flip = False):
        self.N = N
        self.x = np.empty(N)
        self.w = np.empty(N)
        self.setup_points()
        if flip:
            self.x = 1.0 - self.x

    def setup_points(self):
        if self.N > 11:
            raise Exception("Gaussian log weighted quadrature is not " +
                    "implemented for that order.")
        if self.N == 1:
            self.x[0] = 0.3333333333333333
        elif self.N == 2:
            self.x[0] = 0.1120088061669761
            self.x[1] = 0.6022769081187381
        elif self.N == 3:
            self.x[0] = 0.06389079308732544
            self.x[1] = 0.3689970637156184
            self.x[2] = 0.766880303938942
        elif self.N == 4:
            self.x[0] = 0.04144848019938324
            self.x[1] = 0.2452749143206022
            self.x[2] = 0.5561654535602751
            self.x[3] = 0.848982394532986
        elif self.N == 5:
            self.x[0] = 0.02913447215197205
            self.x[1] = 0.1739772133208974
            self.x[2] =  0.4117025202849029
            self.x[3] = 0.6773141745828183
            self.x[4] = 0.89477136103101
        elif self.N == 6:
            self.x[0] = 0.02163400584411693
            self.x[1] = 0.1295833911549506
            self.x[2] = 0.3140204499147661
            self.x[3] = 0.5386572173517997
            self.x[4] = 0.7569153373774084
            self.x[5] = 0.922668851372116
        elif self.N == 7:
            self.x[0] = 0.0167193554082585
            self.x[1] = 0.100185677915675
            self.x[2] = 0.2462942462079286
            self.x[3] = 0.4334634932570557
            self.x[4] = 0.6323509880476823
            self.x[5] = 0.81111862674023
            self.x[6] = 0.940848166743287
        elif self.N == 8:
            self.x[0] = 0.01332024416089244
            self.x[1] = 0.07975042901389491
            self.x[2] = 0.1978710293261864
            self.x[3] =   0.354153994351925
            self.x[4] =   0.5294585752348643
            self.x[5] = 0.7018145299391673
            self.x[6] = 0.849379320441094
            self.x[7] = 0.953326450056343
        elif self.N == 9:
            self.x[0] = 0.01086933608417545
            self.x[1] = 0.06498366633800794
            self.x[2] = 0.1622293980238825
            self.x[3] = 0.2937499039716641
            self.x[4] = 0.4466318819056009
            self.x[5] = 0.6054816627755208
            self.x[6] = 0.7541101371585467
            self.x[7] = 0.877265828834263
            self.x[8] = 0.96225055941096
        elif self.N == 10:
            self.x[0] = 0.00904263096219963
            self.x[1] = 0.05397126622250072
            self.x[2] =  0.1353118246392511
            self.x[3] = 0.2470524162871565
            self.x[4] = 0.3802125396092744
            self.x[5] = 0.5237923179723384
            self.x[6] = 0.6657752055148032
            self.x[7] = 0.7941904160147613
            self.x[8] = 0.898161091216429
            self.x[9] = 0.9688479887196
        elif self.N == 11:
            self.x[0] = 0.007643941174637681
            self.x[1] = 0.04554182825657903
            self.x[2] = 0.1145222974551244
            self.x[3] = 0.2103785812270227
            self.x[4] = 0.3266955532217897
            self.x[5] = 0.4554532469286375
            self.x[6] = 0.5876483563573721
            self.x[7] = 0.7139638500230458
            self.x[8] = 0.825453217777127
            self.x[9] = 0.914193921640008
            self.x[10] = 0.973860256264123

        if self.N == 1:
            self.w[0] = -1.0
        elif self.N == 2:
            self.w[0] = -0.7185393190303845
            self.w[1] = -0.2814606809696154
        elif self.N == 3:
            self.w[0] = -0.5134045522323634
            self.w[1] = -0.3919800412014877
            self.w[2] = -0.0946154065661483
        elif self.N == 4:
            self.w[0] =-0.3834640681451353
            self.w[1] =-0.3868753177747627
            self.w[2] =-0.1904351269501432
            self.w[3] =-0.03922548712995894
        elif self.N == 5:
            self.w[0] =-0.2978934717828955
            self.w[1] =-0.3497762265132236
            self.w[2] =-0.234488290044052
            self.w[3] =-0.0989304595166356
            self.w[4] =-0.01891155214319462
        elif self.N == 6:
            self.w[0] = -0.2387636625785478
            self.w[1] = -0.3082865732739458
            self.w[2] = -0.2453174265632108
            self.w[3] = -0.1420087565664786
            self.w[4] = -0.05545462232488041
            self.w[5] = -0.01016895869293513
        elif self.N == 7:
            self.w[0] = -0.1961693894252476
            self.w[1] = -0.2703026442472726
            self.w[2] = -0.239681873007687
            self.w[3] = -0.1657757748104267
            self.w[4] = -0.0889432271377365
            self.w[5] = -0.03319430435645653
            self.w[6] = -0.005932787015162054
        elif self.N == 8:
            self.w[0] = -0.164416604728002
            self.w[1] = -0.2375256100233057
            self.w[2] = -0.2268419844319134
            self.w[3] = -0.1757540790060772
            self.w[4] = -0.1129240302467932
            self.w[5] = -0.05787221071771947
            self.w[6] = -0.02097907374214317
            self.w[7] = -0.003686407104036044
        elif self.N == 9:
            self.w[0] = -0.1400684387481339
            self.w[1] = -0.2097722052010308
            self.w[2] = -0.211427149896601
            self.w[3] = -0.1771562339380667
            self.w[4] = -0.1277992280331758
            self.w[5] = -0.07847890261203835
            self.w[6] = -0.0390225049841783
            self.w[7] = -0.01386729555074604
            self.w[8] = -0.002408041036090773
        elif self.N == 10:
            self.w[0] = -0.12095513195457
            self.w[1] = -0.1863635425640733
            self.w[2] = -0.1956608732777627
            self.w[3] = -0.1735771421828997
            self.w[4] = -0.135695672995467
            self.w[5] = -0.0936467585378491
            self.w[6] = -0.05578772735275126
            self.w[7] = -0.02715981089692378
            self.w[8] = -0.00951518260454442
            self.w[9] = -0.001638157633217673
        elif self.N == 11:
            self.w[0] = -0.1056522560990997
            self.w[1] = -0.1665716806006314
            self.w[2] = -0.1805632182877528
            self.w[3] = -0.1672787367737502
            self.w[4] = -0.1386970574017174
            self.w[5] = -0.1038334333650771
            self.w[6] = -0.06953669788988512
            self.w[7] = -0.04054160079499477
            self.w[8] = -0.01943540249522013
            self.w[9] = -0.006737429326043388
            self.w[10] = -0.001152486965101561


class QuadGaussLogR(object):
    """
    Quadrature points and weights for integrating a function with
    log(|x - x0| / q)
    singularity on the interval [0, 1].
    """
    def __init__(self, N, q, x0):
        self.N = N
        self.q = q
        self.x0 = x0
        if self.x0 == 0 or self.x0 ==0:
            self.split = 1.0
        else:
            self.split = self.x0

        qlog = QuadGaussLog(N)
        qplain = QuadGauss(N)
        self.w = qlog.w * self.split
        self.x = qlog.x * self.split
        if self.split != 1.0 or self.q != 1.0:
            self.x = np.append(self.x, qplain.x * self.split)
            self.w = np.append(self.w,
                    -np.log(self.q / self.split) * qplain.w * self.split)
        if self.split != 1.0:
            self.x = np.append(self.x, qlog.x * (1 - self.split) + self.split)
            self.w = np.append(self.w, qlog.w * (1 - self.split))
            self.x = np.append(self.x, qplain.x * (1 - self.split) + self.split)
            self.w = np.append(self.w, -np.log(self.q / (1 - self.split))
                    * qplain.w * (1 - self.split))
        self.w /= np.log(np.abs(self.x - self.x0)) / self.q;


def test_QuadGauss():
    f = lambda x: 3 * x ** 2
    F = lambda x: x ** 3
    exact = F(1) - F(0)
    q = QuadGauss(2)
    est = np.sum([w_val * f(x_val) for (w_val, x_val) in zip(q.w, q.x)])
    np.testing.assert_almost_equal(exact, est)

def test_QuadGaussLog1():
    # f = log(x) * 1.0, but log(x) is included in the weights
    # integral of log(x) from 0 to 1 = -1.0
    exact = -1.0
    q = QuadGaussLog(2)
    est = np.sum([w_val for (w_val, x_val) in zip(q.w, q.x)])
    np.testing.assert_almost_equal(exact, est)

def test_QuadGaussLog2():
    f = lambda x: x ** 7
    exact = -1.0 / 64.0
    # 2n - 1 = 7 ==> n = 4
    q = QuadGaussLog(4)
    est = np.sum([w_val * f(x_val) for (w_val, x_val) in zip(q.w, q.x)])
    np.testing.assert_almost_equal(exact, est)

def test_QuadGaussLog3():
    f = lambda x: x
    exact = -0.75
    q = QuadGaussLog(2, True)
    est = np.sum([w_val * f(x_val) for (w_val, x_val) in zip(q.w, q.x)])
    np.testing.assert_almost_equal(exact, est)

def test_QuadGaussLogR():
    f = lambda r: np.log(np.abs(r - 0.5))
    exact = -1.0 - np.log(2.0)
    q = QuadGaussLogR(1, 1.0, 0.5)
    est = np.sum([w_val * f(x_val) for (w_val, x_val) in zip(q.w, q.x)])
    np.testing.assert_almost_equal(exact, est)



