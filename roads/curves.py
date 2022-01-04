import csv
import json

import matplotlib.pyplot as plt
from pyclothoids import Clothoid

from .geometry import *
from .utils import range_t

__all__ = [
    'saveReport',
    'saveCurves',
    'loadCurves',
    'printCurves',
    'prepareCurves',
    'Curve',
    'Spiral',
    'Circle'
]


def saveReport(curves, filename, path):
    print("Analyzing curve error and writing report")
    with open(filename, "wt") as f:
        writer = csv.writer(f)
        row = ['Type', 'Length (m)', 'Curvature (1/m)', 'Radius (m)', 'Parameter A', 'Mean square error (m^2)',
               'Std dev (m)', 'Max error (m)']
        writer.writerow(row)

        tot_len = 0
        for c in curves:
            mse, maxe, sdev = c.computeErrors()
            slen = c.length
            if isinstance(c, Line):
                row = ['Line', f"{slen:.2f}", 0, 'inf', '', f"{mse:.2f}", f"{sdev:.2f}", f"{maxe:.2f}"]
            elif isinstance(c, Circle):
                row = ['Arc', f"{slen:.2f}", f"{c.curvature0:.4f}", f"{c.radius:.2f}", '', f"{mse:.2f}", f"{sdev:.2f}",
                       f"{maxe:.2f}"]
            elif isinstance(c, Spiral):
                curvature = c.curvature0
                if abs(curvature) < 0.000001:
                    radius = ""
                else:
                    radius = abs(1.0 / curvature)
                    radius = f"{radius:.2f}"
                row = ['Spiral', f"{slen:.2f}", f"{curvature:.4f}", radius, f"{c.parameterA:.3f}", f"{mse:.2f}",
                       f"{sdev:.2f}", f"{maxe:.2f}"]
            tot_len += slen

            writer.writerow(row)
        row = ['Total:', f"{tot_len:.2}m"]
        writer.writerow(row)
    print("Done writing report")


def saveCurves(curves, filename):
    cdata = []
    for c in curves:
        cdata.append(c.serialize())

    with open(filename, "wt") as f:
        json.dump(cdata, f, indent=4)


def loadCurves(path, filename):
    with open(filename, "rt") as f:
        cdata = json.load(f)

    classes = {
        Circle.__name__: Circle,
        Line.__name__: Line,
        Spiral.__name__: Spiral
    }

    curves = []

    for dat in cdata:
        cls = classes[dat['type']]
        curves.append(cls.deserialize(path, dat))

    prepareCurves(curves)

    return curves


def printCurves(curves):
    for c in curves:
        c.print()


def prepareCurves(curves):
    tot_curves = len(curves)
    m = tot_curves - 1
    for i in range(tot_curves):
        pr = curves[i - 1] if i > 0 else None
        nx = curves[i + 1] if i < m else None
        curves[i].setAdjacentCurves(pr, nx)

    for c in curves:
        if isinstance(c, Spiral):
            continue
        c.setParams(c.t0, c.t1)

    for c in curves:
        if not isinstance(c, Spiral):
            continue
        c.setParams(c.t0, c.t1)


# ----------------------------------------------------------------
def CheckT(fn):
    def setParams(self, t0, t1, *args):
        return fn(self, max(t0, 0.0001), min(t1, 0.9999), *args)

    return setParams


class Curve:
    def __init__(self, path, t0, t1):
        self._path = path
        self._t0o = t0
        self._t1o = t1
        self._t0 = t0  # t within path
        self._t1 = t1
        self._p0 = None
        self._p1 = None
        self._prev_c = None
        self._next_c = None
        self._size_scale = path.size_scale
        self.clearErrorTallys()

    @property
    def path(self):
        return self._path

    @property
    def type(self):
        return self.__class__.__name__[0]

    @property
    def length(self):
        return 0

    @property
    def p0(self):
        return self._p0

    @property
    def p1(self):
        return self._p1

    @property
    def t0(self):
        return self._t0

    @property
    def t1(self):
        return self._t1

    @property
    def tan0(self):
        return self._path.unit_tangent(self._t0)

    @property
    def tan1(self):
        return self._path.unit_tangent(self._t1)

    @property
    def curvature0(self):
        return self._path.curvature(self._t0)

    @property
    def curvature1(self):
        return self._path.curvature(self._t1)

    @property
    def next_c(self):
        return self._next_c

    @property
    def prev_c(self):
        return self._prev_c

    def setAdjacentCurves(self, prev_c, next_c):
        self._prev_c = prev_c
        self._next_c = next_c

    @classmethod
    def merged(cls, c1, c2):
        return cls(c1.path, c1.t0, c2.t1)

    def getExtraParameters(self):
        return None

    def clearErrorTallys(self):
        self._errors = []

    def getErrorTally(self):
        return {}

    def computeErrors(self, samples=100):
        errors = np.zeros(100, dtype=float)

        for i, t in enumerate(range_t(samples, 0, 1, False)):
            p = self.point(t)
            dist, t, seg = self._path.radialrange(p)[0]  # closest point in path
            errors[i] = dist

        mse = (errors ** 2).mean()
        max_e = np.max(errors)
        sdev = np.std(errors)

        return mse, max_e, sdev

    def serialize(self):
        return dict(type=self.__class__.__name__,
                    data=dict(
                        t0o=self._t0o,
                        t1o=self._t1o,
                        t0=self._t0,
                        t1=self._t1,
                        p0x=self._p0.real,
                        p0y=self._p0.imag,
                        p1x=self._p1.real,
                        p1y=self._p1.imag,
                    )
                    )

    @classmethod
    def deserialize(cls, path, data):
        if data['type'] != cls.__name__: return None
        d = data['data']
        obj = cls(path, d['t0o'], d['t1o'])
        obj._t0 = d['t0']
        obj._t1 = d['t1']
        return obj

    def print(self):
        print(self.__class__.__name__, id(self))
        print(f"   L: {self.length}")
        print(f"   T range:  {self.t0}, {self.t1}")
        print(f"   C range:  {self.curvature0}, {self.curvature1}")
        print(f"   Tangents: {self.tan0}, {self.tan1}")
        print(f"   Extrema:  {self.p0}, {self.p1}")
        adj1 = "--" if self.prev_c is None else id(self.prev_c)
        adj2 = "--" if self.next_c is None else id(self.next_c)
        print(f"   Adjacent: {adj1}, {adj2}")


class Spiral(Curve):
    def __init__(self, path, t0, t1):
        super().__init__(path, t0, t1)

    @property
    def length(self):
        return self._c.length

    @property
    def tan0(self):
        return self._tan0

    @property
    def tan1(self):
        return self._tan1

    @property
    def curvature0(self):
        return self._c.KappaStart

    @property
    def curvature1(self):
        return self._c.KappaEnd

    @property
    def parameterA(self):
        dk = self._c.dk
        a = 1 / math.sqrt(abs(dk))
        return -a if dk < 0 else a

    #	c_change = 1/self.curvature1 - 1/self.curvature0
    #	a = math.sqrt(abs(self.length * c_change))
    #	return a if c_change >= 0 else -a

    @property
    def legend(self):
        return f"Spiral L={self.length:.2f}m A={self.parameterA:.2f}"

    def point(self, t):  # t in range 0..1
        s = t * self.length
        return complex(self._c.X(s), self._c.Y(s))

    def clearErrorTallys(self):
        self._errors = [0, 0]

    def getErrorTally(self):
        return {'distance from path': self._errors[0], 'curvature discontinuity': self._errors[1]}

    def getError(self, err_params):
        # compute max distance from circle to path in the segment t0..t1
        sample_points = 10
        dist_err = 0
        clen = self.length
        for s in range_t(sample_points, 0, clen, False):
            p = complex(self._c.X(s), self._c.Y(s))
            dist, t, seg = self._path.radialrange(p)[0]  # closest point in path
            dist_err += dist * dist

        dist_err /= sample_points

        # compute the deviation from the curvature at the extrema
        cerr0 = (self.curvature0 - self.prev_c.curvature1) if self.prev_c else 0
        cerr1 = (self.curvature1 - self.next_c.curvature0) if self.next_c else 0

        e0 = dist_err * err_params['spiral_distance_from_path_penalty']
        e1 = (cerr0 * cerr0 + cerr1 * cerr1) * err_params['curvature_discontinuity_penalty'] * 10000

        self._errors[0] += e0
        self._errors[1] += e1

        return e0 + e1

    @CheckT
    def setParams(self, t0, t1):
        self._t0 = t0
        self._t1 = t1

        if not self.prev_c or isinstance(self.prev_c, Spiral):
            p0 = self._path.point(t0)
            tan0 = self._path.unit_tangent(t0)
        else:
            p0 = self.prev_c.p1
            tan0 = self.prev_c.tan1

        if not self.next_c or isinstance(self.next_c, Spiral):
            p1 = self._path.point(t1)
            tan1 = self._path.unit_tangent(t1)
        else:
            p1 = self.next_c.p0
            tan1 = self.next_c.tan0

        self._p0 = p0
        self._p1 = p1

        self._tan0 = tan0
        self._tan1 = tan1

        a0 = math.atan2(tan0.imag, tan0.real)
        a1 = math.atan2(tan1.imag, tan1.real)

        self._c = Clothoid.G1Hermite(p0.real, p0.imag, a0, p1.real, p1.imag, a1)

    def plot(self, *args, axis=None, plot_curvature=False, **kargs):
        c = self._c

        if plot_curvature:
            points = [
                complex(self._t0, self.curvature0),
                complex(self._t1, self.curvature1),
            ]
            (axis or plt).plot(np.real(points), np.imag(points), *args, **kargs)
        else:
            tot_points = 20
            clen = self.length
            points = np.zeros((tot_points, 2))
            for i, s in enumerate(range_t(tot_points, 0, clen)):
                points[i][0] = c.X(s)
                points[i][1] = c.Y(s)

            (axis or plt).plot(points[:, 0], points[:, 1], *args, **kargs)

    def serialize(self):
        ser = super().serialize()
        d = ser['data']
        d.update(
            tan0x=self._tan0.real,
            tan0y=self._tan0.imag,
            tan1x=self._tan1.real,
            tan1y=self._tan1.imag,
        )
        return ser

    @classmethod
    def deserialize(cls, path, data):
        obj = super().deserialize(path, data)
        if not obj: return None
        d = data['data']
        obj._tan0 = complex(d['tan0x'], d['tan0y'])
        obj._tan1 = complex(d['tan1x'], d['tan1y'])
        obj._p0 = complex(d['p0x'], d['p0y'])
        obj._p1 = complex(d['p1x'], d['p1y'])
        return obj


class Circle(Curve):
    def __init__(self, path, t0, t1):
        super().__init__(path, t0, t1)
        self._center = None
        self._pm = None
        self._tan_dir = None
        self._ang0 = None
        self._ang1 = None
        self._dang = None
        self._radius = None
        self._d = None

    def _getCircle(self, t0, t1, d):
        p0 = self._path.point(t0)
        p1 = self._path.point(t1)

        pm = self._path.point((t0 + t1) * 0.5)

        c, r = centerOfCircle(p0, p1, pm)

        pm = (pm - c) * d + c

        c, r = centerOfCircle(p0, p1, pm)

        return p0, p1, pm, c, r

    @property
    def length(self):
        a1 = math.atan2(self._p0.imag - self._center.imag, self._p0.real - self._center.real)
        a2 = math.atan2(self._p1.imag - self._center.imag, self._p1.real - self._center.real)

        da = a1 - a2
        while da < 0: da += 2 * math.pi
        while da > 2 * math.pi: da -= 2 * math.pi

        s1 = crossProduct(self._center - self._p0, self._p1 - self._p0)
        s2 = crossProduct(self._pm - self._p0, self._p1 - self._p0)

        if (s1 > 0) == (s2 > 0):  # center and mid point on same side
            angle = da if da >= math.pi else 2 * math.pi - da  # take long angle
        else:
            angle = da if da <= math.pi else 2 * math.pi - da  # take short angle

        return angle * self._radius

    @property
    def radius(self):
        return self._radius

    @property
    def center(self):
        return self._center

    # this is so the tangents we report point in the path's direction
    def _computeTanDir(self):
        c = normalized(self._p0 - self._center)
        tan = complex(-c.imag, c.real)
        if dotProduct(tan, self._path.unit_tangent(self._t0)) < 0:
            self._tan_dir = -1
        else:
            self._tan_dir = 1

    @property
    def tan0(self):
        if self._tan_dir is None: self._computeTanDir()
        c = normalized(self._p0 - self._center)
        if self._tan_dir == -1:
            return complex(c.imag, -c.real)
        else:
            return complex(-c.imag, c.real)

    @property
    def tan1(self):
        if self._tan_dir is None: self._computeTanDir()
        c = normalized(self._p1 - self._center)
        if self._tan_dir == -1:
            return complex(c.imag, -c.real)
        else:
            return complex(-c.imag, c.real)

    @property
    def curvature0(self):
        if self._tan_dir is None: self._computeTanDir()
        if self._tan_dir == -1:
            return -1 / self._radius
        return 1 / self._radius

    @property
    def curvature1(self):
        if self._tan_dir is None: self._computeTanDir()
        if self._tan_dir == -1:
            return -1 / self._radius
        return 1 / self._radius

    @property
    def legend(self):
        return f"Circle L={self.length:.2f}m R={self.radius:.2f}m"

    def point(self, t):  # t in range 0..1
        if self._ang0 is None:
            v = self._p0 - self._center
            a = math.atan2(v.imag, v.real)
            self._ang0 = a

        if self._ang1 is None:
            v = self._p1 - self._center
            a = math.atan2(v.imag, v.real)
            self._ang1 = a

        if self._dang is None:
            self._dang = subtractAngles(self._ang1, self._ang0)

        a = self._dang * t + self._ang0

        c = complex(math.cos(a), math.sin(a)) * self._radius

        return c + self._center

    # def getExtraParameters(self):
    # return [1] # parameter "d"

    def clearErrorTallys(self):
        self._errors = [0, 0]

    def getErrorTally(self):
        return {'distance from path': self._errors[0], 'tangent mismatch': self._errors[1]}

    def getError(self, err_params):
        t0, t1 = self._t0, self._t1
        p0, p1, pm, c, r = self._p0, self._p1, self._pm, self._center, self._radius

        # compute max distance from circle to path in the segment t0..t1
        sample_points = 10
        dist_err = 0
        for t in range_t(sample_points, t0, t1, False):
            p = self._path.point(t)
            # comnpute the distance from p to the circle at center c
            d = abs(distanceBetweenPoints(p, c) - r)
            dist_err += d * d

        dist_err /= sample_points

        # compute the tangent deviation at p0 and p1
        tan0 = self._path.unit_tangent(t0)
        tan1 = self._path.unit_tangent(t1)

        v0 = p0 - c  # vector from center to t0
        v1 = p1 - c

        v0 /= abs(v0)  # normalize
        v1 /= abs(v1)

        d0 = abs(dotProduct(v0, tan0))
        d1 = abs(dotProduct(v1, tan1))

        tan_err = max(d0, d1)

        # length = t1-t0

        e0 = dist_err * err_params['distance_from_path_penalty']
        e1 = tan_err * err_params['tangent_mismatch_penalty']

        self._errors[0] += e0
        self._errors[1] += e1

        return e0 + e1

    @CheckT
    def setParams(self, t0, t1, d=1):
        try:
            p0, p1, pm, c, r = self._getCircle(t0, t1, d)
        except:
            print(t0, t1)
            raise
        self._center = c
        self._radius = r
        self._t0 = t0
        self._t1 = t1
        self._p0 = p0
        self._p1 = p1
        self._d = d
        self._ang0 = None
        self._ang1 = None
        self._dang = None
        self._pm = pm  # a third point to define the circle

    def plot(self, *args, axis=None, plot_curvature=False, **kargs):
        if plot_curvature:
            points = [
                complex(self._t0, self.curvature0),
                complex(self._t1, self.curvature1),
            ]
            (axis or plt).plot(np.real(points), np.imag(points), *args, **kargs)
        else:
            tot_points = 20
            points = []
            for t in range_t(tot_points, self._t0, self._t1):
                p = self._path.point(t)
                # make p a point that lies on the circle with center c
                p = p - self._center
                p = (p / abs(p)) * self._radius
                p = p + self._center
                points.append(p)

            (axis or plt).plot(np.real(points), np.imag(points), *args, **kargs)

            p0 = self._path.point(self._t0)
            p1 = self._path.point(self._t1)

            points = [p0, self._center, p1]

            (axis or plt).plot(np.real(points), np.imag(points), "g--")

    def serialize(self):
        ser = super().serialize()
        d = ser['data']
        d.update(
            d=self._d,
        )
        return ser

    @classmethod
    def deserialize(cls, path, data):
        obj = super().deserialize(path, data)
        if not obj: return None
        return obj


class Line(Curve):

    @property
    def length(self):
        return abs(self._p1 - self._p0)

    @property
    def tan0(self):
        return normalized(self._p1 - self._p0)

    @property
    def tan1(self):
        return normalized(self._p1 - self._p0)

    @property
    def curvature0(self):
        return 0.0

    @property
    def curvature1(self):
        return 0.0

    @property
    def legend(self):
        return f"Line L={self.length:.2f}m"

    def point(self, t):  # t in range 0..1
        return (self._p1 - self._p0) * t + self._p0

    def clearErrorTallys(self):
        self._errors = [0, 0]

    def getErrorTally(self):
        return {'distance from path': self._errors[0], 'tangent mismatch': self._errors[1]}

    def getError(self, err_params):
        t0, t1 = self._t0, self._t1
        p0, p1 = self._p0, self._p1
        path = self._path

        length = abs(p1 - p0)

        sample_points = 10
        dist_err = 0
        for t in range_t(sample_points, t0, t1, False):
            p = path.point(t)
            dist = abs(crossProduct(p - p0, p1 - p0) / length)
            dist_err += dist * dist

        dist_err /= sample_points

        # compute the tangent deviation at p0 and p1
        tan0 = path.unit_tangent(t0)
        tan1 = path.unit_tangent(t1)

        v = (p1 - p0) / length  # normalized vector parallel to the line

        d0 = abs(crossProduct(v, tan0))
        d1 = abs(crossProduct(v, tan1))

        tan_err = max(d0, d1)

        # length = t1-t0

        e0 = dist_err * err_params['distance_from_path_penalty']
        e1 = tan_err * err_params['tangent_mismatch_penalty']

        self._errors[0] += e0
        self._errors[1] += e1

        return e0 + e1

    @CheckT
    def setParams(self, t0, t1):
        p0 = self._path.point(t0)
        p1 = self._path.point(t1)

        self._t0 = t0
        self._t1 = t1
        self._p0 = p0
        self._p1 = p1

    def plot(self, *args, axis=None, plot_curvature=False, **kargs):
        if plot_curvature:
            points = [
                complex(self._t0, self.curvature0),
                complex(self._t1, self.curvature1),
            ]
            (axis or plt).plot(np.real(points), np.imag(points), *args, **kargs)
        else:
            p0 = self._path.point(self._t0)
            p1 = self._path.point(self._t1)

            points = [p0, p1]

            (axis or plt).plot(np.real(points), np.imag(points), *args, **kargs)

    def serialize(self):
        ser = super().serialize()
        return ser

    @classmethod
    def deserialize(cls, path, data):
        obj = super().deserialize(path, data)
        if not obj: return None
        return obj
