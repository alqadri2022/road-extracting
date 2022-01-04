import matplotlib.pyplot as plt
from svgpathtools import CubicBezier, Path, disvg
from svgpathtools.polytools import rational_limit, real, imag

from .curve_fit import *
from .geometry import *


def positionSampler(src, t, *args):
    return src.point(t)


def curvatureSampler(src, t, *args):
    if isinstance(src, CubicBezier):
        return segmentCurvature(src, t)
    return complex(t, src.curvature(t))


def curvatureSamplerNum(src, t, *args):
    return complex(t, src.curvatureN(t, *args))


def derivativeSamplerNum(src, t, *args):
    return src.derivativNe(t, *args)


def bearingChangeSamplerNum(src, t, *args):
    ddv = src.derivative(t, 2)
    tan = src.unit_tangent(t)

    return complex(t, crossProduct(tan, ddv))


def getSampler(sampler):
    if not callable(sampler):
        if sampler == "position" or sampler is None:
            sampler = positionSampler
        elif sampler == "curvature":
            sampler = curvatureSampler
        elif sampler == "num_curvature":
            sampler = curvatureSamplerNum
        elif sampler == "bearing-change":
            sampler = bearingChangeSamplerNum
        else:
            raise ValueError("Unknown sampler: %s" % sampler)
    return sampler


# this is similar to the one already implemented in svgpathtools,
# but returns signed curvature
def segmentCurvature(segment, t):
    """
    returns the curvature of the segment at t.
    Notes
    -----
    If you receive a RuntimeWarning, run command
    >>> old = np.seterr(invalid='raise')
    This can be undone with
    >>> np.seterr(**old)
    """

    dz = segment.derivative(t)
    ddz = segment.derivative(t, n=2)
    dx, dy = dz.real, dz.imag
    ddx, ddy = ddz.real, ddz.imag
    old_np_seterr = np.seterr(invalid='raise')

    try:
        kappa = (dx * ddy - dy * ddx) / math.sqrt(dx * dx + dy * dy) ** 3
    except (ZeroDivisionError, FloatingPointError):
        return 0
        # tangent vector is zero at t, use polytools to find limit
        p = segment.poly()
        dp = p.deriv()
        ddp = dp.deriv()

        dx, dy = real(dp), imag(dp)
        ddx, ddy = real(ddp), imag(ddp)
        f2 = (dx * ddy - dy * ddx) ** 2
        g2 = (dx * dx + dy * dy) ** 3

        lim2 = rational_limit(f2, g2, t)
        if lim2 < 0:  # impossible, must be numerical error
            return 0
        kappa = math.sqrt(lim2)
    finally:
        np.seterr(**old_np_seterr)
    return kappa


# given two connected CubicBezier instances, make them have continuous curvature at the joint
def makeContinuousCurvature(bez1, bez2):
    ax, bx, cx, dx = tuple(real(a) for a in bez1)
    ay, by, cy, dy = tuple(imag(a) for a in bez1)

    ex, fx, gx, hx = tuple(real(a) for a in bez2)
    ey, fy, gy, hy = tuple(imag(a) for a in bez2)

    a = (dx - cx)
    b = (dy - cy)
    c = (fx - ex)
    d = (fy - ey)

    r1 = r2 = (math.sqrt(a * a + b * b) + math.sqrt(c * c + d * d)) * 0.5

    a1 = math.atan2(b, a)
    a2 = math.atan2(d, c)
    angle = averageAngles(a1, a2)

    assert (math.fabs(a1 - a2) < 0.1 and math.fabs(a1 - angle) < 0.1)

    # dx,dy should be equal to ex,ey already
    dx = (dx + ex) / 2
    dy = (dy + ey) / 2

    def pointsFromParameters(X):
        r1, r2, a = X

        ca = math.cos(a)
        sa = math.sin(a)

        cxp = dx - ca * r1
        cyp = dy - sa * r1
        fxp = dx + ca * r2
        fyp = dy + sa * r2

        return cxp, cyp, fxp, fyp

    def err(X):  # this measures how much the given curves deviate from the originals
        cxp, cyp, fxp, fyp = pointsFromParameters(X)

        a = cxp - cx
        b = cyp - cy
        c = fxp - fx
        d = fyp - fy

        return a * a + b * b + c * c + d * d

    def eq(X):  # contraint (keep continuous curvature)
        cxp, cyp, fxp, fyp = pointsFromParameters(X)

        cxdx = cxp - dx
        cydy = cyp - dy

        fxgx = fxp - gx
        fygy = fyp - gy

        dxfx = dx - fxp
        dyfy = dy - fyp

        return ((bx * cydy - by * cxdx + cxp * dy - cyp * dx) / ((cxdx * cxdx + cydy * cydy) ** 1.5) +
                (dy * fxgx - dx * fygy + fyp * gx - fxp * gy) / ((dxfx * dxfx + dyfy * dyfy) ** 1.5))

    res = minimize(err, [r1, r2, angle], constraints={'type': 'eq', 'fun': eq}, tol=1e-10)

    cxp, cyp, fxp, fyp = pointsFromParameters(res.x)

    # cxp, cyp, fxp, fyp = pointsFromParameters((r1,r2,angle))
    p1 = CubicBezier(bez1[0], bez1[1], complex(cxp, cyp), complex(dx, dy))
    p2 = CubicBezier(complex(dx, dy), complex(fxp, fyp), bez2[2], bez2[3])

    return p1, p2


class Path2(Path):

    def __init__(self, *args, **kargs):

        if args and isinstance(args[0], Path):  # copy constructor
            super().__init__(*(args[0]))
        else:
            super().__init__(*args, **kargs)

    def copy(self):
        return Path2(*[seg for seg in self])

    def makeContinuousCurvature(self, iterations=3):
        for it in range(iterations):
            for i in range(0, len(self) - 1):
                self[i], self[i + 1] = makeContinuousCurvature(self[i], self[i + 1])

    # adapted from
    #  http://www.particleincell.com/2012/bezier-splines/
    # returns a new path from several points that is
    # C2 continuous everywhere
    @classmethod
    def smoothFromPoints(cls, points):

        n = len(points) - 1  # last point

        p1 = np.zeros(n, dtype=complex)
        p2 = np.zeros(n, dtype=complex)

        a = np.zeros(n, dtype=complex)
        b = np.zeros(n, dtype=complex)
        c = np.zeros(n, dtype=complex)
        r = np.zeros(n, dtype=complex)

        a[0] = 0
        b[0] = 2
        c[0] = 1
        r[0] = points[0] + 2 * points[1]

        for i in range(1, n - 1):
            a[i] = 1
            b[i] = 4
            c[i] = 1
            r[i] = 4 * points[i] + 2 * points[i + 1]

        a[n - 1] = 2
        b[n - 1] = 7
        c[n - 1] = 0
        r[n - 1] = 8 * points[n - 1] + points[n]

        for i in range(1, n):
            m = a[i] / b[i - 1]
            b[i] = b[i] - m * c[i - 1]
            r[i] = r[i] - m * r[i - 1]

        p1[n - 1] = r[n - 1] / b[n - 1]

        for i in range(n - 2, -1, -1):
            p1[i] = (r[i] - c[i] * p1[i + 1]) / b[i]

        for i in range(0, n - 1):
            p2[i] = 2 * points[i + 1] - p1[i + 1]

        p2[n - 1] = 0.5 * (points[n] + p1[n - 1])

        segments = []
        for i in range(n):  # this many curves
            segments.append(
                CubicBezier(
                    points[i],
                    p1[i],
                    p2[i],
                    points[i + 1]
                )
            )

        return cls(*segments)

    def curvature(self, t):
        seg_idx, t = self.T2t(t)
        seg = self[seg_idx]
        c = segmentCurvature(seg, t)
        return c

    def curvatureN(self, t, radius=0.03):
        while True:
            p0 = self.point(t - radius)
            p1 = self.point(t + radius)
            p = self.point(t)

            if p0 is None:
                t += radius
                continue
            if p1 is None:
                t -= radius
                continue

            dz = (p1 - p0) / (2 * radius)
            ddz = (p1 + p0 - 2 * p) / (radius * radius)

            break

        dx, dy = dz.real, dz.imag
        ddx, ddy = ddz.real, ddz.imag

        return (dx * ddy - dy * ddx) / math.sqrt(dx * dx + dy * dy) ** 3

    # returns a list of points after sampling from the
    # path, if uniform, the samples are evenly distributed
    # along the path, otherwise they are taken from each segment
    # fn is the function used to sample the segment/path
    # and is by default point(t)
    def samples(self, n=3, *args, uniform=True, sample_fn="position"):
        sample_fn = getSampler(sample_fn)
        if not uniform:
            if n is None: n = 2  # 2 = edges + center
            tot_samples = n * len(self) + 1
            coords = np.zeros(tot_samples, dtype=complex)
            coords[0] = sample_fn(self[0], 0, *args)
            j = 1
            for seg in self:
                if n == 1:
                    coords[j] = sample_fn(seg, 1, *args)
                    j += 1
                else:
                    for i in range(n):
                        t = (i + 1) / (n - 1)
                        coords[j] = sample_fn(seg, t, *args)
                        j += 1
        else:
            if n is None: n = len(self) * 2
            sample_fn = getSampler(sample_fn)
            coords = np.zeros(n, dtype=complex)
            for i in range(n):
                t = i / (n - 1)
                coords[i] = sample_fn(self, (t * 0.99) + 0.005, *args)

        return coords

    def _headingAtPos(self, t, prev_ang):
        n = self.unit_tangent(t)
        a = math.atan2(n.imag, n.real)
        if prev_ang is None:
            return a
        else:
            d = a - prev_ang
            if d > math.pi:
                d -= 2 * math.pi
            elif d < -math.pi:
                d += 2 * math.pi
            return prev_ang + d

    def toHeading(self, samples=200, shift=0):
        heading = np.zeros(samples, dtype=complex)
        ang = None
        for i in range(samples):
            t = min(1, max(0, i / samples + shift))
            ang = self._headingAtPos(t, ang)
            heading[i] = complex(t, ang)
        return heading

    def toHeadingChange(self, samples=200, h=0.01, shift=0):
        hchange = np.zeros(samples, dtype=complex)
        ang1 = None
        ang2 = None

        ang1 = self.toHeading(samples, -h + shift)
        ang2 = self.toHeading(samples, +h + shift)

        for i, (a1, a2) in enumerate(zip(ang1, ang2)):
            t1 = a1.real
            t2 = a2.real
            t = (t1 + t2) * 0.5
            h = (t2 - t1)
            d = a2.imag - a1.imag
            hchange[i] = complex(t, d / h)
        return hchange

    # sample several points in each segment and return a single point per segment in the path
    # we can specify what (point, curvature, derivative) by setting sample_fn
    # and we can specify how it it sampled (average, median, min etc..) by setting combine_fn
    def sampleSegments(self, samples_per_segment, sample_fn, combine_fn):
        sample_fn = getSampler(sample_fn)

        tot_steps = len(self)

        samples = np.zeros(tot_steps, dtype=complex)
        buf = np.zeros(samples_per_segment, dtype=complex)

        for seg_num, segment in enumerate(self):
            for j in range(samples_per_segment):
                t = j / (samples_per_segment - 1)
                buf[j] = sample_fn(segment, t)

            c = combine_fn(buf)
            samples[seg_num] = c

        return samples

    def sampleSegmentsMean(self, steps=3, sample_fn="position"):
        return self.sampleSegments(steps, sample_fn, np.mean)

    def sampleSegmentsMedian(self, steps=3, sample_fn="position"):
        return self.sampleSegments(steps, sample_fn, np.average)

    @property
    def size_scale(self):
        return 1

    def show(self):
        disvg(self)

    @classmethod
    def showMany(cls, *paths):
        disvg(paths)

    def plot(self, *style, show=True, q=3, sampler_fn="position", mode=None, axis=None):
        if mode is None:
            tot_samples = sum(int(math.ceil(elem.length() / q)) for elem in self)
            coords = self.samples(n=tot_samples, sample_fn=sampler_fn, uniform=True)
        elif mode == "segment-average":
            coords = self.sampleSegmentsMean(q, sampler_fn)
        elif mode == "segment-median":
            coords = self.sampleSegmentsMedian(q, sampler_fn)

        (axis or plt).plot(np.real(coords), np.imag(coords), *style)

        if show:
            plt.show()
