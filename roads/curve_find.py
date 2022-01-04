from .algorithms import getLargestSpans, SpanTree, getDisjointSets
from .curves import Circle, Line, Spiral, prepareCurves
from .geometry import *
from .utils import range_t, iter_lag


class Samples:
    def __init__(self, path, distance_between_samples):
        self._distance_between_samples = distance_between_samples
        plen = path.length()
        tot_samples = int(plen / distance_between_samples)
        self._tot_samples = tot_samples
        max_t = (tot_samples * distance_between_samples) / plen
        self._max_t = max_t

        print(f"Path length: {plen}")
        print(f"# of samples: {tot_samples}")
        print(f"Placed every {distance_between_samples}m")

        if tot_samples * distance_between_samples < plen:
            self._position = np.zeros(tot_samples + 1, dtype=complex)
        else:
            self._position = np.zeros(tot_samples, dtype=complex)

        for i, t in enumerate(range_t(tot_samples, 0, max_t)):
            self._position[i] = path.point(t)

        # the last point
        if tot_samples * distance_between_samples < plen:
            self._position[tot_samples] = path.end

    def iter_n(self, n=1):
        return range(n, self._tot_samples - n)

    def clampN(self, n, pad):
        return max(pad, min(self._tot_samples - pad - 1, n))

    def __len__(self):
        return self._tot_samples

    def __getitem__(self, n):
        return self._position[n]

    def NtoT(self, n):
        return (n / (self._tot_samples - 1)) * self._max_t

    def numCurvature(self, n):
        n = self.clampN(n, 1)
        return threePointCurvature(
            self._position[n - 1],
            self._position[n],
            self._position[n + 1]
        )

    def numTangent(self, n):
        n = self.clampN(n, 1)
        return normalized((self._position[n + 1] - self._position[n - 1]) / (2 * self._distance_between_samples))


def partitionCurve(samples, a, b, n, k=0.0001):
    # compute the integral of the curvature in a,b, plus a term k(b-a)
    ctot = 0
    curv = []
    for i in range(a, b):
        c = abs(samples.numCurvature(i)) + k
        curv.append(c)
        ctot += c

    ctot *= 1.001  # # this is so the last point is not added in the loop

    cseg = ctot / n

    split_points = [0]
    sumc = 0
    for i, c in enumerate(curv):
        if sumc >= cseg:
            sumc -= cseg
            split_points.append(i)
        sumc += c

    split_points.append(len(curv))

    return split_points


def identifyCurve(samples, a, b, spiral_thd, line_thd):
    px = list(range(a, b))
    py = [samples.numCurvature(i) * 100 for i in px]
    # we fit a line trhough the curvature points
    _err, m, b = fitLine(px, py)

    if (abs(m * px[0] + b) < line_thd) and (abs(m * px[1] + b) < line_thd):
        return 'L'
    if abs(m) > spiral_thd:  # Curvature line too slant? -> spiral
        return 'S'

    return 'C'


def pickRanges(samples, single_curve_deviate_thd, partition_k):
    # checks whether the samples in range [a:b] can conform a single curve
    def isSingleCurvePred(a, b):
        if b - a <= 2: return True
        px = list(range(a, b))
        py = [samples.numCurvature(i) * 100 for i in px]
        # we fit a line trhough the curvature points
        return fitLine(px, py)[0] < single_curve_deviate_thd

    spans = getLargestSpans(len(samples), isSingleCurvePred)

    st = SpanTree(spans)

    overlapped, isolated = st.getOverlaps()

    disjoint = getDisjointSets(overlapped, isolated)

    for i, d in enumerate(disjoint):
        print(f"{i} - {d}")

    for dset in disjoint:
        # dset is a list of overlapping segments which individually form a valid curves
        # a,b is the range of all those overlapping segments
        a, b = min(x[0] for x in dset), max(x[1] for x in dset)
        print("Disjoint interval:", (a, b))

        # partition the path in the range a,b into individual partitions which are single curves
        # we start by splitting in one, check, then in two, check, until all fragments are valid curves
        n = 1
        while True:
            part = partitionCurve(samples, a, b, n, partition_k)
            print(f"{n} - Partitions: {part}")
            if all(isSingleCurvePred(a, b) for a, b in iter_lag(part)):
                break
            n += 1

        print("Curves in interval:", n)

        for ab in iter_lag(part):
            yield ab


def simplifyCurves(curves, s_thd, c_thd, l_thd):
    print("Simplifying curves")

    # see if an arc became a line
    for i in range(len(curves)):
        c = curves[i]
        if c.type == 'C' and c.radius > 2000:
            curves[i] = Line(c.path, c.t0, c.t1)
            print(f"{i} C->L")

    prepareCurves(curves)

    while True:
        pairs_s = []
        pairs_c = []
        pairs_l = []

        for i, (c1, c2) in enumerate(iter_lag(curves)):
            print(c1.type, c2.type)
            if c1.type == 'S' == c2.type:
                e = abs(c1.parameterA - c2.parameterA)
                print(i, c1.type, e)
                if e < s_thd:
                    pairs_s.append((e, c1, c2))
            if c1.type == 'C' == c2.type:
                e = abs(c1.curvature0 - c2.curvature0)
                print(i, c1.type, e)
                if e < c_thd:
                    pairs_c.append((e, c1, c2))
            if c1.type == 'L' == c2.type:
                e = threePointCurvature(c1.p0, c1.p1, c2.p1 + (c1.p1 - c2.p0))
                print(i, c1.type, e)
                if e < l_thd:
                    pairs_l.append((e, c1, c2))

        if not any([pairs_s, pairs_c, pairs_l]): return

        for pairs in (pairs_s, pairs_c, pairs_l):
            if not pairs: continue
            pairs.sort()
            e, c1, c2 = pairs[0]

            print(f"Removing one {c1.__class__.__name__}")

            # not efficient, but given the number of curves there are, it's fast
            for i, c in enumerate(curves):
                if c == c1:
                    new_c = c.__class__.merged(c1, c2)
                    curves[i:i + 2] = [new_c]
                    break

        prepareCurves(curves)


def fromPattern(path, pattern, parameters):
    distance_between_samples = parameters['distance_between_samples']
    samples = Samples(path, distance_between_samples)
    part = partitionCurve(samples, 0, len(samples), len(pattern), k=0.0001)

    curves = []
    for cid, (a, b) in zip(pattern, iter_lag(part)):
        if cid == 'C':
            c = Circle(path, samples.NtoT(a), samples.NtoT(b))
        elif cid == 'L':
            c = Line(path, samples.NtoT(a), samples.NtoT(b))
        else:
            c = Spiral(path, samples.NtoT(a), samples.NtoT(b))

        curves.append(c)

    prepareCurves(curves)

    return curves


def buildCurves(path, parameters):
    distance_between_samples = parameters['distance_between_samples']
    single_curve_deviate_thd = parameters['single_curve_deviate_thd']
    partition_k = parameters['partition_k']
    spiral_thd = parameters['spiral_thd']
    line_thd = parameters['line_thd']

    samples = Samples(path, distance_between_samples)

    curves = []

    for a, b in pickRanges(samples, single_curve_deviate_thd, partition_k):

        cid = identifyCurve(samples, a, b, spiral_thd, line_thd)
        if cid == 'C':
            c = Circle(path, samples.NtoT(a), samples.NtoT(b))
        elif cid == 'L':
            c = Line(path, samples.NtoT(a), samples.NtoT(b))
        else:
            c = Spiral(path, samples.NtoT(a), samples.NtoT(b))

        curves.append(c)

    prepareCurves(curves)

    return curves
