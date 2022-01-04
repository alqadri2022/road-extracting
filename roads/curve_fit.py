# from scipy.optimize import minimize
if 0:
    import mystic.symbolic as ms
    import mystic.solvers as my
import random

import numpy as np
from scipy.optimize import minimize

from .curves import Spiral, prepareCurves
from .ev_solver import EvSolver
from .geometry import distanceBetweenPoints
from .utils import iter_lag, range_t


def fitCurves2(curves, cfg_params):
    print("Adjusting curves")
    random.seed(1234)
    deltas = []
    for c in curves:
        deltas.append(c.length)

    deltas = np.array(deltas, dtype=float)

    EvSolver.normalizeDT(deltas)
    initial_t = EvSolver.DTtoT(deltas)

    def error_fn(params):
        try:
            for c, (t0, t1) in zip(curves, iter_lag(params)):
                if isinstance(c, Spiral): continue
                c.setParams(max(t0, 0.001), min(t1, 0.999))

            for c, (t0, t1) in zip(curves, iter_lag(params)):
                if not isinstance(c, Spiral): continue
                c.setParams(max(t0, 0.001), min(t1, 0.999))
        except:
            print(params)
            raise

        err = 0
        for c in curves:
            err += c.getError(cfg_params)

        return err

    ev = EvSolver(error_fn, initial_t, generation_size=cfg_params['generation_size'])
    ev.makeInitialGeneration()

    constraints = ev.applyDTLimits

    for i in range(cfg_params['iterations']):
        ev.advanceGeneration(constraints)
        print(i, ev.fitness_avg)

    params = ev.best[1]

    for c, (t0, t1) in zip(curves, iter_lag(params)):
        if isinstance(c, Spiral):
            continue
        c.setParams(max(t0, 0.001), min(t1, 0.999))

    for c, (t0, t1) in zip(curves, iter_lag(params)):
        if not isinstance(c, Spiral):
            continue
        c.setParams(max(t0, 0.001), min(t1, 0.999))


def growArcs(curves):
    print("Growing arcs")
    max_i = len(curves) - 1
    curve_info = []

    pad = 0.05

    for i in range(len(curves)):
        c = curves[i]
        mid = (c.t0 + c.t1) * 0.5

        min_t0, max_t1 = c.t0, c.t1
        can_expand_back = i > 0 and curves[i - 1].type != 'C'
        if can_expand_back:
            min_t0 = min((curves[i - 1].t0 + curves[i - 1].t1) * 0.5 + pad, c.t0)

        can_expand_fwd = i < max_i and curves[i + 1].type != 'C'
        if can_expand_fwd:
            max_t1 = max((curves[i + 1].t0 + curves[i + 1].t1) * 0.5 - pad, c.t1)

        if can_expand_back or can_expand_fwd:
            curve_info.append(
                ((can_expand_back, can_expand_fwd), (min_t0, max(c.t0, mid - pad)), (min(c.t1, mid + pad), max_t1), c))

    # do gradient descent on each arc

    def func(c, t0, t1):
        path = c.path

        sample_points = 10
        dist_err = 0
        for t in range_t(sample_points, t0, t1, False):
            p = path.point(t)
            # comnpute the distance from p to the circle at center c
            d = abs(distanceBetweenPoints(p, c.center) - c.radius)
            dist_err += d * d

        dist_err /= sample_points

        print("Dist err=", dist_err)

        return dist_err - (t1 - t0) * 0.00001

    for dir, t0_range, t1_range, c in curve_info:
        t0, t1 = c.t0, c.t1

        mf = lambda t_arr: func(c, t_arr[0], t_arr[1])

        if dir == (True, True):
            res = minimize(mf, (t0, t1), bounds=[t0_range, t1_range])
            t0, t1 = res.x[0], res.x[1]
        if dir == (True, False):
            res = minimize(mf, (t0,), bounds=[t0_range])
            t0, t1 = res.x[0], c.t1
        if dir == (False, True):
            res = minimize(mf, (t1,), bounds=[t1_range])
            t0, t1 = c.t0, res.x[1]

        c.setParams(t0, t1)
        if c.prev_c:
            p = c.prev_c
            p.setParams(p.t0, t0)
        if c.next_c:
            p = c.next_c
            p.setParams(t1, p.t1)

    prepareCurves(curves)


if 0:  # not used
    def fitCurves(curves, err_params, maxiter=None):
        params = []  # err_params that will be fitted
        param_range = []  # ranges of extra err_params within params for each curve
        bounds = []
        t_loc = []  # location in bounds and params where t err_params are placed
        max_i = len(curves) - 1
        for i, c in enumerate(curves):
            p0 = len(params)  # include the t0
            ep = c.getExtraParameters()
            if ep:
                params.extend(ep)
                for _e in ep:
                    bounds.append((None, None))  # extra err_params are not bounded
            param_range.append((p0, len(params)))
            if i < max_i:
                params.append(c.t1)
                t_loc.append(len(bounds))
                bounds.append((0, 1))  # "t" parameter bounds

            c.clearErrorTallys()

        def curveIter(params):
            t0 = 0
            for i, c in enumerate(curves):
                if i < max_i:
                    t1 = params[param_range[i][1]]  # next parameter after the last extra one (t1)
                else:
                    t1 = 1.0
                yield i, c, t0, t1
                t0 = t1  # all curves are connected

        def fitness(params):
            # check all t are monotonically increasing. If not, mark error

            for i, c, t0, t1 in curveIter(params):  # not Clothoids
                if isinstance(c, Spiral): continue
                if t0 < 0.00001: t0 = 0.00001
                if t1 > 0.9999:  t1 = 0.9999
                if t1 <= t0 + 0.01:
                    return 100000000.0

                c.setParams(t0, t1, *params[param_range[i][0]:param_range[i][1]])

            for i, c, t0, t1 in curveIter(params):  # Clothoids
                if not isinstance(c, Spiral): continue
                if t0 < 0.00001: t0 = 0.00001
                if t1 > 0.9999:  t1 = 0.9999
                if t1 <= t0 + 0.01:
                    return 100000000.0

                c.setParams(t0, t1, *params[param_range[i][0]:param_range[i][1]])

            err = 0
            for c in curves:
                err += c.getError(err_params)

            print("Err=", err)
            return err

        # construct the "t" constraints equation set
        eqs = []
        spacing = 0.01
        for i in range(len(t_loc) - 1):
            eqs.append(f"x{t_loc[i + 1]} - x{t_loc[i]} >= ({spacing})")
        eqs = "\n".join(eqs)

        print("T-loc=", t_loc)
        print("bounds=", bounds)
        print("param_range=", param_range)
        print("params=", params)

        if not params:
            print("Nothing to fit")
            return

        if not eqs:
            print("No related parameter constraints")
            constraints = None
        else:
            print("Parameter constraints:")
            print(eqs)

            eqs = ms.simplify(eqs)
            constraints = ms.generate_constraint(ms.generate_solvers(eqs))

        params = my.diffev2(fitness, x0=params, bounds=bounds, constraints=constraints, maxiter=maxiter, handler=True)
        print(list(params))

        # res = minimize(fitness, params, constraints={'type':'eq','fun':constraints})
        # params = res.x

        for i, c, t0, t1 in curveIter(params):
            if isinstance(c, Spiral): continue
            c.setParams(t0, t1, *params[param_range[i][0]:param_range[i][1]])

        for i, c, t0, t1 in curveIter(params):
            if not isinstance(c, Spiral): continue
            c.setParams(t0, t1, *params[param_range[i][0]:param_range[i][1]])

# ----------------------------------------------------------------
