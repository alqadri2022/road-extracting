import random
import math
import numpy as np
from .utils import range_t


class EvSolver:
    def __init__(self, error_fn, initial_t_vector, *args, generation_size=100):
        self._error_fn = error_fn
        self._fn_args = args

        if isinstance(initial_t_vector, int):
            initial_t_vector = list(range_t(initial_t_vector, 0, 1))

        self._tot_t = len(initial_t_vector)
        self._initial_t_vec = initial_t_vector
        self._initial_dif_t = self.TtoDT(initial_t_vector)

        self._current_generation = None  # list of (dt_vec, item_id)
        self._generation_fitness = None  # list of (item_fitness, item_number)

        self._items_fitness_cache = {}  # map item_id -> item_fitness

        self._mut_prob = 0.05
        self._generation_size = generation_size

        self._best_solution = None

        self._id_num = 0  # a counter that increments only

        self._mutate_sigma = 1 / (2 * self._tot_t)

    def newRandomDT(self):
        # create a completely random t vector
        arr = np.zeros(self._tot_t - 1, dtype=float)
        for i in range(self._tot_t - 1):
            arr[i] = random.random()

        arr /= np.sum(arr)

        return arr

    @classmethod
    def DTtoT(cls, dt_arr):
        arr = np.zeros(len(dt_arr) + 1)
        arr[1:] = np.cumsum(dt_arr)
        return arr

    @classmethod
    def TtoDT(cls, t_arr):
        return np.ediff1d(t_arr)

    def applyDTLimits(self, dt_arr, limit=0.5):
        limit = (1 / (self._tot_t - 1)) * limit
        dt_arr[:] = np.maximum(dt_arr, limit)
        dt_arr /= np.sum(dt_arr)

    @classmethod
    def normalizeDT(cls, dt_arr):
        dt_arr /= np.sum(dt_arr)

    @property
    def new_id(self):
        self._id_num += 1
        return self._id_num

    # merge arrays at random
    def combineDT_1(self, dt_vec0, dt_vec1):
        size = len(dt_vec0)
        assert (size == len(dt_vec1))
        arr = np.zeros(size)
        for i in range(size):
            if random.random() >= 0.5:
                arr[i] = dt_vec0[i]
            else:
                arr[i] = dt_vec1[i]

        arr /= np.sum(arr)
        return arr

    # average arrays
    def combineDT_2(self, dt_vec0, dt_vec1):
        size = len(dt_vec0)
        assert (size == len(dt_vec1))
        arr = np.zeros(size)
        for i in range(size):
            arr[i] = dt_vec0[i] + dt_vec1[i]

        arr /= np.sum(arr)
        return arr

    def mutateDT(self, dt_vec, force_mutate=False):
        size = len(dt_vec)
        mutated = False

        while random.random() < self._mut_prob or force_mutate:
            if random.random() > 0.3:
                n = random.randrange(size)
                m = random.randrange(size)
                dt_vec[n], dt_vec[m] = dt_vec[m], dt_vec[n]
            else:
                n = random.randrange(size)
                nv = max(dt_vec[n] + random.gauss(0, self._mutate_sigma), 0.01)
                dt_vec[n] = nv
                mutated = True

            force_mutate = False

        dt_vec /= np.sum(dt_vec)

        return mutated

    def evaluateGenerationFitness(self):
        gfit = []

        for i, (dt_vec, item_id) in enumerate(self._current_generation):
            if item_id in self._items_fitness_cache:
                fitness = self._items_fitness_cache[item_id]
            else:
                t = self.DTtoT(dt_vec)
                try:
                    fitness = -abs(self._error_fn(t, *self._fn_args))
                except KeyboardInterrupt:
                    raise
                except:
                    print("Error calling fitness curve for t:", t)
                    fitness = -math.inf
                self._items_fitness_cache[item_id] = fitness

            gfit.append((fitness, i))

        gfit.sort(reverse=True)  # most fit first

        self._generation_fitness = gfit

        self.updateBest()

    # requires evaluateGenerationFitness to have been called on current generation
    def updateBest(self):
        fitness, idx = self._generation_fitness[0]

        if self._best_solution is None or self._best_solution[0] < fitness:
            self._best_solution = (fitness, self._current_generation[idx][0].copy())

    def trimErrorCache(self):
        cache_to_delete = set(self._items_fitness_cache.keys())
        for _dt_vec, item_id in self._current_generation:
            cache_to_delete.discard(item_id)

        for item_id in cache_to_delete:
            del self._items_fitness_cache[item_id]

    def makeInitialGeneration(self):
        cg = []
        n = self._generation_size

        # half is variations on the original dt
        for _i in range(n // 2):
            dt_vec = np.array(self._initial_dif_t)
            self.mutateDT(dt_vec, True)
            cg.append((dt_vec, self.new_id))

        # other half is completely random vectors
        for _i in range(n - n // 2):
            cg.append((self.newRandomDT(), self.new_id))

        self._current_generation = cg

        self.evaluateGenerationFitness()

        fitness, _idx = self._generation_fitness[0]
        return fitness

    # requires evaluateGenerationFitness to have been called on current generation
    def pickBest(self):
        _fitness, idx = self._generation_fitness[0]
        return self._current_generation[idx]

    # requires evaluateGenerationFitness to have been called on current generation
    def pickOneOfTheBest(self, alpha=1, beta=3):
        while True:
            while True:
                v = random.paretovariate(alpha) - 1
                if v < beta: break

            n = (v * self._generation_size) / beta
            # the following if should be always true, but check anyway in
            # case there were rounding errors
            if n <= self._generation_size:
                break
        n = int(n)

        _fitness, idx = self._generation_fitness[n]
        return self._current_generation[idx]

    def pickRandom(self):
        return self._current_generation[random.randrange(self._generation_size)]

    def checkGeneration(self):
        for dt_vec, ident in self._current_generation:
            if sum(dt_vec) > 1.000001 or any(dt < 0 for dt in dt_vec):
                print("Error: ", ident, dt_vec)
                raise Exception()

    def advanceGeneration(self, constraints=None):
        n = self._generation_size
        new_generation = []

        # first, add the best one from previous generation
        new_generation.append(self.pickBest())

        # then add 1/4 of the elements from the best, with possible mutations
        for _i in range(1, n // 4):
            dt_vec, vec_id = self.pickOneOfTheBest()  # [0] ->  the dt_vector

            mutated = self.mutateDT(dt_vec.copy())

            if constraints:
                constraints(dt_vec)
            if mutated:
                new_generation.append((dt_vec, self.new_id))
            else:
                new_generation.append((dt_vec, vec_id))

        # then add 1/3 of the elements as a combination of two other vectors
        for _i in range(n // 3):
            dt_vec1, _ = self.pickOneOfTheBest()
            dt_vec2, _ = self.pickOneOfTheBest()

            dt_vec = self.combineDT_1(dt_vec1, dt_vec2)
            self.mutateDT(dt_vec)
            if constraints:
                constraints(dt_vec)
            new_generation.append((dt_vec, self.new_id))

        # add the rest with random vectors
        for _i in range(n - len(new_generation)):
            dt_vec = self.newRandomDT()
            if constraints: constraints(dt_vec)
            new_generation.append((dt_vec, self.new_id))

        self._current_generation = new_generation
        self._generation_fitness = None

        # self.checkGeneration()

        self.evaluateGenerationFitness()
        self.trimErrorCache()

        fitness, _idx = self._generation_fitness[0]
        return fitness

    @property
    def best(self):
        fitness, dt_vec = self._best_solution
        return fitness, self.DTtoT(dt_vec)

    @property
    def fitness_avg(self):
        return np.average(np.array(self._generation_fitness)[:10, 0])


if __name__ == "__main__":
    # these is the solution t vector: t -> [0.112603,0.276337,0.723663,0.887397]
    def errfn(t_vec):
        err = np.array([math.cos((t - 0.5) ** 2 * 31.4) for t in t_vec])
        return np.sum(err * err)


    es = EvSolver(errfn, 6)
    fitness = es.makeInitialGeneration()
    print(fitness)

    constraints = lambda dt_arr: EvSolver.applyDTLimits(dt_arr)

    for i in range(1000):
        es.advanceGeneration(constraints)
        print(es.fitness_avg)

    print(es.best)
