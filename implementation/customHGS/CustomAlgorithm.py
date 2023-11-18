from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Collection, Tuple

from pyvrp.Result import Result
from pyvrp.Statistics import Statistics

if TYPE_CHECKING:
    from pyvrp.PenaltyManager import PenaltyManager
    from pyvrp.Population import Population
    from pyvrp._pyvrp import (
        CostEvaluator,
        ProblemData,
        RandomNumberGenerator,
        Solution,
    )
    from pyvrp.search.SearchMethod import SearchMethod
    from pyvrp.stop.StoppingCriterion import StoppingCriterion

import torch
from Models import SREXmodel
from data.utils.GraphData import ParentGraph, FullGraph
from implementation.customHGS.SolutionTransformer import SolutionTransformer
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from data.utils.Normalize import normalize_graphs
from torch_geometric.data import Batch

PE = AddLaplacianEigenvectorPE(6, attr_name=None, is_undirected=True)


@dataclass
class GeneticAlgorithmParams:
    """
    Parameters for the genetic algorithm.

    Parameters
    ----------
    repair_probability
        Probability (in :math:`[0, 1]`) of repairing an infeasible solution.
        If the reparation makes the solution feasible, it is also added to
        the population in the same iteration.
    nb_iter_no_improvement
        Number of iterations without any improvement needed before a restart
        occurs.

    Attributes
    ----------
    repair_probability
        Probability of repairing an infeasible solution.
    nb_iter_no_improvement
        Number of iterations without improvement before a restart occurs.

    Raises
    ------
    ValueError
        When ``repair_probability`` is not in :math:`[0, 1]`, or
        ``nb_iter_no_improvement`` is negative.
    """

    repair_probability: float = 0.80
    nb_iter_no_improvement: int = 20_000

    def __post_init__(self):
        if not 0 <= self.repair_probability <= 1:
            raise ValueError("repair_probability must be in [0, 1].")

        if self.nb_iter_no_improvement < 0:
            raise ValueError("nb_iter_no_improvement < 0 not understood.")


class GeneticAlgorithm:
    """
    Creates a GeneticAlgorithm instance.

    Parameters
    ----------
    data
        Data object describing the problem to be solved.
    penalty_manager
        Penalty manager to use.
    rng
        Random number generator.
    population
        Population to use.
    search_method
        Search method to use.
    crossover_op
        Crossover operator to use for generating offspring.
    initial_solutions
        Initial solutions to use to initialise the population.
    params
        Genetic algorithm parameters. If not provided, a default will be used.

    Raises
    ------
    ValueError
        When the population is empty.
    """

    def __init__(
            self,
            data: ProblemData,
            model: SREXmodel,
            model_CostEvaluator: CostEvaluator,
            full_graph: FullGraph,
            penalty_manager: PenaltyManager,
            rng: RandomNumberGenerator,
            population: Population,
            search_method: SearchMethod,
            crossover_op: Callable[
                [
                    Tuple[Solution, Solution],
                    ProblemData,
                    CostEvaluator,
                    RandomNumberGenerator,
                ],
                Solution,
            ],
            initial_solutions: Collection[Solution],
            params: GeneticAlgorithmParams = GeneticAlgorithmParams(),
            sol_transform: SolutionTransformer = SolutionTransformer(),

    ):
        if len(initial_solutions) == 0:
            raise ValueError("Expected at least one initial solution.")

        self._data = data
        self._pm = penalty_manager
        self._rng = rng
        self._pop = population
        self._search = search_method
        self._crossover = crossover_op
        self._initial_solutions = initial_solutions
        self._params = params
        self._model = model
        self._full_graph = full_graph
        self._sol_transform = sol_transform
        self._model_CostEvaluator = model_CostEvaluator

        # Find best feasible initial solution if any exist, else set a random
        # infeasible solution (with infinite cost) as the initial best.
        self._best = min(initial_solutions, key=self._cost_evaluator.cost)

    @property
    def _cost_evaluator(self) -> CostEvaluator:
        return self._pm.get_cost_evaluator()

    def run(self, stop: StoppingCriterion):
        """
        Runs the genetic algorithm with the provided stopping criterion.

        Parameters
        ----------
        stop
            Stopping criterion to use. The algorithm runs until the first time
            the stopping criterion returns ``True``.

        Returns
        -------
        Result
            A Result object, containing statistics and the best found solution.
        """
        start = time.perf_counter()
        stats = Statistics()
        iters = 0
        iters_no_improvement = 1

        for sol in self._initial_solutions:
            self._pop.add(sol, self._cost_evaluator)

        while not stop(self._cost_evaluator.cost(self._best)):
            iters += 1

            if iters_no_improvement == self._params.nb_iter_no_improvement:
                iters_no_improvement = 1
                self._pop.clear()

                for sol in self._initial_solutions:
                    self._pop.add(sol, self._cost_evaluator)

            curr_best = self._cost_evaluator.cost(self._best)

            # TODO: add implementation Loop
            parents = self._pop.select(self._rng, self._cost_evaluator)
            if iters >= 5000:
                configuration, boost = self.get_srex_config(parents)

                offspring = self._crossover(parents, self._data, self._cost_evaluator,
                                            self._rng, configuration)
                self._improve_offspring(offspring, boost)
            else:
                offspring = self._crossover(parents, self._data, self._cost_evaluator,
                                            self._rng)
                self._improve_offspring(offspring, False)

            new_best = self._cost_evaluator.cost(self._best)

            if new_best < curr_best:
                iters_no_improvement = 1
            else:
                iters_no_improvement += 1

            stats.collect_from(self._pop, self._cost_evaluator)

        end = time.perf_counter() - start
        return Result(self._best, stats, iters, end)

    def _improve_offspring(self, sol: Solution, boost: Bool):
        def is_new_best(sol):
            cost = self._cost_evaluator.cost(sol)
            best_cost = self._cost_evaluator.cost(self._best)
            return cost < best_cost

        def add_and_register(sol):
            self._pop.add(sol, self._cost_evaluator)
            self._pm.register_load_feasible(not sol.has_excess_load())
            self._pm.register_time_feasible(not sol.has_time_warp())

        if boost:
            sol = self._search(sol, self._model_CostEvaluator)
        else:
            sol = self._search(sol, self._cost_evaluator)

        add_and_register(sol)

        if is_new_best(sol):
            self._best = sol

        # Possibly repair if current solution is infeasible. In that case, we
        # penalise infeasibility more using a penalty booster.
        if (
                not sol.is_feasible()
                and self._rng.rand() < self._params.repair_probability
        ):
            sol = self._search(sol, self._pm.get_booster_cost_evaluator())

            if sol.is_feasible():
                add_and_register(sol)

            if is_new_best(sol):
                self._best = sol

    def get_srex_config(self, parents):

        parent1, parent2 = parents

        p1_data = ParentGraph(*self._sol_transform(instance=self._data,
                                                   get_full_graph=False,
                                                   parent_solution=parent1))
        p1_data = normalize_graphs(p1_data)
        p1_data.to("cuda")
        p1_data = PE(p1_data)

        p2_data = ParentGraph(*self._sol_transform(instance=self._data,
                                                   get_full_graph=False,
                                                   parent_solution=parent2))
        p2_data = normalize_graphs(p2_data)
        p2_data.to("cuda")
        p2_data = PE(p2_data)

        p1_b = Batch.from_data_list([p1_data])
        p2_b = Batch.from_data_list([p2_data])
        fg_b = Batch.from_data_list([self._full_graph])
        instance_batch = torch.tensor(0).repeat(len(self._full_graph.x))

        self._model.to("cuda")
        fg_b.to("cuda")
        instance_batch.to("cuda")

        self._model.eval()
        out, batch = self._model(p1_b, p2_b, fg_b, instance_batch, 1)

        p1, p2, numMove = parent1.num_routes(), parent2.num_routes(), min(parent1.num_routes(), parent2.num_routes())

        output_shaped = out.reshape(numMove, p1, p2)
        max_v = output_shaped.max()

        SREX_param = torch.where(output_shaped > 0.5)

        if SREX_param[0].nelement() != 0:
            pos = self._rng.randint(SREX_param[0].nelement())
            numMove, p1_idx, P2_idx = SREX_param[0][pos], SREX_param[1][pos], SREX_param[2][pos]
            return (p1_idx, P2_idx, numMove + 1), True
        elif torch.where(output_shaped > 0.3)[0].nelement() != 0:
            SREX_param = torch.where(output_shaped > 0.3)
            pos = self._rng.randint(SREX_param[0].nelement())
            numMove, p1_idx, P2_idx = SREX_param[0][pos], SREX_param[1][pos], SREX_param[2][pos]
            return (p1_idx, P2_idx, numMove + 1), False

        else:
            return None, False









