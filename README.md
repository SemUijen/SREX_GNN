## Graph Neural Network for optimizing Hybrid Genetic Search

This repository contains the code of my thesis, which delved into optimizing the hybrid genetic search algorithm ([Vidal (2022)][1]). For this thesis, the [PyVRP][2] implementation of Hybrid Genetic Search was used.

> [!TIP]
> If you are new to vehicle routing or metaheuristics, the PyVRP library perfectly explains the basics on their website: [introduction to VRP][3] and [introduction to HGS][4].

## Table of Content

- [Abstract](#abstract)
- [Summary of Research Method](#summary-of-research-method)

  - [Introduction](#introduction)
  - [Graph Features](#graph-features)
  - [The Proposed model](#the-proposed-model)
  - [The Node Embedding Transformations](#node-embedding-transformation)

- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Abstract

In recent years, the paradigm Learn to Optimize is making strive in learning how to solve combinatorial optimization problem using machine learning. Most models still get out-performed by the state-of-the-art algorithms, so If you can’t beat them, why not just join them? \
Hybrid Genetic Search(HGS) is one of these state-of-the-art algorithms. Eventhough HGS is state-of-the-art, it still leaves room for increasing efficiency due to its many random aspects, among which is the crossover step. Many studies have used operator selection or parameter adaption methods to address this randomness. Doing direct local genetic selection is less studied.\
To fill this gap, we propose NeuroSREX, which utilizes Selective Route Exchange (SREX) and a Graph Neural Network (GNN) to select specific route(s) of each parent to crossover. While NeuroSREX is not yet practically viable due to difficulties in dealing with constraints and the added computation time, the results demonstrate the efficiency and effectiveness of the model.\
Meaning, we found lower cost solutions in less iterations than HGS. We outperformed HGS on a multitude of instances, taking the first step in the evolution of optimizing evolution.

## Summary of Research Method

### Introduction

The proposed methodology for optimizing the evolution strategy of HGS consists of two parts: The model's prediction of the correct configuration of [Selective Route Exchange crossover][8] (SREX) parameters and the model implementation in HGS. This section briefly discusses how each part is set up.\
The combination of fast-explorative evolutionary search and the aggressive-improvement capabilities of local search is a significant part of HGS performance. Implementing a model for the evolutionary search (i.e. SREX) will add computational overhead to an otherwise very fast step. Therefore, an implementation should not only have good predictions, but should also try to minimize the computational overhead.

### Selective Route Exchange (SREX)

SREX crossover operator combines routes from both parents to generate a new offspring solution. It does this by carefully selecting routes from the second parent that could be exchanged with routes from the first parent. This often results in incomplete offspring that can then be repaired using a search method.\
The code below show the parameters used by the [PyVRP implementation][9]. The `start_indices` and `num_moved_routes` determine which routes of parent 1 and 2 are used for the crossover. These parameters are selected randomly, which is what makes the genetic search fast.

```python
def selective_route_exchange(
    parents: tuple[Solution, Solution],
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    start_indices: tuple[int, int],
    num_moved_routes: int,
) -> Solution
```

> [!IMPORTANT]
> The proposed model of this thesis will make a prediction for all configurations of `start_indices` and `num_moved_routes` given two parents. The best configuration is then selected.

### Graph Features

| **Graph Features**            | **Description**                                               | **Dimension**                    |
| ----------------------------- | ------------------------------------------------------------- | -------------------------------- |
| **Edge Index**                | Graph connectivity in COO format                              | `[2, num_edges]`                 |
| **Edge Weight**               | The cost ($c_{i,j}$) associated with an edge                  | `[num_edges, 1] `                |
| **Node Features**             | Node feature matrix                                           | `[num_nodes, num_node_features]` |
| Coordinates ($x_{i}, y_{i}$)  | Represents the position of customer $i$ on a coordinate graph | `[num_nodes, 2]`                 |
| Timewindow $[a_{i}$, $b_{i}]$ | The time interval customer $i$                                | `[num_nodes, 2]`                 |
| Demand $d_{i}$                | The demand of customer $i$                                    | `[num_nodes, 1]`                 |
| Service time $s_{i}$          | The service time of customer $i$                              | `[num_nodes, 1]`                 |
| Vehicle capacity $Q$          | The demand capacity of vehicles for a given route instance    | `[num_nodes, 1]`                 |
| Number of vehicles $K$        | The number of vehicles for a given route instance             | `[num_nodes, 1]`                 |
| Positional Embeddings $p_i$   | The [LapPE][6] or [RWPE][7] with number of dimension $k$      | `[num_nodes, k]`                 |

### The Proposed model

Figure 1 illustrates the model proposed in this thesis. The architecture comprises three main elements: [the Graph Attention (GAT)](5) networks, the embedding transformations, and, finally, the fully-connected (FC) layers.

<div class="container" align="center">
    <img width=350 src="images/method_images/Model_diagram.png" />
    <div class="overlay">Figure 1: Diagram of proposed model</div>
</div>
</br>
The single input for the model consists of three different graphs: Parent-1 and Parent-2 which both represent a single solution to a VRP-problems, and the Full Graph which represent the whole VRP-problem graph(i.e. all edges and nodes of the problem).

The output, shown in figure 2, is heatmap showing the estimated quality of configurations of the Selective Route Exchange Crossover function given Parent-1 and Parent-2.

<div class="container" align="center">
    <img width=550 src="images/model_result_plots/PlottedResults_Working_1.png" />
    <div class="overlay">Figure 2: Heatmap of estimated best configurations of SREX-crossover for a single crossover</div>
</div>
</br>

### Node Embedding Transformation

Following the GAT layers, node representations are obtained. However, to predict the routes that should be switched between parents (configuration of SREX parameters) to find a better offspring, these embeddings need to be transformed to reflect this interaction. A diagram illustrating these transformation is shown in figure 3.\
In this section, we briefly explain the steps taken to calculate the embeddings for all configurations of the SREX parameters.

> [!NOTE]
> For how the steps are implemented to minimize added computation, we refer to the [source][10] code

The initial step involves aggregating the node embedding into an embedding for each route for both Parent $A$ and $B$, where the total number of routes for a parent is $R$. This aggregation is achieved by summing and then averaging all node embeddings that belong to the same route. This results in a representation for the route of Parent A and B with dimension [$R_a$, $h$] and [$R_b$, $h$] respectively.\
Since SREX switches one or more routes ($R_{move}$), the subsequent step is to aggregate each route embedding to the sum of ordered routes within the range of 1 to $R_{move}$.\
Subsequently, there are also routes that aren't used for the crossover. For each configuration, we also aggregated the route that aren't used and concatenate them with the aggration of routes used. This results in an embedding for each configuration(i.e. `starting_index` of that parent and `num_routes_moved`) of a single parent with dimensions [$R_a$ * $\max (R_{a}, R_b)$, $2*h$].\
Finally, to enable a comparison between Parent $A$ and $B$, the embeddings need to be combined. This is accomplished by concatenating the embeddings that have the same number of routes moved, because the number of routes selected(`num_routes_moved`) is always the same for both parents. The result is a representation for every SREX configuration given two parents, with dimensions
[$R_{a}$ * $R_{b}$ * $\max (R_{a}, R_b)$, 4 * $h$]\\

<div class="container" align="center">
    <img src="images/method_images/NodeEmbeddingTransformation.png" />
    <div class="overlay">Figure 3: Diagram of Embedding Transformations: Each color represent a single h-dimensional embedding representing a node, route or a single configuration(i.e. routes (not) selected for crossover from both parents)</div>
</div>

## Conclusion

When integrated into the Hybrid Genetic Search, the NeuroSREX model approach exhibited potential for enhancing solution quality given enough iterations(30.000+). However, significant compute overhead was introduced through requiring positional encodings([LapPE][6]) for every solution. Addressing this limitation to improve iteration speed remains imperative for practical viability.\
In conclusion, we have taken the first steps toward optimizing genetic selection within the context of routing problems. While it is not currently ready for practical implementation, this study has demonstrated a significant positive impact, warranting further exploration in the optimization of evolutionary algorithms. As with natural evolution, progress is achieved on step at a time.

## Future Work

To create practical viability, we recommend the following steps for future work in order of importance:

- Improve the current implementation by adding positional encodings as a feature to every solution. This would eliminate the need to calculate it repeatedly.
- Train NeuroSREX using an alternative approach for computing the Positional Embeddings: compute the eigenvector for each graph edge, incorporating them as edge features, and subsequently deriving Node PEs based on the Parent Graph edges. This method requires calculating eigenvectors only once for each route instance before running HGS.

- If the aforementioned steps result in practical viability for instances with less challenging constraints, subsequent research efforts should focus on developing methods to incorporate the capability to handle more complex constraints.

[1]: https://doi.org/10.1016/j.cor.2021.105643
[2]: https://github.com/PyVRP/PyVRP
[3]: https://pyvrp.org/setup/introduction_to_vrp.html
[4]: https://pyvrp.org/setup/introduction_to_hgs.html
[5]: https://arxiv.org/abs/1710.10903
[6]: https://arxiv.org/abs/2003.00982
[7]: https://arxiv.org/abs/2110.07875
[8]: https://link.springer.com/chapter/10.1007/978-3-642-15844-5_54
[9]: https://github.com/PyVRP/PyVRP//blob/a860efe/pyvrp/crossover/selective_route_exchange.py#L13
[10]: https://github.com/SemUijen/SREX_GNN/blob/main/models/utils/node_embedding_transformations.py
