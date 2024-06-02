import torch
from torch import Tensor

from data.utils import ParentGraph


def transform_to_route(
    graph_data: ParentGraph,
    embeddings: Tensor,
    batch_indices: Tensor,
    batch_idx: Tensor,
    device,
) -> Tensor:
    """Transforms the node embeddings to route embeddings

    Args:
        graph_data::[ParentGraph]
            Contains the graph features of a parent which contains the "client_route_vector" needed for the implementation
        embeddings::[Tensor]
            The node embeddings of a parent after the GAT layers
        batch_indices::[Tensor]
            The indices of nodes being part of parent x in a batch, needed to do batch calculations
        batch_idx::[Tensor]
            The current batch being calculated
        device
            if model is run on CUDA or CPU

    Returns:
        route_embedding::[Tensor]
            The route_embeddings of all parents in the batch
    """

    # the node_to_route_vector contains the route each node belongs to (e.g. [0,1,1,2])
    node_to_route_vector = graph_data.client_route_vector[batch_indices == batch_idx]
    number_of_customers = torch.tensor(len(node_to_route_vector))
    node_to_route_matrix = torch.zeros(
        number_of_customers, graph_data.num_routes[batch_idx], device=device
    )

    # Create the node_to_route matrix [nr_nodes, nr_routes] by assigning 1 where node is in route
    node_to_route_matrix[
        torch.arange(number_of_customers).long(), node_to_route_vector.long()
    ] = 1

    # Step 3: Multiply node embeddings by transposed node_to_route_matrix to create routes embeddings
    route_embedding = torch.matmul(
        node_to_route_matrix.t(), embeddings[batch_indices == batch_idx]
    )

    # step 4: Average embeddings by amount of nodes(stops) in a route
    nodes_per_route = torch.sum(node_to_route_matrix.t(), 1).unsqueeze(-1)
    route_embedding = route_embedding / nodes_per_route

    return route_embedding


def transform_to_nrRoutes(route_embeddings: Tensor, max_move: Tensor, device):
    """Transforms the route embeddings to sum of routes

    Args:
        route_embeddings::[Tensor]
            The embeddings for each route in a parent
        max_moved::[Tensor]
            The maximum number of routes to moved (max number of routes of one of the parents)
        device
            if model is run on CUDA or CP

    """

    embedding_dim = route_embeddings.shape[1]
    num_routes = route_embeddings.shape[0]

    route_embeddings2 = torch.cat((route_embeddings, route_embeddings), 0)
    cumsum2 = torch.cat(
        (
            torch.zeros(1, embedding_dim, device=device),
            torch.cumsum(route_embeddings2, 0),
        ),
        0,
    )

    # routes that are moved
    i1 = torch.arange(num_routes)
    num_move = torch.arange(1, max_move + 1)
    end_idx = i1[:, None] + num_move[None, :]
    start_idx = i1[:, None]
    diff = cumsum2[end_idx, :] - cumsum2[start_idx]
    num_move_batch = num_move.repeat(num_routes)
    embeddings_moved = diff.view(-1, embedding_dim)
    embeddings_moved = embeddings_moved / num_move_batch.unsqueeze(-1).to(device)

    # Routes that are not moved
    i2 = torch.arange(1, num_routes + 1)
    num_move2 = torch.flip(torch.arange(num_routes - max_move, num_routes), dims=(0,))
    end_idx2 = i2[:, None] + num_move2[None, :]
    start_idx2 = i2[:, None]
    diff2 = cumsum2[end_idx2, :] - cumsum2[start_idx2]

    embeddings_others = diff2.view(-1, embedding_dim)
    num_move2_batch = num_move2.repeat(num_routes)
    num_move2_batch[num_move2_batch == 0] = 1
    embeddings_others = embeddings_others / num_move2_batch.unsqueeze(-1).to(device)

    return torch.cat((embeddings_moved, embeddings_others), dim=1), num_move_batch


def get_all_config(
    PtoP_embeddings,
    PtoP_batch,
    p1_sum_of_routes,
    p1_Route_batch,
    p2_sum_of_routes,
    p2_Route_batch,
    batch_idx,
    device,
):
    full_matrix = torch.tensor([], device=device)
    for NrRoutes_move in range(1, max_to_move + 1):
        a, b = torch.broadcast_tensors(
            p1_sum_of_routes[p1_Route_batch == NrRoutes_move][:, None],
            p2_sum_of_routes[p2_Route_batch == NrRoutes_move][None, :],
        )
        test = torch.cat((a, b), -1)
        full_matrix = torch.cat((full_matrix, test.flatten(0, 1)))

    PtoP_embeddings = torch.cat((PtoP_embeddings, full_matrix))
    PtoP_batch = torch.cat(
        (
            PtoP_batch,
            torch.tensor([batch_idx] * full_matrix.size(0), device=device),
        )
    )

    return PtoP_embeddings, PtoP_batch


def get_lim_config(
    PtoP_embeddings,
    PtoP_batch,
    p1_sum_of_routes,
    p2_sum_of_routes,
    batch_idx,
    max_to_move,
    P1_routes,
    p2_routes,
    device,
):
    if p1_sum_of_routes.shape[0] == p2_sum_of_routes.shape[0]:
        PtoP = torch.cat((p1_sum_of_routes, p2_sum_of_routes), dim=-1)

    elif p1_sum_of_routes.shape[0] > p2_sum_of_routes.shape[0]:
        p2_sum_of_routes = torch.cat(
            (
                p2_sum_of_routes,
                p2_sum_of_routes[:max_to_move].repeat((P1_routes - p2_routes, 1)),
            )
        )
        PtoP = torch.cat((p1_sum_of_routes, p2_sum_of_routes), dim=-1)
    else:
        PtoP = torch.cat(
            (p1_sum_of_routes, p2_sum_of_routes[: p1_sum_of_routes.shape[0]]),
            dim=-1,
        )

    PtoP_embeddings = torch.cat((PtoP_embeddings, PtoP))
    PtoP_batch = torch.cat(
        (PtoP_batch, torch.tensor([batch_idx] * PtoP.size(0), device=device))
    )

    return PtoP_embeddings, PtoP_batch
