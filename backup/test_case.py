import torch
import torch.nn as nn

num_features = 2
hidden_dim = 64
num_heads = 8
num_encoder_layers = 2
initial_layer = nn.Linear(num_features, hidden_dim)
max_routes_to_swap = 90
route_combination_head = nn.Linear(2 * hidden_dim, max_routes_to_swap + 1)

transformer = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(hidden_dim, num_heads),
    num_encoder_layers
)
number_of_customers = 15

# Simple 15 nodes with 2 features
graph_p1 = torch.rand(number_of_customers, num_features)
graph_p2 = torch.rand(number_of_customers, num_features)


# for ease of testing just a simple layer
P1_emmbedding = initial_layer(graph_p1)
P2_emmbedding = initial_layer(graph_p2)
print("Embedding Shape: ", P1_emmbedding.shape)


## Parent 1
P1_number_of_routes = 6
P1_Cnode_to_route = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5])
P1_node_to_route_matrix = torch.zeros(number_of_customers, P1_number_of_routes)
P1_node_to_route_matrix[torch.arange(number_of_customers), P1_Cnode_to_route] = 1


## Parent 2
P2_number_of_routes = 4
P2_Cnode_to_route = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
P2_node_to_route_matrix = torch.zeros(number_of_customers, P2_number_of_routes)
P2_node_to_route_matrix[torch.arange(number_of_customers), P2_Cnode_to_route] = 1


P1_route_aggregating = torch.matmul(P1_node_to_route_matrix.t(), P1_emmbedding)
P2_route_aggregating = torch.matmul(P2_node_to_route_matrix.t(), P2_emmbedding)
print("Parent 1 Route features: ", P1_route_aggregating.shape)
print("Parent 2 Route features: ", P2_route_aggregating.shape)

a, b = torch.broadcast_tensors(P1_route_aggregating[:, None], P1_route_aggregating[None, :])
parent_route_combination_representations = torch.cat((a, b), -1)
print("parent_route_combo: ", parent_route_combination_representations.shape)

max_to_swap = min(P1_number_of_routes, P2_number_of_routes)
full_prediction = route_combination_head(parent_route_combination_representations)
print("Full prediction shape: ", full_prediction.shape)

# Actual size of the allowed matrix =  (P1_number_of_routes, P2_number_of_routes, max_to_swap)
logits = full_prediction[:P1_number_of_routes, :P2_number_of_routes, :max_to_swap + 1]
probs = torch.softmax(logits.flatten(), -1).view_as(logits)

print('Output: ', probs.shape, probs.sum())


highest_prob = probs.max()
SREX_param = torch.where(probs == highest_prob)
print(f"best SrexParams: P1x = {SREX_param[0].item()}, P2x = {SREX_param[1].item()}, NumRoutesMove = {SREX_param[2].item()}")

"""
    embedding1 = torch.ones(8,64)
    embedding2 = torch.ones(8,68)

    emb_stack = torch.stack((embedding1, embedding2))
    print("emb: ", emb_stack.shape)
    ## Parent 2
    P2_Cnode_to_route = torch.tensor([[0,0,0,1,1,2,3,3], [0,1,1,1,1,2,2,3]])
    print(P2_Cnode_to_route.shape)
    P2_node_to_route_matrix = torch.zeros(2, 8, 4)

    batchsize = P2_Cnode_to_route.shape[0]
    for i in range(batchsize):
        P2_node_to_route_matrix[i, torch.arange(8), P2_Cnode_to_route[i]] = 1

    P2_route_aggregating = torch.matmul(P2_node_to_route_matrix[0].t(), embedding1)
    print(P2_route_aggregating)
    result = torch.einsum('abcd, abcd->abcd', P2_node_to_route_matrix.t(), emb_stack)
    print(result.shape)
    print(result)

    #P2_node_to_route_matrix[torch.arange(number_of_customers), P1_Cnode_to_route] = 1
    #P1_route_aggregating = torch.matmul(P1_node_to_route_matrix.t(), P1_emmbedding)
    # batch matrix multiplication
    As = torch.ones(3, 3, 3)
    Bs = torch.ones(3, 3, 3) * 6
    print(Bs)
    sA = torch.ones(3, 3, 3) * 7
    sB = torch.ones(3, 3, 3) * 6

    batch = torch.stack((As, Bs))
    m_batch = torch.stack((sA,sB))
    print("Batch", batch.shape)


    result = torch.einsum('abcd, abcd->abcd', batch, m_batch)
    print(result.shape)
    print(result)"""