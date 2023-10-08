import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class SREXmodel(nn.Module):

    def __init__(self, num_node_features: int, hidden_dim: int = 64, max_routes_to_swap: int = 60, num_heads: int = 8,
                 dropout: float = 0.2):
        super(SREXmodel, self).__init__()

        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.max_routes_to_swap = max_routes_to_swap
        self.num_heads = num_heads
        self.dropout = dropout

        # the NN learning the representation of Parent solutions with Node = Customer
        self.GAT_SolutionGraph = GATConv(in_channels=self.num_node_features, out_channels=self.hidden_dim,
                                         heads=self.num_heads, dropout=self.dropout)

        self.route_combination_head = nn.Linear(2 * self.num_heads * self.hidden_dim, self.max_routes_to_swap + 1)

    def forward(self, parent1_data: Data, parent2_data: Data):
        # TODO: torch_geometric has a dataloader that can work with Batches
        # get graph input for solution1
        P1_nodefeatures, P1_edge_index, P1_edgeFeatures, P1_number_of_routes = parent1_data.x, parent1_data.edge_index, parent1_data.edge_attr, parent1_data.num_routes
        # get graph input for solution 2
        P2_nodefeatures, P2_edge_index, P2_edgeFeatures, P2_number_of_routes = parent2_data.x, parent2_data.edge_index, parent2_data.edge_attr, parent2_data.num_routes

        batch_idx = parent1_data.batch

        print("Batch_size:", len(parent2_data))
        print("node_feature_shape: ", P1_nodefeatures[batch_idx == 0, :].shape)
        number_of_customers = parent1_data.num_nodes

        # TODO: both embedding have no activation function yet: embedding = self.relu(embedding)?
        # Node(Customer) Embedding Parent1 (Current setup is without whole graph)
        P1_embedding = self.GAT_SolutionGraph(x=P1_nodefeatures.float(), edge_index=P1_edge_index,
                                              edge_attr=P1_edgeFeatures)

        # Node(Customer) Embedding Parent2 (Current setup is without whole graph)
        P2_embedding = self.GAT_SolutionGraph(x=P2_nodefeatures, edge_index=P2_edge_index, edge_attr=P2_edgeFeatures)

        print("node_embedding_shape: ", P1_embedding.shape)

        # TODO: Current steps dont work with batches LOOK AT TORCH.EINSUM
        # CustomerNode Embedding to RouteNode Embedding: Average over Customers in Route.
        # Cnode_Rnode_vector[0] = 1 means that customer node 0 belongs to route 1
        print(torch.sum(P1_number_of_routes).item())
        ## Parent 1
        P1_Cnode_to_route = parent1_data.client_route_vector
        P1_node_to_route_matrix = torch.zeros(number_of_customers, torch.sum(P1_number_of_routes).item())
        P1_node_to_route_matrix[torch.arange(number_of_customers), P1_Cnode_to_route] = 1

        ## Parent 2
        P2_Cnode_to_route = parent2_data.client_route_vector
        P2_node_to_route_matrix = torch.zeros(number_of_customers, P2_number_of_routes)
        P2_node_to_route_matrix[torch.arange(number_of_customers), P2_Cnode_to_route] = 1

        P1_route_aggregating = torch.matmul(P1_node_to_route_matrix.t(), P1_embedding)
        P2_route_aggregating = torch.matmul(P2_node_to_route_matrix.t(), P2_embedding)

        # Route_combination_head
        a, b = torch.broadcast_tensors(P1_route_aggregating[:, None], P2_route_aggregating[None, :])
        parent_route_combination_representations = torch.cat((a, b), -1)

        # TODO Add extra linear layers
        full_prediction = self.route_combination_head(parent_route_combination_representations)

        # Soft_MAX
        # Actual size of the allowed matrix =  (P1_number_of_routes, P2_number_of_routes, max_to_swap)
        max_to_swap = min(P1_number_of_routes, P2_number_of_routes)
        logits = full_prediction[:P1_number_of_routes, :P2_number_of_routes, :max_to_swap - 1]
        probs = torch.softmax(logits.flatten(), -1).view_as(logits)

        return probs
