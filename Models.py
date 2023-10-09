import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch.nn.functional import pad


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

        self.route_combination_head = nn.Linear(2 * self.num_heads * self.hidden_dim, self.max_routes_to_swap - 1)

    def transform_clientEmbeddings_to_routeEmbeddings(self, p1_graph_data, p2_graph_data, p1_embeddings, p2_embeddings):
        def transform(graph_data, embeddings):
            node_to_route_vector = graph_data.client_route_vector[batch_indices == i]
            number_of_customers = len(node_to_route_vector)
            node_to_route_matrix = torch.zeros(number_of_customers, graph_data.num_routes[i])
            node_to_route_matrix[torch.arange(number_of_customers), node_to_route_vector] = 1
            route_embedding = torch.matmul(node_to_route_matrix.t(), embeddings[batch_indices == i])
            return route_embedding

        # batch size and indices are the same for both parents
        batch_size = len(p1_graph_data)
        batch_indices = p1_graph_data.batch
        PtoP_embeddings = []
        for i in range(batch_size):
            p1_route_embedding = transform(p1_graph_data, p1_embeddings)
            p2_route_embedding = transform(p2_graph_data, p2_embeddings)

            a, b = torch.broadcast_tensors(p1_route_embedding[:, None], p2_route_embedding[None, :])
            parent_to_parent_embedding = torch.cat((a, b), -1)
            x, y, z = parent_to_parent_embedding.shape
            padded_PtoP_embedding = pad(parent_to_parent_embedding,
                                        pad=(0, 0, 0, self.max_routes_to_swap - y, 0, self.max_routes_to_swap - x),
                                        mode='constant', value=0)
            PtoP_embeddings.append(padded_PtoP_embedding)

        PtoP_embeddings = torch.stack(PtoP_embeddings)

        return PtoP_embeddings

    def forward(self, parent1_data: Data, parent2_data: Data):

        # get graph input for solution1
        P1_nodefeatures, P1_edge_index, P1_edgeFeatures = parent1_data.x, parent1_data.edge_index, parent1_data.edge_attr
        # get graph input for solution 2
        P2_nodefeatures, P2_edge_index, P2_edgeFeatures = parent2_data.x, parent2_data.edge_index, parent2_data.edge_attr

        # batch_idx = parent1_data.batch
        # print("node_feature_shape: ", P1_nodefeatures[batch_idx == 0, :].shape)

        # TODO: both embedding have no activation function yet: embedding = self.relu(embedding)?
        # Node(Customer) Embedding Parent1 (Current setup is without whole graph)
        P1_embedding = self.GAT_SolutionGraph(x=P1_nodefeatures.float(), edge_index=P1_edge_index,
                                              edge_attr=P1_edgeFeatures)

        # Node(Customer) Embedding Parent2 (Current setup is without whole graph)
        P2_embedding = self.GAT_SolutionGraph(x=P2_nodefeatures, edge_index=P2_edge_index, edge_attr=P2_edgeFeatures)

        route_to_route_embeddings = self.transform_clientEmbeddings_to_routeEmbeddings(parent1_data, parent2_data,
                                                                                       P1_embedding, P2_embedding)

        # TODO Add extra linear layers
        full_prediction = self.route_combination_head(route_to_route_embeddings)

        # Soft_MAX
        probs = torch.softmax(full_prediction.flatten(), -1).view_as(full_prediction)

        return probs
