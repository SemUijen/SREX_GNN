import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch.nn.functional import pad, softmax
from torch import Tensor


class SREXmodel(nn.Module):

    def __init__(self, num_node_features: int, hidden_dim: int = 64, max_routes_to_swap: int = 60, num_heads: int = 8,
                 dropout: float = 0.2):
        super(SREXmodel, self).__init__()

        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.max_routes_to_swap = max_routes_to_swap
        self.num_heads = num_heads

        # the NN learning the representation of Parent solutions with Node = Customer
        self.GAT_SolutionGraph = GATConv(in_channels=self.num_node_features, out_channels=self.hidden_dim,
                                         heads=self.num_heads, dropout=dropout)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        # TODO: add gat model for FULL Graph

        # TODO: Add extra layers
        self.fc1 = nn.Linear(2 * self.num_heads * self.hidden_dim, self.num_heads * self.hidden_dim)
        self.fc2 = nn.Linear(self.num_heads * self.hidden_dim, int(self.num_heads * self.hidden_dim / 2))
        self.head = nn.Linear(int(self.num_heads * self.hidden_dim / 2), 1)

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def transform_clientEmbeddings_to_routeEmbeddings(self, p1_graph_data, p2_graph_data, p1_embeddings, p2_embeddings):
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        def transform_to_route(graph_data, embeddings):
            node_to_route_vector = graph_data.client_route_vector[batch_indices == i]
            number_of_customers = torch.tensor(len(node_to_route_vector))
            node_to_route_matrix = torch.zeros(number_of_customers, graph_data.num_routes[i], device=device)
            node_to_route_matrix[torch.arange(number_of_customers).long(), node_to_route_vector.long()] = 1
            route_embedding = torch.matmul(node_to_route_matrix.t(), embeddings[batch_indices == i])
            return route_embedding

        def transform_to_nrRoutes(route_embeddings, max_move):

            temp_tensor = []
            nrRoutes_batch = []
            for i1 in range(route_embeddings.shape[0]):
                for i2 in range(1, max_move):
                    # TODO: Sum mean? global pooling?

                    if i1 + i2 > max_move:
                        if i1 > max_move:
                            indices = torch.arange(0, (i1 + i2) - max_move)
                        else:
                            indices = torch.cat((torch.arange(0, (i1 + i2) - max_move), torch.arange(i1, max_move)))

                    else:
                        indices = torch.arange(i1, i1 + i2)
                    temp_tensor.append(torch.sum(route_embeddings[indices], -2))
                    nrRoutes_batch.append(i2)

            return torch.stack(temp_tensor), torch.tensor(nrRoutes_batch)

        # batch size and indices are the same for both parents
        batch_size = len(p1_graph_data)
        batch_indices = p1_graph_data.batch
        PtoP_embeddings = torch.tensor([], device=device)
        PtoP_batch = torch.tensor([], device=device)
        for i in range(batch_size):
            p1_route_embedding = transform_to_route(p1_graph_data, p1_embeddings)
            p2_route_embedding = transform_to_route(p2_graph_data, p2_embeddings)

            max_to_move = min(p1_route_embedding.size(0), p2_route_embedding.size(0))

            p1_sum_of_routes, p1_Route_batch = transform_to_nrRoutes(p1_route_embedding, max_to_move)
            p2_sum_of_routes, p2_Route_batch = transform_to_nrRoutes(p2_route_embedding, max_to_move)

            full_matrix = torch.tensor([], device=device)
            for NrRoutes_move in range(1, max_to_move):
                a, b = torch.broadcast_tensors(p1_sum_of_routes[p1_Route_batch == NrRoutes_move][:, None],
                                               p2_sum_of_routes[p2_Route_batch == NrRoutes_move][None, :])
                test = torch.cat((a, b), -1)
                full_matrix = torch.cat((full_matrix, test.flatten(0, 1)))

            PtoP_embeddings = torch.cat((PtoP_embeddings, full_matrix))

            PtoP_batch = torch.cat((PtoP_batch, torch.tensor([i] * full_matrix.size(0), device=device)))

        return PtoP_embeddings, PtoP_batch

    def forward(self, parent1_data: Data, parent2_data: Data):
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        # get graph input for solution1
        P1_nodefeatures, P1_edge_index, P1_edgeFeatures = parent1_data.x, parent1_data.edge_index, parent1_data.edge_attr
        # get graph input for solution 2
        P2_nodefeatures, P2_edge_index, P2_edgeFeatures = parent2_data.x, parent2_data.edge_index, parent2_data.edge_attr

        # TODO: both embedding have no activation function yet: embedding = self.relu(embedding)?
        # Node(Customer) Embedding Parent1 (Current setup is without whole graph)
        P1_embedding = self.GAT_SolutionGraph(x=P1_nodefeatures.float(), edge_index=P1_edge_index,
                                              edge_attr=P1_edgeFeatures)
        P1_embedding = self.relu(P1_embedding)
        # Node(Customer) Embedding Parent2 (Current setup is without whole graph)
        P2_embedding = self.GAT_SolutionGraph(x=P2_nodefeatures, edge_index=P2_edge_index, edge_attr=P2_edgeFeatures)
        P2_embedding = self.relu(P2_embedding)

        # node embeddings to PtoP_embeddings
        PtoP_embeddings, PtoP_batch = self.transform_clientEmbeddings_to_routeEmbeddings(parent1_data, parent2_data,
                                                                                         P1_embedding, P2_embedding)

        # TODO Add extra linear layers
        # linear layers
        out = self.fc1(PtoP_embeddings)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        full_prediction = self.head(out)

        # Soft_MAX
        output_probs = torch.tensor([], device=device)
        for batch in range(len(parent1_data)):
            batch_predict = full_prediction[PtoP_batch == batch]

            probs = self.sigmoid(batch_predict.flatten())

            output_probs = torch.cat((output_probs, probs))

        return output_probs, PtoP_batch
