import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.data import Data
from torch.nn.functional import pad, softmax
from torch import Tensor
from torch_geometric.nn.norm import BatchNorm, LayerNorm

class SREXmodel(nn.Module):

    def __init__(self, num_node_features: int, hidden_dim: int = 64, num_heads: int = 8,
                 dropout: float = 0.2):
        super(SREXmodel, self).__init__()

        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # the NN learning the representation of Parent solutions with Node = Customer
        self.GAT_SolutionGraph = GATv2Conv(in_channels=self.num_node_features, out_channels=self.hidden_dim,
                                         heads=self.num_heads, dropout=0, edge_dim=1)

        self.GAT_FullGraph = GATv2Conv(in_channels=self.num_node_features, out_channels=self.hidden_dim,
                                     heads=self.num_heads, dropout=0, edge_dim=1)

        self.GAT_both = GATv2Conv(in_channels=2*self.hidden_dim*self.num_heads, out_channels=2*self.hidden_dim,
                                     heads=self.num_heads, dropout=0, edge_dim=1)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        # TODO: add gat model for FULL Graph

        self.PtoPNorm = BatchNorm(8 * self.num_heads * self.hidden_dim)
        self.BothNorm = BatchNorm(2 * self.num_heads * self.hidden_dim)

        # TODO: Add extra layers
        self.fc1 = nn.Linear(8 * self.num_heads * self.hidden_dim, int(self.num_heads * self.hidden_dim)*4)
        self.fc2 = nn.Linear(4*int(self.num_heads * self.hidden_dim), int(self.num_heads * self.hidden_dim)*2)
        self.fc3 = nn.Linear(int(self.num_heads * self.hidden_dim)*2, int(self.num_heads * self.hidden_dim))
        self.fc4 = nn.Linear(int(self.num_heads * self.hidden_dim), int(self.num_heads * self.hidden_dim / 2))
        self.fc5 = nn.Linear(int(self.num_heads * self.hidden_dim/2), int(self.num_heads * self.hidden_dim/4))
        self.fc6 = nn.Linear(int(self.num_heads * self.hidden_dim/4), int(self.num_heads * self.hidden_dim/8))
        self.head = nn.Linear(int(self.num_heads * self.hidden_dim / 8), 1)

        self.sigmoid = nn.Sigmoid()

    def transform_clientEmbeddings_to_routeEmbeddings(self, p1_graph_data, p2_graph_data, p1_embeddings, p2_embeddings, epoch):
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        def transform_to_route(graph_data, embeddings, batch_indices, batch_idx):
            node_to_route_vector = graph_data.client_route_vector[batch_indices == batch_idx]
            number_of_customers = torch.tensor(len(node_to_route_vector))
            node_to_route_matrix = torch.zeros(number_of_customers, graph_data.num_routes[batch_idx], device=device)
            node_to_route_matrix[torch.arange(number_of_customers).long(), node_to_route_vector.long()] = 1
            route_embedding = torch.matmul(node_to_route_matrix.t(), embeddings[batch_indices == batch_idx])

            nodes_per_route = torch.sum(node_to_route_matrix.t(), 1).unsqueeze(-1)
            route_embedding = route_embedding / nodes_per_route
            return route_embedding

        def transform_to_nrRoutes(route_embeddings, max_move):

            embedding_dim = route_embeddings.shape[1]
            num_routes = route_embeddings.shape[0]

            route_embeddings2 = torch.cat((route_embeddings, route_embeddings), 0)
            cumsum2 = torch.cat((torch.ones(1, embedding_dim, device=device), torch.cumsum(route_embeddings2, 0)), 0)

            #routes that are moved
            i1 = torch.arange(num_routes)
            num_move = torch.arange(1, max_move + 1)
            end_idx = i1[:, None] + num_move[None, :]
            start_idx = i1[:, None]
            diff = (cumsum2[end_idx, :] - cumsum2[start_idx])
            num_move_batch = num_move.repeat(num_routes)
            embeddings_moved = diff.view(-1, embedding_dim)
            embeddings_moved = embeddings_moved/num_move_batch.unsqueeze(-1).to(device)

            # Routes that are not moved
            i2 = torch.arange(1, num_routes + 1)
            num_move2 = torch.flip(torch.arange(num_routes - max_move, num_routes), dims=(0,))
            end_idx2 = i2[:, None] + num_move2[None, :]
            start_idx2 = i2[:, None]
            diff2 = (cumsum2[end_idx2, :] - cumsum2[start_idx2])

            embeddings_others = diff2.view(-1, embedding_dim)
            num_move2_batch = num_move2.repeat(num_routes)
            num_move2_batch[num_move2_batch==0] = 1
            embeddings_others = embeddings_others / num_move2_batch.unsqueeze(-1).to(device)

            return torch.cat((embeddings_moved, embeddings_others), dim=1), num_move_batch

        # batch size and indices are the same for both parents
        batch_size = len(p1_graph_data)
        P1_batch_indices = p1_graph_data.batch
        P2_batch_indices = p2_graph_data.batch
        PtoP_embeddings = torch.tensor([], device=device)
        PtoP_batch = torch.tensor([], device=device)
        for batch_idx in range(batch_size):
            p1_route_embedding = transform_to_route(p1_graph_data, p1_embeddings, P1_batch_indices, batch_idx)
            p2_route_embedding = transform_to_route(p2_graph_data, p2_embeddings, P2_batch_indices, batch_idx)

            max_to_move = min(p1_route_embedding.size(0), p2_route_embedding.size(0))

            p1_sum_of_routes, p1_Route_batch = transform_to_nrRoutes(p1_route_embedding, max_to_move)
            p2_sum_of_routes, p2_Route_batch = transform_to_nrRoutes(p2_route_embedding, max_to_move)


            full_matrix = torch.tensor([], device=device)

            for NrRoutes_move in range(1, max_to_move + 1):
                a, b = torch.broadcast_tensors(p1_sum_of_routes[p1_Route_batch == NrRoutes_move][:, None],
                                               p2_sum_of_routes[p2_Route_batch == NrRoutes_move][None, :])
                test = torch.cat((a, b), -1)
                full_matrix = torch.cat((full_matrix, test.flatten(0, 1)))


            PtoP_embeddings = torch.cat((PtoP_embeddings, full_matrix))
            PtoP_batch = torch.cat((PtoP_batch, torch.tensor([batch_idx] * full_matrix.size(0), device=device)))

        return PtoP_embeddings, PtoP_batch

    def forward(self, parent1_data: Data, parent2_data: Data, full_graph: Data, instance_batch, epoch):

        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        # get graph input for solution1
        P1_nodefeatures, P1_edge_index, P1_edgeFeatures = parent1_data.x, parent1_data.edge_index, parent1_data.edge_attr
        # get graph input for solution 2
        P2_nodefeatures, P2_edge_index, P2_edgeFeatures = parent2_data.x, parent2_data.edge_index, parent2_data.edge_attr
        # get graph input for full graph
        nodefeatures, edge_index, edgeFeatures = full_graph.x, full_graph.edge_index, full_graph.edge_attr

        # TODO: both embedding have no activation function yet: embedding = self.relu(embedding)?
        # Node(Customer) Embedding Parent1 (Current setup is without whole graph)
        P1_embedding = self.GAT_SolutionGraph(x=P1_nodefeatures.float(), edge_index=P1_edge_index,
                                              edge_attr=P1_edgeFeatures)
        P1_embedding = self.relu(P1_embedding)
        # Node(Customer) Embedding Parent2 (Current setup is without whole graph)
        P2_embedding = self.GAT_SolutionGraph(x=P2_nodefeatures, edge_index=P2_edge_index, edge_attr=P2_edgeFeatures)
        P2_embedding = self.relu(P2_embedding)

        full_embedding = self.GAT_FullGraph(x=nodefeatures, edge_index=edge_index, edge_attr=edgeFeatures)


        P1f_embedding = torch.tensor([], device=device)
        P2f_embedding = torch.tensor([], device=device)
        for fg_idx in range(len(full_graph)):
            repeat = int(
                len(instance_batch[instance_batch == fg_idx]) / len(full_embedding[full_graph.batch == fg_idx]))
            temp = torch.cat((P1_embedding[instance_batch == fg_idx], full_embedding[full_graph.batch == fg_idx].repeat(repeat, 1)), dim=1)
            P1f_embedding = torch.cat((P1f_embedding, temp))

            temp = torch.cat((P2_embedding[instance_batch == fg_idx], full_embedding[full_graph.batch == fg_idx].repeat(repeat, 1)), dim=1)
            P2f_embedding = torch.cat((P2f_embedding, temp))

        P1f_embedding = self.BothNorm(P1f_embedding)
        P2f_embedding = self.BothNorm(P2f_embedding)

        P1f_embedding = self.GAT_both(x=P1f_embedding, edge_index=P1_edge_index, edge_attr=P1_edgeFeatures)
        P1f_embedding = self.relu(P1f_embedding)
        P2f_embedding = self.GAT_both(x=P2f_embedding, edge_index=P2_edge_index, edge_attr=P2_edgeFeatures)
        P2f_embedding = self.relu(P2f_embedding)

        # node embeddings to PtoP_embeddings
        PtoP_embeddings, PtoP_batch = self.transform_clientEmbeddings_to_routeEmbeddings(parent1_data, parent2_data,
                                                                                         P1f_embedding, P2f_embedding, epoch)


        PtoP_embeddings = self.PtoPNorm(PtoP_embeddings)
        out = self.fc1(PtoP_embeddings)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.head(out)
        probs = self.sigmoid(out)

        return probs.flatten(), PtoP_batch

