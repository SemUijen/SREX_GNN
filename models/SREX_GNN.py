from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.norm import BatchNorm

from .utils import (
    get_lim_config,
    transform_to_nrRoutes,
    transform_to_route,
)


class SREXmodel(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.05,
    ):
        super(SREXmodel, self).__init__()

        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # the NN learning the representation of Parent solutions with Node = Customer
        self.GAT_SolutionGraph = GATv2Conv(
            in_channels=self.num_node_features,
            out_channels=self.hidden_dim,
            heads=self.num_heads,
            dropout=0,
            edge_dim=1,
        )

        self.GAT_FullGraph = GATv2Conv(
            in_channels=self.num_node_features,
            out_channels=self.hidden_dim,
            heads=self.num_heads,
            dropout=0,
            edge_dim=1,
        )

        self.GAT_both = GATv2Conv(
            in_channels=2 * self.hidden_dim * self.num_heads,
            out_channels=2 * self.hidden_dim,
            heads=self.num_heads,
            dropout=0,
            edge_dim=1,
        )

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        # TODO: add gat model for FULL Graph

        self.PtoPNorm = BatchNorm(
            8 * self.num_heads * self.hidden_dim, track_running_stats=False
        )
        self.BothNorm = BatchNorm(
            2 * self.num_heads * self.hidden_dim, track_running_stats=False
        )

        self.fc1 = nn.Linear(
            8 * self.num_heads * self.hidden_dim,
            int(self.num_heads * self.hidden_dim) * 4,
        )
        self.fc2 = nn.Linear(
            4 * int(self.num_heads * self.hidden_dim),
            int(self.num_heads * self.hidden_dim) * 2,
        )
        self.fc3 = nn.Linear(
            int(self.num_heads * self.hidden_dim) * 2,
            int(self.num_heads * self.hidden_dim),
        )
        self.fc4 = nn.Linear(
            int(self.num_heads * self.hidden_dim),
            int(self.num_heads * self.hidden_dim / 2),
        )
        self.fc5 = nn.Linear(
            int(self.num_heads * self.hidden_dim / 2),
            int(self.num_heads * self.hidden_dim / 4),
        )
        self.fc6 = nn.Linear(
            int(self.num_heads * self.hidden_dim / 4),
            int(self.num_heads * self.hidden_dim / 8),
        )
        self.fc7 = nn.Linear(
            int(self.num_heads * self.hidden_dim / 8),
            int(self.num_heads * self.hidden_dim / 16),
        )
        self.head = nn.Linear(int(self.num_heads * self.hidden_dim / 16), 1)

        self.sigmoid = nn.Sigmoid()

    def transform_clientEmbeddings_to_routeEmbeddings(
        self,
        p1_graph_data: Data,
        p2_graph_data: Data,
        p1_embeddings: Tensor,
        p2_embeddings: Tensor,
    ) -> Union[Tensor, Tensor]:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        # batch size and indices are the same for both parents
        batch_size = len(p1_graph_data)
        P1_batch_indices = p1_graph_data.batch
        P2_batch_indices = p2_graph_data.batch

        # Creating two empty Tensor (eventual results of transformation)
        PtoP_embeddings = torch.tensor([], device=device)
        PtoP_batch = torch.tensor([], device=device)

        # Each batch consists of multiple prediction of parents to crossover
        for batch_idx in range(batch_size):
            # First turn each individual node embedding to the sum of nodes in a route -> Route Embeddings
            p1_route_embedding = transform_to_route(
                p1_graph_data, p1_embeddings, P1_batch_indices, batch_idx
            )
            p2_route_embedding = transform_to_route(
                p2_graph_data, p2_embeddings, P2_batch_indices, batch_idx
            )

            # p1_n and p2_m are the number of routes in parent 1 and parent 2 respectively
            p1_n, p2_n = p1_route_embedding.size(0), p2_route_embedding.size(0)
            max_to_move = min(p1_n, p2_n)

            # The second step is to sum up the Route Embeddings -> nrRoutes Embeddings
            p1_sum_of_routes, p1_Route_batch = transform_to_nrRoutes(
                p1_route_embedding, max_to_move
            )
            p2_sum_of_routes, p2_Route_batch = transform_to_nrRoutes(
                p2_route_embedding, max_to_move
            )

            # The final step is to get the configurations(i.e. a single local selection as illustrated in the README) of SREX parameters
            PtoP_embeddings, PtoP_batch = get_lim_config(
                PtoP_embeddings,
                PtoP_batch,
                p1_sum_of_routes,
                p2_sum_of_routes,
                batch_idx,
                max_to_move,
                p1_n,
                p2_n,
            )

        return PtoP_embeddings, PtoP_batch

    def forward(
        self,
        parent1_data: Data,
        parent2_data: Data,
        full_graph: Data,
        instance_batch: Tensor,
        epoch: int,
    ) -> Union[Tensor, Tensor]:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        # get graph input for solution1
        P1_nodefeatures, P1_edge_index, P1_edgeFeatures = (
            parent1_data.x,
            parent1_data.edge_index,
            parent1_data.edge_attr,
        )
        # get graph input for solution 2
        P2_nodefeatures, P2_edge_index, P2_edgeFeatures = (
            parent2_data.x,
            parent2_data.edge_index,
            parent2_data.edge_attr,
        )
        # get graph input for full graph
        nodefeatures, edge_index, edgeFeatures = (
            full_graph.x,
            full_graph.edge_index,
            full_graph.edge_attr,
        )

        # Node(Customer) Embedding Parent1 (Current setup is without whole graph)
        P1_embedding = self.GAT_SolutionGraph(
            x=P1_nodefeatures.float(),
            edge_index=P1_edge_index,
            edge_attr=P1_edgeFeatures,
        )

        # Node(Customer) Embedding Parent2 (Current setup is without whole graph)
        P2_embedding = self.GAT_SolutionGraph(
            x=P2_nodefeatures, edge_index=P2_edge_index, edge_attr=P2_edgeFeatures
        )

        # Embeddings of full graph
        full_embedding = self.GAT_FullGraph(
            x=nodefeatures, edge_index=edge_index, edge_attr=edgeFeatures
        )

        # Creating to empty tensors for combinining Full graph and parent graph embeddings
        P1f_embedding = torch.tensor([], device=device)
        P2f_embedding = torch.tensor([], device=device)
        for fg_idx in range(len(full_graph)):
            repeat = int(
                len(instance_batch[instance_batch == fg_idx])
                / len(full_embedding[full_graph.batch == fg_idx])
            )
            temp = torch.cat(
                (
                    P1_embedding[instance_batch == fg_idx],
                    full_embedding[full_graph.batch == fg_idx].repeat(repeat, 1),
                ),
                dim=1,
            )
            P1f_embedding = torch.cat((P1f_embedding, temp))

            temp = torch.cat(
                (
                    P2_embedding[instance_batch == fg_idx],
                    full_embedding[full_graph.batch == fg_idx].repeat(repeat, 1),
                ),
                dim=1,
            )
            P2f_embedding = torch.cat((P2f_embedding, temp))

        # normalization
        P1f_embedding = self.BothNorm(P1f_embedding)
        P2f_embedding = self.BothNorm(P2f_embedding)

        # One more GAT Layer
        P1f_embedding = self.GAT_both(
            x=P1f_embedding, edge_index=P1_edge_index, edge_attr=P1_edgeFeatures
        )

        P2f_embedding = self.GAT_both(
            x=P2f_embedding, edge_index=P2_edge_index, edge_attr=P2_edgeFeatures
        )

        # node embeddings to PtoP_embeddings (i.e. all Local Selections)
        PtoP_embeddings, PtoP_batch = (
            self.transform_clientEmbeddings_to_routeEmbeddings(
                parent1_data, parent2_data, P1f_embedding, P2f_embedding, epoch
            )
        )

        # Fully connected layers with relu activations
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
        out = self.fc7(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.head(out)
        probs = self.sigmoid(out)

        return probs.flatten(), PtoP_batch
