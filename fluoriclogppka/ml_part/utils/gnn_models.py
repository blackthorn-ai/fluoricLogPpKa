import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import function as fn
from dgl.nn.pytorch.softmax import edge_softmax

from dgllife.model.model_zoo.gcn_predictor import GCNPredictor

class AttentiveGRU1(nn.Module):
    """Update node features with attention and GRU.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node (atom) features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, edge_feats, node_feats):
        """
        Parameters
        ----------
        g : DGLGraph
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        edge_feats : float32 tensor of shape (E, M1)
            Previous edge features.
        node_feats : float32 tensor of shape (V, M2)
            Previous node features.

        Returns
        -------
        float32 tensor of shape (V, M2)
            Updated node features.
        """
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class AttentiveGRU2(nn.Module):
    """Update node features with attention and GRU.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node (atom) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        """
        Parameters
        ----------
        g : DGLGraph
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        node_feats : float32 tensor of shape (V, M2)
            Previous node features.

        Returns
        -------
        float32 tensor of shape (V, M2)
            Updated node features.
        """
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.u_mul_e('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class GetContext(nn.Module):
    """Generate context for each node (atom) by message passing at the beginning.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node (atom) features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    graph_feat_size : int
        Size of the learned graph representation (molecular fingerprint).
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def apply_edges1(self, edges):
        """Edge feature update."""
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        """Edge feature update."""
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        """
        Parameters
        ----------
        g : DGLGraph
            Constructed DGLGraphs.
        node_feats : float32 tensor of shape (V, N1)
            Input node features. V for the number of nodes and N1 for the feature size.
        edge_feats : float32 tensor of shape (E, N2)
            Input edge features. E for the number of edges and N2 for the feature size.

        Returns
        -------
        float32 tensor of shape (V, N3)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        temp_hv_new = g.ndata['hv_new']
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        temp_he1_new = g.edata['he1']
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        temp_ehe1_new = g.edata['he1']
        g.apply_edges(self.apply_edges2)
        temp_he2_new = g.edata['he2']
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])

class GNNLayer(nn.Module):
    """GNNLayer for updating node features.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    graph_feat_size : int
        Size for the input graph features.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.g = None

    def apply_edges(self, edges):
        """Edge feature update by concatenating the features of the destination
        and source nodes."""
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        """
        Parameters
        ----------
        g : DGLGraph
            Constructed DGLGraphs.
        node_feats : float32 tensor of shape (V, N1)
            Input node features. V for the number of nodes and N1 for the feature size.

        Returns
        -------
        float32 tensor of shape (V, N1)
            Updated node features.
        """
        if 'he' in g.edata:
            pass
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        self.g = g

        if 'he' in g.edata:
            pass

        return self.attentive_gru(g, logits, node_feats)


class Pka_acidic_view(nn.Module):
    """
    A neural network model for predicting pKa values of acidic compounds.

    This model takes as input molecular graphs and predicts pKa values for each atom.

    Attributes:
        node_feat_size (int): The size of node features.
        edge_feat_size (int): The size of edge features.
        num_layers (int): The number of GNN layers.
        graph_feat_size (int): The size of graph-level features.
        output_size (int): The output size of the model.
        dropout (float): The dropout rate.

    Methods:
        __init__(): Initializes the Pka_acidic_view object.
        forward(): Defines the forward pass of the model.
    """
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers,
                 graph_feat_size,
                 output_size,
                 dropout):
        """
        Initialize the Pka_acidic_view object.

        Args:
            node_feat_size (int): The size of node features.
            edge_feat_size (int): The size of edge features.
            num_layers (int): The number of GNN layers.
            graph_feat_size (int): The size of graph-level features.
            output_size (int): The output size of the model.
            dropout (float): The dropout rate.

        Returns:
            None
        """
        super(Pka_acidic_view,self).__init__()

        self.device = torch.device("cpu")

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, output_size)
        )

    def forward(self, g, node_feats, edge_feats, get_node_weight=False):
        """
        Defines the forward pass of the model.

        Args:
            g (DGLGraph): The input molecular graph.
            node_feats (tensor): Node features.
            edge_feats (tensor): Edge features.
            get_node_weight (bool): Whether to compute node weights.

        Returns:
            g_feats (tensor): Graph-level features.
            atom_pka_out (list): Predicted pKa values for each atom.
        """
        mask = torch.sum(g.ndata['h'][:,-4:],dim = 1) * (1 - g.ndata['h'][:,0])
        mask = 1/mask -1
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        atom_pka = self.predict(node_feats)
        atom_pka = -(atom_pka + mask.reshape(-1,1))
        g.ndata['h'] = torch.pow(10,atom_pka)
        g_feats = -torch.log10(dgl.sum_nodes(g, 'h'))
        atom_pka_out = atom_pka * -1
        atom_pka_out = torch.squeeze(atom_pka_out)
        
        return g_feats, atom_pka_out.detach().cpu().numpy().tolist()


class Pka_basic_view(nn.Module):
    """
    A neural network model for predicting pKa values of basic compounds.

    This model takes as input molecular graphs and predicts pKa values for each atom.

    Attributes:
        node_feat_size (int): The size of node features.
        edge_feat_size (int): The size of edge features.
        num_layers (int): The number of GNN layers.
        graph_feat_size (int): The size of graph-level features.
        output_size (int): The output size of the model.
        dropout (float): The dropout rate.

    Methods:
        __init__(): Initializes the Pka_basic_view object.
        forward(): Defines the forward pass of the model.
    """
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers,
                 graph_feat_size,
                 output_size,
                 dropout):
        """
        Initialize the Pka_basic_view object.

        Args:
            node_feat_size (int): The size of node features.
            edge_feat_size (int): The size of edge features.
            num_layers (int): The number of GNN layers.
            graph_feat_size (int): The size of graph-level features.
            output_size (int): The output size of the model.
            dropout (float): The dropout rate.

        Returns:
            None
        """
        super(Pka_basic_view,self).__init__()

        self.device = torch.device("cpu")

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, output_size)
        )
        
    def forward(self, g, node_feats, edge_feats, get_node_weight=False):
        """
        Defines the forward pass of the model.

        Args:
            g (DGLGraph): The input molecular graph.
            node_feats (tensor): Node features.
            edge_feats (tensor): Edge features.
            get_node_weight (bool): Whether to compute node weights.

        Returns:
            g_feats (tensor): Graph-level features.
            atom_pka_out (list): Predicted pKa values for each atom.
        """
        mask = g.ndata['h'][:,1] * (1 - g.ndata['h'][:,61])

        mask = -1/mask +1

        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        atom_pka = self.predict(node_feats)

        atom_pKa_after_predict = atom_pka.clone()
        atom_pka = (atom_pka + mask.reshape(-1,1))

        g.ndata['h'] = torch.pow(10,atom_pka)
        g_feats = torch.log10(dgl.sum_nodes(g, 'h'))
        atom_pka_out = torch.squeeze(atom_pka)
        
        return g_feats, atom_pka_out.detach().cpu().numpy().tolist()

class PKaAcidicModel:
    """
    A wrapper class for the pKa acidic prediction model.

    This class provides methods for loading and using the pre-trained pKa acidic prediction model.

    Attributes:
        model (nn.Module): The pre-trained pKa acidic prediction model.

    Methods:
        __init__(): Initializes the PKaAcidicModel object.
        eval(): Puts the model in evaluation mode.
        predict(): Makes predictions using the loaded model.
    """
    def __init__(self, 
                 model_path) -> None:
        self.model = load_pKa_acidic_model(model_path=model_path)

    def eval(self):
        self.model.eval()

    def predict(self, bg):
        """
        Makes predictions using the loaded model.

        Args:
            bg (DGLGraph): The input molecular graph.

        Returns:
            prediction (float): The predicted pKa value.
        """
        prediction, _ = self.model(bg, bg.ndata['h'], bg.edata['e'])

        return prediction.item()


class PKaBasicModel:
    """
    A wrapper class for the pKa basic prediction model.

    This class provides methods for loading and using the pre-trained pKa basic prediction model.

    Attributes:
        model (nn.Module): The pre-trained pKa basic prediction model.

    Methods:
        __init__(): Initializes the PKaBasicModel object.
        eval(): Puts the model in evaluation mode.
        predict(): Makes predictions using the loaded model.
    """
    def __init__(self, 
                 model_path) -> None:
        self.model = load_pKa_basic_model(model_path=model_path)

    def eval(self):
        self.model.eval()

    def predict(self, bg):
        """
        Makes predictions using the loaded model.

        Args:
            bg (DGLGraph): The input molecular graph.

        Returns:
            prediction (float): The predicted pKa value.
        """
        prediction, _ = self.model(bg, bg.ndata['h'], bg.edata['e'])

        return prediction.item()


class LogPModel:
    """
    A wrapper class for the logP prediction model.

    This class provides methods for loading and using the pre-trained logP prediction model.

    Attributes:
        model (nn.Module): The pre-trained logP prediction model.

    Methods:
        __init__(): Initializes the LogPModel object.
        eval(): Puts the model in evaluation mode.
        predict(): Makes predictions using the loaded model.
    """
    def __init__(self, 
                 model_path) -> None:
        self.model = load_logP_model(model_path=model_path)

    def eval(self):
        self.model.eval()

    def predict(self, bg):
        """
        Makes predictions using the loaded model.

        Args:
            bg (DGLGraph): The input molecular graph.

        Returns:
            prediction (float): The predicted logP value.
        """
        prediction = self.model(bg, bg.ndata['h'])

        return prediction.item()


def load_pKa_acidic_model(model_path):
    """
    Load the pre-trained pKa acidic prediction model.

    Args:
        model_path (str): The path to the pre-trained model file.

    Returns:
        pka_model (Pka_acidic_view): The loaded pKa acidic prediction model.
    """
    pka_model = Pka_acidic_view(
        node_feat_size = 74,
        edge_feat_size = 12,
        output_size = 1,
        num_layers= 6,
        graph_feat_size=200,
        dropout=0).to('cpu')
    
    pka_model.load_state_dict(torch.load(model_path,map_location='cpu'))
    
    return pka_model


def load_pKa_basic_model(model_path):
    """
    Load the pre-trained pKa basic prediction model.

    Args:
        model_path (str): The path to the pre-trained model file.

    Returns:
        pka2_model (Pka_basic_view): The loaded pKa basic prediction model.
    """
    pka_model = Pka_basic_view(
        node_feat_size = 74,
        edge_feat_size = 12,
        output_size = 1,
        num_layers= 6,
        graph_feat_size=200,
        dropout=0).to('cpu')
    
    pka_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    return pka_model

def load_logP_model(model_path):
    """
    Load the pre-trained logP prediction model.

    Args:
        model_path (str): The path to the pre-trained model file.

    Returns:
        logP_model (GCNPredictor): The loaded logP prediction model.
    """
    dropout = 0.0
    num_gnn_layers = 2
    
    logP_model = GCNPredictor(in_feats=74,
                        hidden_feats=[128] * num_gnn_layers,
                        activation=[F.relu] * num_gnn_layers,
                        residual=[True] * num_gnn_layers,
                        batchnorm=[False] * num_gnn_layers,
                        dropout=[dropout] * num_gnn_layers,
                        predictor_hidden_feats=16,
                        predictor_dropout=dropout,
                        n_tasks=1).to('cpu')

    logP_model.load_state_dict(torch.load(model_path, map_location='cpu'))

    return logP_model
