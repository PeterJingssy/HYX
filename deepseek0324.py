import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, degree
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC
import torch.optim as optim
from sklearn.metrics import r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ---------- H100 优化设置 ----------
torch.set_float32_matmul_precision('high')   # 启用 TF32
torch.backends.cudnn.benchmark = True        # cuDNN 自动调优

# ---------- 模块1：离散Ricci曲率计算（静态，修正版） ----------
class DiscreteRicciCurvature:
    """计算图的Ollivier-Ricci曲率（静态版本，使用正确的Sinkhorn算法）"""
    def __init__(self, alpha=0.5, lambda_reg=0.1, sinkhorn_iters=20):
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.sinkhorn_iters = sinkhorn_iters

    def compute_node_distribution(self, G, node):
        """计算节点的概率分布μ_i（基于拓扑，均匀分布）"""
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 0:
            return {node: 1.0}
        dist = {node: self.alpha}
        for nb in neighbors:
            dist[nb] = (1 - self.alpha) / len(neighbors)
        return dist

    def wasserstein_distance(self, dist_u, dist_v, G):
        """使用Sinkhorn算法计算两个分布之间的1-Wasserstein距离"""
        nodes_u = list(dist_u.keys())
        nodes_v = list(dist_v.keys())
        m, n = len(nodes_u), len(nodes_v)
        # 代价矩阵：最短路径长度
        C = np.zeros((m, n))
        for i, ni in enumerate(nodes_u):
            for j, nj in enumerate(nodes_v):
                try:
                    C[i, j] = nx.shortest_path_length(G, ni, nj)
                except nx.NetworkXNoPath:
                    C[i, j] = 1e6
        p = np.array([dist_u[ni] for ni in nodes_u])
        q = np.array([dist_v[nj] for nj in nodes_v])
        # Sinkhorn 迭代
        K = np.exp(-self.lambda_reg * C)
        K = np.maximum(K, 1e-12)
        u = np.ones(m) / m
        v = np.ones(n) / n
        for _ in range(self.sinkhorn_iters):
            u = p / (K @ v + 1e-12)
            v = q / (K.T @ u + 1e-12)
        P = u[:, None] * K * v[None, :]
        wass_dist = np.sum(P * C)
        return wass_dist

    def compute_edge_curvatures(self, G):
        """计算所有边的曲率"""
        edge_curvatures = {}
        for u, v in G.edges():
            dist_u = self.compute_node_distribution(G, u)
            dist_v = self.compute_node_distribution(G, v)
            wass_dist = self.wasserstein_distance(dist_u, dist_v, G)
            edge_dist = 1.0
            curvature = 1 - wass_dist / edge_dist
            edge_curvatures[(u, v)] = curvature
        return edge_curvatures


# ---------- 模块2：曲率编码器 ----------
class CurvatureEncoder(nn.Module):
    """将曲率值编码为可学习的嵌入向量"""
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, curvature_values):
        return self.mlp(curvature_values)


# ---------- 模块3：边特征编码器 ----------
class EdgeEncoder(nn.Module):
    """将化学键特征编码为嵌入向量"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Linear(in_channels, out_channels)

    def forward(self, edge_attr):
        return self.encoder(edge_attr)


# ---------- 模块4：曲率增强的Transformer层 ----------
class CurvphormerLayer(nn.Module):
    """图Transformer层，使用曲率嵌入作为注意力偏置"""
    def __init__(self, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.curvature_bias_proj = nn.Linear(hidden_dim, num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, curvature_embeddings):
        """x: [N, D], edge_index: [2, E], curvature_embeddings: [E, D]"""
        x_norm = self.norm1(x)
        N, D = x.shape
        E = edge_index.size(1)

        # 多头投影
        q = self.q_proj(x_norm).reshape(N, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).reshape(N, self.num_heads, self.head_dim)
        v = self.v_proj(x_norm).reshape(N, self.num_heads, self.head_dim)

        src = edge_index[0]
        tgt = edge_index[1]

        q_src = q[src]                     # [E, H, D_head]
        k_tgt = k[tgt]                     # [E, H, D_head]
        attn_scores = (q_src * k_tgt).sum(dim=-1) / (self.head_dim ** 0.5)   # [E, H]

        # 曲率偏置
        bias = self.curvature_bias_proj(curvature_embeddings)  # [E, H]
        attn_scores = attn_scores + bias

        # 按目标节点 softmax
        attn_probs = torch.zeros_like(attn_scores)
        for node in torch.unique(tgt):
            mask = tgt == node
            attn_probs[mask] = F.softmax(attn_scores[mask], dim=0)

        # 聚合消息
        v_tgt = v[tgt]                     # [E, H, D_head]
        attn_probs_exp = attn_probs.unsqueeze(-1)  # [E, H, 1]
        messages = attn_probs_exp * v_tgt          # [E, H, D_head]
        messages = messages.reshape(E, -1)         # [E, D]

        out = torch.zeros_like(x)
        out.index_add_(0, src, messages)
        out = self.out_proj(out)

        x = x + self.dropout(out)
        x_norm2 = self.norm2(x)
        x = x + self.dropout(self.ffn(x_norm2))
        return x


# ---------- 辅助函数：邻居列表（用于动态曲率） ----------
def build_neighbor_lists(edge_index, num_nodes, include_self=True):
    """返回每个节点的邻居列表（tensor）和填充后的批量数据"""
    device = edge_index.device
    # 构建邻居集合
    neighbor_sets = [set() for _ in range(num_nodes)]
    src = edge_index[0].cpu().numpy()
    tgt = edge_index[1].cpu().numpy()
    for u, v in zip(src, tgt):
        neighbor_sets[u].add(v)
        neighbor_sets[v].add(u)
    if include_self:
        for i in range(num_nodes):
            neighbor_sets[i].add(i)
    # 转换为排序列表
    neighbor_lists = [sorted(list(s)) for s in neighbor_sets]
    # 转为 tensor
    neighbor_tensors = [torch.tensor(lst, dtype=torch.long, device=device) for lst in neighbor_lists]
    return neighbor_tensors


def batch_sinkhorn(mu_list, nu_list, C_list, lambda_reg=0.1, num_iters=20, eps=1e-12):
    """
    批量 Sinkhorn 计算 Wasserstein 距离。
    mu_list: list of [m_i] 源分布
    nu_list: list of [n_i] 目标分布
    C_list: list of [m_i, n_i] 代价矩阵
    返回: wass_dist_list: list of scalar tensors
    """
    device = mu_list[0].device
    batch_size = len(mu_list)
    # 找出最大尺寸
    max_m = max(len(mu) for mu in mu_list)
    max_n = max(len(nu) for nu in nu_list)

    # 填充为 [batch, max_m], [batch, max_n], [batch, max_m, max_n]
    mu_pad = torch.zeros(batch_size, max_m, device=device)
    nu_pad = torch.zeros(batch_size, max_n, device=device)
    C_pad = torch.full((batch_size, max_m, max_n), 1e6, device=device)  # 大数填充
    mask_m = torch.zeros(batch_size, max_m, dtype=torch.bool, device=device)
    mask_n = torch.zeros(batch_size, max_n, dtype=torch.bool, device=device)

    for i in range(batch_size):
        m_i = len(mu_list[i])
        n_i = len(nu_list[i])
        mu_pad[i, :m_i] = mu_list[i]
        nu_pad[i, :n_i] = nu_list[i]
        C_pad[i, :m_i, :n_i] = C_list[i]
        mask_m[i, :m_i] = True
        mask_n[i, :n_i] = True

    # 核矩阵
    K = torch.exp(-lambda_reg * C_pad)          # [B, M, N]
    K = torch.clamp(K, min=eps)

    # 初始化 u, v
    u = torch.ones(batch_size, max_m, device=device) / max_m
    v = torch.ones(batch_size, max_n, device=device) / max_n

    for _ in range(num_iters):
        # u = mu / (K v)
        Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1)   # [B, M]
        Kv = torch.where(mask_m, Kv, torch.ones_like(Kv))
        u = mu_pad / (Kv + eps)
        u = torch.where(mask_m, u, torch.ones_like(u))

        # v = nu / (K^T u)
        KTu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)  # [B, N]
        KTu = torch.where(mask_n, KTu, torch.ones_like(KTu))
        v = nu_pad / (KTu + eps)
        v = torch.where(mask_n, v, torch.ones_like(v))

    # 传输计划
    P = u.unsqueeze(-1) * K * v.unsqueeze(1)     # [B, M, N]
    # 只计算有效位置的代价
    wass_dist = (P * C_pad).sum(dim=(1, 2))       # [B]
    return wass_dist


# ---------- 模块5：完整的 DCurvphormer-Mol 模型 ----------
class Curvphormer(nn.Module):
    """支持静态/动态曲率，边特征，H100 优化版"""
    def __init__(self,
                 in_channels,
                 hidden_dim=64,
                 out_channels=1,
                 num_layers=6,
                 num_heads=4,
                 dropout=0.1,
                 use_dynamic_curvature=False,
                 edge_in_channels=None,
                 beta=1.0,
                 lambda_reg=0.1,
                 sinkhorn_iters=20,
                 use_sinkhorn=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_dynamic_curvature = use_dynamic_curvature
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.sinkhorn_iters = sinkhorn_iters
        self.use_sinkhorn = use_sinkhorn

        self.node_encoder = nn.Linear(in_channels, hidden_dim)
        self.curvature_encoder = CurvatureEncoder(1, hidden_dim)

        if edge_in_channels is not None:
            self.edge_encoder = EdgeEncoder(edge_in_channels, hidden_dim)
        else:
            self.edge_encoder = None

        self.layers = nn.ModuleList([
            CurvphormerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_channels)
        )

    def compute_static_curvatures(self, data):
        """静态曲率（用于非动态模式）"""
        G = to_networkx(data, to_undirected=True)
        curvature_calc = DiscreteRicciCurvature(alpha=0.5,
                                                lambda_reg=self.lambda_reg,
                                                sinkhorn_iters=self.sinkhorn_iters)
        edge_curvatures_dict = curvature_calc.compute_edge_curvatures(G)
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        curvature_values = torch.zeros(num_edges, 1, device=edge_index.device)
        for i in range(num_edges):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            if (u, v) in edge_curvatures_dict:
                curvature_values[i] = edge_curvatures_dict[(u, v)]
            else:
                curvature_values[i] = edge_curvatures_dict.get((v, u), 0.0)
        return curvature_values

    def compute_dynamic_curvatures(self, h, edge_index, neighbor_lists):
        """
        批量并行计算动态曲率（H100 优化版）
        h: [N, D]
        edge_index: [2, E]
        neighbor_lists: list of tensors，每个元素是节点的邻居索引（包括自身）
        """
        device = h.device
        N, D = h.shape
        E = edge_index.size(1)

        # 收集每条边的源节点、目标节点
        src_nodes = edge_index[0]   # [E]
        tgt_nodes = edge_index[1]   # [E]

        # 为每条边准备邻居特征和掩码
        # 找出最大邻居数
        max_deg = max(len(neighbor_lists[i]) for i in torch.cat([src_nodes, tgt_nodes]))
        # 填充邻居特征: [E, max_deg, D]
        src_neigh_feats = torch.zeros(E, max_deg, D, device=device)
        tgt_neigh_feats = torch.zeros(E, max_deg, D, device=device)
        src_mask = torch.zeros(E, max_deg, dtype=torch.bool, device=device)
        tgt_mask = torch.zeros(E, max_deg, dtype=torch.bool, device=device)
        src_neigh_sizes = torch.zeros(E, dtype=torch.long, device=device)
        tgt_neigh_sizes = torch.zeros(E, dtype=torch.long, device=device)

        for e in range(E):
            u = src_nodes[e].item()
            v = tgt_nodes[e].item()
            u_neigh = neighbor_lists[u]    # tensor
            v_neigh = neighbor_lists[v]
            du = len(u_neigh)
            dv = len(v_neigh)
            src_neigh_sizes[e] = du
            tgt_neigh_sizes[e] = dv
            src_neigh_feats[e, :du] = h[u_neigh]
            tgt_neigh_feats[e, :dv] = h[v_neigh]
            src_mask[e, :du] = True
            tgt_mask[e, :dv] = True

        # 计算分布 mu_i 和 mu_j
        # mu_i: [E, du] 基于当前节点特征与邻居特征的相似度
        # 获取当前节点特征
        h_src = h[src_nodes]      # [E, D]
        h_tgt = h[tgt_nodes]      # [E, D]

        # 计算相似度矩阵: [E, max_deg]  (只对有效位置)
        sim_src = torch.bmm(h_src.unsqueeze(1), src_neigh_feats.transpose(1, 2)).squeeze(1)   # [E, max_deg]
        sim_tgt = torch.bmm(h_tgt.unsqueeze(1), tgt_neigh_feats.transpose(1, 2)).squeeze(1)   # [E, max_deg]

        # 屏蔽无效位置（填充部分设为 -inf）
        sim_src = torch.where(src_mask, sim_src, torch.tensor(-float('inf'), device=device))
        sim_tgt = torch.where(tgt_mask, sim_tgt, torch.tensor(-float('inf'), device=device))

        mu_src = F.softmax(self.beta * sim_src, dim=1)  # [E, max_deg]
        mu_tgt = F.softmax(self.beta * sim_tgt, dim=1)  # [E, max_deg]

        # 代价矩阵 C: [E, max_deg, max_deg]  特征欧氏距离
        # 计算每对邻居间的距离（只对有效位置）
        # 使用广播: (E,1,D) - (E,D,1) 不好直接，采用 torch.cdist 批量计算
        # 但我们有填充，需要掩码
        # 更高效：对每个边独立计算距离，因为分子图度数很小，循环可能更快？但为了并行，我们尝试用 torch.cdist
        # 由于最大度数很小（通常 ≤10），我们可以直接计算所有边的全距离，然后用掩码置零
        # 但为简单，此处仍用循环，因为度数小且边数不大（分子图 ≤ 500 边），循环开销可接受。
        # 若边数很大可优化，但分子图场景下循环足够。
        # 但为了体现批量优化，我们仍用批量方式：
        # 计算差矩阵 [E, max_deg, max_deg, D]
        # 实际上 torch.cdist 可以批量计算：输入 [E, max_deg, D] 和 [E, max_deg, D]
        C = torch.cdist(src_neigh_feats, tgt_neigh_feats, p=2)   # [E, max_deg, max_deg]
        # 将无效位置设为极大值（因为会参与 softmax，但实际会被 mask 忽略）
        C = torch.where(src_mask.unsqueeze(2) & tgt_mask.unsqueeze(1), C, torch.tensor(1e6, device=device))

        # 准备批量 Sinkhorn
        mu_list = [mu_src[e, :src_neigh_sizes[e]] for e in range(E)]
        nu_list = [mu_tgt[e, :tgt_neigh_sizes[e]] for e in range(E)]
        C_list = [C[e, :src_neigh_sizes[e], :tgt_neigh_sizes[e]] for e in range(E)]

        if self.use_sinkhorn:
            wass_dists = batch_sinkhorn(mu_list, nu_list, C_list,
                                        lambda_reg=self.lambda_reg,
                                        num_iters=self.sinkhorn_iters)
        else:
            # 消融：简单近似（其实应该用精确的EMD，但为了消融保留）
            # 这里用 L1 距离近似（不准确，仅消融用）
            wass_dists = torch.zeros(E, device=device)
            for e in range(E):
                m = mu_list[e]
                n = nu_list[e]
                C_e = C_list[e]
                # 简单近似：平均运输距离
                wass_dists[e] = torch.sum(torch.abs(m[:, None] - n[None, :]) * C_e)

        # 边长度（动态长度）
        d_ij = torch.norm(h_src - h_tgt, p=2, dim=1)  # [E]
        d_ij = torch.clamp(d_ij, min=1e-6)

        kappa = 1 - wass_dists / d_ij   # [E]
        # 编码
        curvature_embeddings = self.curvature_encoder(kappa.unsqueeze(1))   # [E, D]
        return curvature_embeddings

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.node_encoder(x.float())  # [N, D]

        # 边特征
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if self.edge_encoder is not None and edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr.float())   # [E, D]
        else:
            edge_emb = None

        num_edges = edge_index.size(1)
        if not self.use_dynamic_curvature:
            # 静态曲率：预先计算一次
            curv_values = self.compute_static_curvatures(data).to(x.device)
            curvature_embeddings = self.curvature_encoder(curv_values)
        else:
            # 动态曲率：第一层先零初始化，每层开始前更新
            curvature_embeddings = torch.zeros(num_edges, self.hidden_dim, device=x.device)
            # 预计算邻居列表（每层相同，因为图结构不变）
            neighbor_lists = build_neighbor_lists(edge_index, x.size(0), include_self=True)

        # 逐层传播
        for i, layer in enumerate(self.layers):
            if self.use_dynamic_curvature:
                # 基于当前 h 重新计算曲率
                curvature_embeddings = self.compute_dynamic_curvatures(h, edge_index, neighbor_lists)
            # 融合边特征（静态）
            if edge_emb is not None:
                curvature_embeddings = curvature_embeddings + edge_emb
            h = layer(h, edge_index, curvature_embeddings)

        # 图读出：按 batch 平均
        num_graphs = batch.max().item() + 1
        graph_out = torch.zeros(num_graphs, self.hidden_dim, device=h.device)
        graph_out.index_add_(0, batch, h)
        counts = torch.bincount(batch).float().unsqueeze(1)
        graph_out = graph_out / counts
        out = self.output_layer(graph_out)
        return out


# ---------- 训练/评估函数（支持混合精度） ----------
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    for batch_idx, data in enumerate(loader):
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(data).squeeze()
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        all_preds.append(out.detach().cpu())
        all_targets.append(data.y.cpu())

                # 每 10 个 batch 打印一次进度（可根据需要调整频率）
        if batch_idx % 1 == 0:
            print(f'  Batch {batch_idx} done, loss: {loss.item():.4f}')

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mse = total_loss / len(loader.dataset)
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()
    r2 = r2_score(all_targets.numpy(), all_preds.numpy())
    return mse, mae, rmse, r2


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    for data in loader:
        data = data.to(device, non_blocking=True)
        out = model(data).squeeze()
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.num_graphs
        all_preds.append(out.cpu())
        all_targets.append(data.y.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mse = total_loss / len(loader.dataset)
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()
    r2 = r2_score(all_targets.numpy(), all_preds.numpy())
    return mse, mae, rmse, r2


# ---------- 主程序 ----------
if __name__ == "__main__":
    print("this_is_a_outcome_of_deepseek0324")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据集路径（请根据实际情况修改）
    root_path = './ZINC'
    train_dataset = ZINC(root=root_path, subset=True, split='train')
    val_dataset = ZINC(root=root_path, subset=True, split='val')
    test_dataset = ZINC(root=root_path, subset=True, split='test')

    # 使用多线程和固定内存加速数据加载
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                             num_workers=4, pin_memory=True)

    sample_data = train_dataset[0]
    in_channels = sample_data.x.size(1)

    # 边特征维度
    if hasattr(sample_data, 'edge_attr') and sample_data.edge_attr is not None:
        if sample_data.edge_attr.dim() == 2:
            edge_in_channels = sample_data.edge_attr.size(1)
        elif sample_data.edge_attr.dim() == 1:
            edge_in_channels = 1
        else:
            edge_in_channels = None
    else:
        edge_in_channels = None

    # ================== 配置选项 ==================
    use_dynamic = True          # True: 动态曲率, False: 静态曲率
    use_edge = True             # 是否使用化学键特征
    beta = 1                  # 温度参数 (论文公式4.1)
    use_sinkhorn = True         # 是否使用Sinkhorn
    lambda_reg = 0.05           # Sinkhorn正则化系数
    sinkhorn_iters = 10         # Sinkhorn迭代次数
    # =============================================

    model = Curvphormer(
        in_channels=in_channels,
        hidden_dim=128,
        out_channels=1,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        use_dynamic_curvature=use_dynamic,
        edge_in_channels=edge_in_channels if use_edge else None,
        beta=beta,
        lambda_reg=lambda_reg,
        sinkhorn_iters=sinkhorn_iters,
        use_sinkhorn=use_sinkhorn
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    # 混合精度缩放器
    # scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_mse = float('inf')
    print("Starting training...")
    for epoch in range(1, 51):
        train_mse, train_mae, train_rmse, train_r2 = train_epoch(
            model, train_loader, optimizer, criterion, device)
        val_mse, val_mae, val_rmse, val_r2 = eval_epoch(model, val_loader, criterion, device)

        print(f'Epoch {epoch:03d} | '
              f'Train MSE: {train_mse:.4f} MAE: {train_mae:.4f} RMSE: {train_rmse:.4f} R2: {train_r2:.4f} | '
              f'Val MSE: {val_mse:.4f} MAE: {val_mae:.4f} RMSE: {val_rmse:.4f} R2: {val_r2:.4f}')

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), 'best_model_h100.pth')
            print(f' -> Best model saved (val MSE: {val_mse:.4f})')

    # 测试
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load('best_model_h100.pth'))
    test_mse, test_mae, test_rmse, test_r2 = eval_epoch(model, test_loader, criterion, device)
    print(f'Test MSE: {test_mse:.4f} MAE: {test_mae:.4f} RMSE: {test_rmse:.4f} R2: {test_r2:.4f}')

    print("\n--- Configuration ---")
    print(f"use_dynamic_curvature: {use_dynamic}")
    print(f"use_edge_features: {use_edge}")
    print(f"beta: {beta}")
    print(f"use_sinkhorn: {use_sinkhorn}")
    print(f"lambda_reg: {lambda_reg}")
    print(f"sinkhorn_iters: {sinkhorn_iters}")
    print("this_is_a_outcome_of_ds0324")