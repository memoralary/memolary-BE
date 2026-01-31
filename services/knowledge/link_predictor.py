"""
Knowledge Graph Link Prediction with PyTorch Geometric

Django DB의 지식 그래프를 처리하고, 사용자가 다음에 학습할 노드를 예측합니다.

사용법:
    from services.knowledge.link_predictor import LinkPredictor, GraphDataLoader
    
    # 데이터 로드
    loader = GraphDataLoader()
    data = loader.load_from_db()
    
    # 모델 학습
    predictor = LinkPredictor(embedding_dim=384)
    predictor.train(data)
    
    # 다음 학습 노드 예측
    next_nodes = predictor.predict_next_nodes(current_node_id, top_k=5)
"""

import logging
import pickle
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import negative_sampling

logger = logging.getLogger(__name__)


# =============================================================================
# Graph Data Loader
# =============================================================================

@dataclass
class GraphData:
    """그래프 데이터 컨테이너"""
    x: torch.Tensor                    # 노드 특성 (N, D)
    edge_index: torch.Tensor           # 엣지 인덱스 (2, E)
    edge_attr: Optional[torch.Tensor]  # 엣지 특성 (E, F)
    node_ids: List[str]                # 노드 ID 리스트 (UUID)
    node_titles: List[str]             # 노드 제목 리스트
    id_to_idx: Dict[str, int]          # ID -> 인덱스 매핑
    
    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]
    
    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]
    
    @property
    def embedding_dim(self) -> int:
        return self.x.shape[1]
    
    def to_pyg_data(self) -> Data:
        """PyTorch Geometric Data 객체로 변환"""
        return Data(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=self.num_nodes
        )


class GraphDataLoader:
    """
    Django DB에서 그래프 데이터를 로드합니다.
    """
    
    def __init__(self, default_embedding_dim: int = 384):
        self.default_embedding_dim = default_embedding_dim
    
    def load_from_db(self) -> GraphData:
        """
        Django DB에서 노드와 엣지를 로드하여 GraphData 생성
        
        UUID 변환 오류가 있는 데이터는 무시하고 안전하게 로드합니다.
        """
        from django.db import connection
        
        # Raw SQL로 안전하게 노드 로드 (UUID 변환 오류 방지)
        nodes_data = []
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, title, description, embedding, cluster_id 
                    FROM knowledge_knowledgenode
                """)
                columns = [col[0] for col in cursor.description]
                for row in cursor.fetchall():
                    node_dict = dict(zip(columns, row))
                    nodes_data.append(node_dict)
        except Exception as e:
            logger.error(f"노드 로드 실패: {e}")
            return self._create_empty_graph()
        
        if not nodes_data:
            logger.warning("DB에 노드가 없습니다.")
            return self._create_empty_graph()
        
        # 유효한 노드만 필터링
        valid_nodes = []
        for node in nodes_data:
            try:
                node_id = str(node['id']) if node['id'] else None
                if node_id and node.get('title'):
                    valid_nodes.append({
                        'id': node_id,
                        'title': node['title'],
                        'description': node.get('description', ''),
                        'embedding': node.get('embedding'),
                    })
            except Exception as e:
                logger.warning(f"노드 파싱 스킵 (잘못된 형식): {e}")
                continue
        
        if len(valid_nodes) < 2:
            logger.warning(f"유효한 노드가 부족합니다 ({len(valid_nodes)}개). 최소 2개 필요.")
            return self._create_empty_graph()
        
        logger.info(f"유효한 노드 {len(valid_nodes)}개 로드됨")
        
        # ID -> 인덱스 매핑
        node_ids = [n['id'] for n in valid_nodes]
        node_titles = [n['title'] for n in valid_nodes]
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        
        # 노드 임베딩 추출
        import pickle
        embeddings = []
        for node in valid_nodes:
            emb_data = node.get('embedding')
            emb = None
            
            if emb_data:
                try:
                    if isinstance(emb_data, bytes):
                        emb = pickle.loads(emb_data)
                    elif isinstance(emb_data, memoryview):
                        emb = pickle.loads(bytes(emb_data))
                except Exception as e:
                    logger.debug(f"임베딩 로드 실패: {e}")
            
            if emb is not None:
                embeddings.append(emb)
            else:
                # 임베딩이 없으면 랜덤 초기화
                embeddings.append(
                    np.random.randn(self.default_embedding_dim).astype(np.float32)
                )
        
        x = torch.tensor(np.stack(embeddings), dtype=torch.float32)
        
        # Raw SQL로 엣지 로드
        edges_data = []
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT source_id, target_id, confidence 
                    FROM knowledge_knowledgeedge
                """)
                for row in cursor.fetchall():
                    edges_data.append({
                        'source_id': str(row[0]) if row[0] else None,
                        'target_id': str(row[1]) if row[1] else None,
                        'confidence': float(row[2]) if row[2] else 1.0,
                    })
        except Exception as e:
            logger.warning(f"엣지 로드 실패: {e}")
            edges_data = []
        
        if not edges_data:
            logger.warning("DB에 엣지가 없습니다. 학습에는 최소 2개의 엣지가 필요합니다.")
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = None
        else:
            source_indices = []
            target_indices = []
            confidences = []
            
            for edge in edges_data:
                src_id = edge.get('source_id')
                tgt_id = edge.get('target_id')
                
                if src_id and tgt_id and src_id in id_to_idx and tgt_id in id_to_idx:
                    source_indices.append(id_to_idx[src_id])
                    target_indices.append(id_to_idx[tgt_id])
                    confidences.append(edge.get('confidence', 1.0))
            
            if not source_indices:
                logger.warning("유효한 엣지가 없습니다.")
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = None
            else:
                edge_index = torch.tensor(
                    [source_indices, target_indices],
                    dtype=torch.long
                )
                edge_attr = torch.tensor(confidences, dtype=torch.float32).unsqueeze(1)
        
        logger.info(f"그래프 로드 완료: {len(valid_nodes)}개 노드, {edge_index.shape[1]}개 엣지")
        
        # 학습 가능 여부 확인
        if edge_index.shape[1] < 2:
            logger.warning("⚠️  엣지가 2개 미만이면 Link Prediction 학습이 어렵습니다.")
        
        return GraphData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_ids=node_ids,
            node_titles=node_titles,
            id_to_idx=id_to_idx
        )
    
    def _create_empty_graph(self) -> GraphData:
        """빈 그래프 생성"""
        return GraphData(
            x=torch.zeros((0, self.default_embedding_dim), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=None,
            node_ids=[],
            node_titles=[],
            id_to_idx={}
        )
    
    @staticmethod
    def create_sample_graph(
        num_nodes: int = 5,
        embedding_dim: int = 64,
        edge_probability: float = 0.4
    ) -> GraphData:
        """
        테스트용 샘플 그래프 생성
        
        Args:
            num_nodes: 노드 수
            embedding_dim: 임베딩 차원
            edge_probability: 엣지 생성 확률
            
        Returns:
            샘플 GraphData
        """
        # 랜덤 노드 임베딩
        x = torch.randn(num_nodes, embedding_dim)
        
        # 랜덤 엣지 생성 (Erdős-Rényi)
        source_indices = []
        target_indices = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and np.random.random() < edge_probability:
                    source_indices.append(i)
                    target_indices.append(j)
        
        edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
        
        # 노드 메타데이터
        node_ids = [f"node_{i}" for i in range(num_nodes)]
        node_titles = [f"Concept {i}" for i in range(num_nodes)]
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        
        return GraphData(
            x=x,
            edge_index=edge_index,
            edge_attr=None,
            node_ids=node_ids,
            node_titles=node_titles,
            id_to_idx=id_to_idx
        )


# =============================================================================
# GNN Encoder Models
# =============================================================================

class GCNEncoder(nn.Module):
    """Graph Convolutional Network 인코더"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class SAGEEncoder(nn.Module):
    """GraphSAGE 인코더"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# =============================================================================
# Link Prediction Decoder
# =============================================================================

class LinkDecoder(nn.Module):
    """
    링크 예측 디코더
    
    두 노드 임베딩을 받아 링크 존재 확률을 예측합니다.
    """
    
    def __init__(self, hidden_channels: int, method: str = "dot"):
        super().__init__()
        self.method = method
        
        if method == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_channels, 1)
            )
    
    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z: 노드 임베딩 (N, D)
            edge_index: 예측할 엣지 (2, E)
            
        Returns:
            링크 확률 (E,)
        """
        src = z[edge_index[0]]
        tgt = z[edge_index[1]]
        
        if self.method == "dot":
            # 내적 기반
            return (src * tgt).sum(dim=1)
        elif self.method == "cosine":
            # 코사인 유사도
            return F.cosine_similarity(src, tgt)
        elif self.method == "mlp":
            # MLP 기반
            return self.mlp(torch.cat([src, tgt], dim=1)).squeeze()
        else:
            raise ValueError(f"Unknown method: {self.method}")


# =============================================================================
# Link Prediction Model
# =============================================================================

class LinkPredictionModel(nn.Module):
    """
    Link Prediction 전체 모델
    
    GNN 인코더 + 링크 디코더로 구성됩니다.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        encoder_type: str = "sage",
        decoder_type: str = "dot",
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # 인코더 선택
        if encoder_type == "gcn":
            self.encoder = GCNEncoder(
                in_channels, hidden_channels, out_channels, num_layers, dropout
            )
        elif encoder_type == "sage":
            self.encoder = SAGEEncoder(
                in_channels, hidden_channels, out_channels, num_layers, dropout
            )
        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")
        
        # 디코더
        self.decoder = LinkDecoder(out_channels, decoder_type)
        
        self.out_channels = out_channels
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """노드 임베딩 생성"""
        return self.encoder(x, edge_index)
    
    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """링크 확률 예측"""
        return self.decoder(z, edge_index)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파
        
        Returns:
            (positive_scores, negative_scores)
        """
        z = self.encode(x, edge_index)
        pos_scores = self.decode(z, pos_edge_index)
        neg_scores = self.decode(z, neg_edge_index)
        return pos_scores, neg_scores


# =============================================================================
# Link Predictor (High-level API)
# =============================================================================

class LinkPredictor:
    """
    링크 예측 고수준 API
    
    Django DB의 지식 그래프를 학습하고, 다음 학습 노드를 예측합니다.
    
    Example:
        predictor = LinkPredictor(embedding_dim=384)
        predictor.train(graph_data, epochs=100)
        
        next_nodes = predictor.predict_next_nodes("node_id", top_k=5)
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_channels: int = 128,
        out_channels: int = 64,
        encoder_type: str = "sage",
        decoder_type: str = "dot",
        device: Optional[str] = None
    ):
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model: Optional[LinkPredictionModel] = None
        self.graph_data: Optional[GraphData] = None
        self.node_embeddings: Optional[torch.Tensor] = None
    
    def _build_model(self, in_channels: int) -> LinkPredictionModel:
        """모델 생성"""
        return LinkPredictionModel(
            in_channels=in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            encoder_type=self.encoder_type,
            decoder_type=self.decoder_type
        ).to(self.device)
    
    def train(
        self,
        graph_data: GraphData,
        epochs: int = 100,
        lr: float = 0.01,
        weight_decay: float = 1e-5,
        val_ratio: float = 0.1,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        모델 학습
        
        Args:
            graph_data: 학습 데이터
            epochs: 에포크 수
            lr: 학습률
            weight_decay: L2 정규화
            val_ratio: 검증 데이터 비율
            verbose: 로그 출력 여부
            
        Returns:
            학습 히스토리 {"train_loss": [...], "val_auc": [...]}
        """
        self.graph_data = graph_data
        
        if graph_data.num_edges == 0:
            logger.warning("엣지가 없어 학습할 수 없습니다.")
            return {"train_loss": [], "val_auc": []}
        
        # 모델 생성
        self.model = self._build_model(graph_data.embedding_dim)
        
        # 데이터를 디바이스로 이동
        x = graph_data.x.to(self.device)
        edge_index = graph_data.edge_index.to(self.device)
        
        # Train/Val 분할
        num_edges = edge_index.shape[1]
        num_val = max(1, int(num_edges * val_ratio))
        
        perm = torch.randperm(num_edges)
        val_mask = perm[:num_val]
        train_mask = perm[num_val:]
        
        train_edge_index = edge_index[:, train_mask]
        val_edge_index = edge_index[:, val_mask]
        
        # 옵티마이저
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        history = {"train_loss": [], "val_auc": []}
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            # Negative sampling
            neg_edge_index = negative_sampling(
                edge_index=train_edge_index,
                num_nodes=graph_data.num_nodes,
                num_neg_samples=train_edge_index.shape[1]
            )
            
            # Forward
            pos_scores, neg_scores = self.model(
                x, train_edge_index, train_edge_index, neg_edge_index
            )
            
            # Binary cross-entropy loss
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )
            loss = pos_loss + neg_loss
            
            loss.backward()
            optimizer.step()
            
            history["train_loss"].append(loss.item())
            
            # 검증
            if (epoch + 1) % 10 == 0 and verbose:
                val_auc = self._evaluate(x, train_edge_index, val_edge_index)
                history["val_auc"].append(val_auc)
                logger.info(f"Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, Val_AUC={val_auc:.4f}")
        
        # 최종 노드 임베딩 저장
        self.model.eval()
        with torch.no_grad():
            self.node_embeddings = self.model.encode(x, edge_index)
        
        return history
    
    def _evaluate(
        self,
        x: torch.Tensor,
        train_edge_index: torch.Tensor,
        val_edge_index: torch.Tensor
    ) -> float:
        """검증 AUC 계산"""
        from sklearn.metrics import roc_auc_score
        
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(x, train_edge_index)
            
            # Positive scores
            pos_scores = self.model.decode(z, val_edge_index)
            
            # Negative sampling for validation
            neg_edge_index = negative_sampling(
                edge_index=train_edge_index,
                num_nodes=x.shape[0],
                num_neg_samples=val_edge_index.shape[1]
            )
            neg_scores = self.model.decode(z, neg_edge_index)
            
            # AUC
            y_true = torch.cat([
                torch.ones(pos_scores.shape[0]),
                torch.zeros(neg_scores.shape[0])
            ]).cpu().numpy()
            
            y_score = torch.cat([pos_scores, neg_scores]).sigmoid().cpu().numpy()
            
            try:
                return roc_auc_score(y_true, y_score)
            except:
                return 0.5
    
    def predict_next_nodes(
        self,
        current_node_id: str,
        top_k: int = 5,
        exclude_learned: Optional[List[str]] = None
    ) -> List[Tuple[str, str, float]]:
        """
        현재 노드에서 다음에 학습할 노드 예측
        
        Args:
            current_node_id: 현재 학습한 노드 ID
            top_k: 상위 K개 반환
            exclude_learned: 이미 학습한 노드 ID 리스트
            
        Returns:
            [(node_id, node_title, score), ...] 점수 내림차순
        """
        if self.model is None or self.graph_data is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        if current_node_id not in self.graph_data.id_to_idx:
            raise ValueError(f"노드 ID를 찾을 수 없습니다: {current_node_id}")
        
        exclude_learned = exclude_learned or []
        current_idx = self.graph_data.id_to_idx[current_node_id]
        
        # 모든 노드에 대한 링크 점수 계산
        self.model.eval()
        with torch.no_grad():
            z = self.node_embeddings
            
            # 현재 노드에서 모든 노드로의 잠재적 엣지
            candidate_indices = [
                i for i in range(self.graph_data.num_nodes)
                if i != current_idx and 
                   self.graph_data.node_ids[i] not in exclude_learned
            ]
            
            if not candidate_indices:
                return []
            
            # 엣지 인덱스 생성
            source_indices = [current_idx] * len(candidate_indices)
            target_indices = candidate_indices
            
            pred_edge_index = torch.tensor(
                [source_indices, target_indices],
                dtype=torch.long
            ).to(self.device)
            
            # 점수 예측
            scores = self.model.decode(z, pred_edge_index).sigmoid().cpu().numpy()
        
        # 상위 K개 선택
        results = []
        for idx, score in zip(candidate_indices, scores):
            results.append((
                self.graph_data.node_ids[idx],
                self.graph_data.node_titles[idx],
                float(score)
            ))
        
        # 점수 내림차순 정렬
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:top_k]
    
    def save(self, path: str) -> None:
        """모델 저장"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": {
                "embedding_dim": self.embedding_dim,
                "hidden_channels": self.hidden_channels,
                "out_channels": self.out_channels,
                "encoder_type": self.encoder_type,
                "decoder_type": self.decoder_type,
            }
        }, path)
        logger.info(f"모델 저장됨: {path}")
    
    def load(self, path: str, graph_data: GraphData) -> None:
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.graph_data = graph_data
        self.model = self._build_model(graph_data.embedding_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # 노드 임베딩 계산
        x = graph_data.x.to(self.device)
        edge_index = graph_data.edge_index.to(self.device)
        
        with torch.no_grad():
            self.node_embeddings = self.model.encode(x, edge_index)
        
        logger.info(f"모델 로드됨: {path}")


# =============================================================================
# Sanity Check
# =============================================================================

def run_sanity_check():
    """
    작은 샘플 그래프에서 Loss가 줄어드는지 확인하는 Sanity Check
    """
    print("=" * 60)
    print("🧪 Link Prediction Sanity Check")
    print("=" * 60)
    
    # 샘플 그래프 생성 (5개 노드)
    print("\n📊 샘플 그래프 생성 (5개 노드)...")
    graph_data = GraphDataLoader.create_sample_graph(
        num_nodes=5,
        embedding_dim=64,
        edge_probability=0.5
    )
    print(f"   노드 수: {graph_data.num_nodes}")
    print(f"   엣지 수: {graph_data.num_edges}")
    print(f"   임베딩 차원: {graph_data.embedding_dim}")
    
    if graph_data.num_edges < 2:
        print("⚠️  엣지가 너무 적습니다. 다시 생성합니다.")
        graph_data = GraphDataLoader.create_sample_graph(
            num_nodes=5,
            embedding_dim=64,
            edge_probability=0.7
        )
    
    # 모델 생성
    print("\n🔧 모델 생성...")
    predictor = LinkPredictor(
        embedding_dim=graph_data.embedding_dim,
        hidden_channels=32,
        out_channels=16,
        encoder_type="sage"
    )
    
    # 학습
    print("\n🎓 학습 시작 (50 에포크)...")
    history = predictor.train(
        graph_data,
        epochs=50,
        lr=0.01,
        verbose=False
    )
    
    # Loss 감소 확인
    initial_loss = history["train_loss"][0]
    final_loss = history["train_loss"][-1]
    loss_reduced = final_loss < initial_loss
    
    print(f"\n📈 학습 결과:")
    print(f"   초기 Loss: {initial_loss:.4f}")
    print(f"   최종 Loss: {final_loss:.4f}")
    print(f"   Loss 감소: {'✅ YES' if loss_reduced else '❌ NO'} ({initial_loss - final_loss:.4f})")
    
    # 예측 테스트
    print("\n🔮 다음 학습 노드 예측 테스트:")
    current_node = graph_data.node_ids[0]
    predictions = predictor.predict_next_nodes(current_node, top_k=3)
    
    for node_id, title, score in predictions:
        print(f"   → {title}: {score:.3f}")
    
    # 결과 판정
    print("\n" + "=" * 60)
    if loss_reduced and predictions:
        print("🎉 Sanity Check PASSED!")
    else:
        print("❌ Sanity Check FAILED")
    print("=" * 60)
    
    return loss_reduced


if __name__ == "__main__":
    run_sanity_check()
