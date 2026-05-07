import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalModalityBalance(nn.Module):
    """
    Hierarchical modality balancing for remote sensing detection.

    Addresses vision-text imbalance: remote sensing images are information-dense
    while text annotations are extremely sparse (just class names + locations).
    """

    def __init__(self, hidden_dim=768, num_classes=4):
        super().__init__()

        # Level 1: feature-level balance (dimensionality mismatch)
        self.feature_balancer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Level 2: semantic-level balance (compression + enrichment)
        # Vision compressor: retain damage-detection-relevant features
        self.vision_compressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Text enhancer: expand sparse class information
        self.text_enhancer = nn.ModuleDict({
            'damage_embeddings': nn.Embedding(num_classes, hidden_dim // 4),
            'spatial_embeddings': nn.Linear(4, hidden_dim // 4),  # bbox coordinates
            'context_projector': nn.Linear(hidden_dim // 2, hidden_dim)
        })

        # Level 3: task-level balance (detection-specific)
        self.detection_aligner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.damage_levels = [
            'destroyed building',
            'major damaged building',
            'minor damaged building',
            'undamaged building'
        ]

    def forward(self, vision_features, text_features, labels=None, attention_mask=None):
        """
        Args:
            vision_features: [B, N, D] from Florence-2 image_hidden_states
            text_features: [B, T, D] from decoder_hidden_states[-1]
            labels: [B, T] for extracting class info (optional)
            attention_mask: [B, T] decoder attention mask
        """
        B = vision_features.shape[0]
        device = vision_features.device

        # Modality imbalance estimation
        imbalance_score = self._compute_imbalance(vision_features, text_features, attention_mask)

        # Feature-level balance
        v_feat, t_feat = self._feature_level_balance(
            vision_features, text_features, attention_mask, imbalance_score
        )

        # Semantic-level balance
        v_compressed = self._compress_vision_for_damage_detection(v_feat)
        t_enriched = self._enrich_text_with_damage_context(t_feat, text_features, device)

        # Task-level alignment
        v_aligned = self.detection_aligner(v_compressed)
        t_aligned = self.detection_aligner(t_enriched)

        # Loss computation
        losses = self._compute_hierarchical_losses(
            v_aligned, t_aligned, v_feat, t_feat,
            v_compressed, imbalance_score, device
        )

        losses['total'] = sum(losses.values())

        # Force all HMB params into the graph to avoid DDP unused-parameter errors
        dummy = 0.0
        for p in self.parameters():
            dummy = dummy + p.sum() * 0.0
        losses['total'] = losses['total'] + dummy

        return losses, imbalance_score


    def _compute_imbalance(self, v_feats, t_feats, mask=None):
        """Estimate modality imbalance score."""
        v_global = v_feats.mean(dim=1)  # [B, D]

        if mask is not None:
            t_lengths = mask.sum(dim=1, keepdim=True).float()
            t_global = (t_feats * mask.unsqueeze(-1)).sum(dim=1) / t_lengths.clamp(min=1)
        else:
            t_global = t_feats.mean(dim=1)

        concat_feats = torch.cat([v_global, t_global], dim=-1)
        imbalance_score = self.feature_balancer(concat_feats)  # [B, 1]

        return imbalance_score

    def _feature_level_balance(self, v_feats, t_feats, mask, imbalance_score):
        """Pool both modalities to same dimensionality."""
        v_pooled = v_feats.mean(dim=1)  # [B, D]

        if mask is not None:
            t_lengths = mask.sum(dim=1, keepdim=True).float()
            t_pooled = (t_feats * mask.unsqueeze(-1)).sum(dim=1) / t_lengths.clamp(min=1)
        else:
            t_pooled = t_feats.mean(dim=1)

        return v_pooled, t_pooled

    def _compress_vision_for_damage_detection(self, v_feat):
        """Bottleneck compression preserving damage-relevant information."""
        compressed = self.vision_compressor(v_feat)
        return compressed

    def _enrich_text_with_damage_context(self, t_feat, t_sequence, device):
        """Add damage-level semantic context to sparse text features."""
        B = t_feat.shape[0]

        # Average damage embedding (in practice, parse labels for specific levels)
        damage_embeds = self.text_enhancer['damage_embeddings'].weight.mean(dim=0)
        damage_embeds = damage_embeds.unsqueeze(0).expand(B, -1)

        # Placeholder spatial info (in practice, extract from bboxes)
        spatial_info = torch.randn(B, 4, device=device) * 0.1
        spatial_embeds = self.text_enhancer['spatial_embeddings'](spatial_info)

        enhanced = torch.cat([damage_embeds, spatial_embeds], dim=-1)
        enhanced = self.text_enhancer['context_projector'](enhanced)

        return t_feat + 0.1 * enhanced  # residual connection

    def _compute_hierarchical_losses(self, v_aligned, t_aligned, v_feat, t_feat,
                                     v_compressed, imbalance_score, device):
        """Compute hierarchical loss terms."""
        losses = {}

        # L1: Alignment loss (dynamically weighted)
        weight = 1.0 / (1.0 + imbalance_score)
        losses['align'] = F.mse_loss(v_aligned, t_aligned) * weight.mean()

        # L2: Distribution matching loss (epsilon prevents zero std)
        v_mean, v_std = v_aligned.mean(dim=0), v_aligned.std(dim=0) + 1e-6
        t_mean, t_std = t_aligned.mean(dim=0), t_aligned.std(dim=0) + 1e-6
        losses['distribution'] = (F.mse_loss(v_mean, t_mean) +
                                  0.5 * F.mse_loss(v_std, t_std)) * 0.1

        # L3: Information preservation (prevent over-compression)
        losses['preserve'] = F.mse_loss(v_compressed, v_feat) * 0.1

        # L4: Orthogonality loss (encourage separation between damage levels)
        if hasattr(self.text_enhancer['damage_embeddings'], 'weight'):
            W = self.text_enhancer['damage_embeddings'].weight  # [4, D/4]
            W_norm = F.normalize(W, p=2, dim=1)
            gram = W_norm @ W_norm.T  # [4, 4]
            I = torch.eye(4, device=device)
            # Only penalize off-diagonal elements
            mask = 1.0 - I
            losses['orthogonal'] = (gram * mask).pow(2).mean() * 0.05

        losses['total'] = sum(losses.values())

        return losses


class BalancedHMB(nn.Module):
    """Balanced HMB: accounts for both length and density imbalance."""

    def __init__(self, hidden_dim=768, num_classes=4):
        super().__init__()

        # Vision information compressor (vision is typically over-represented)
        self.vision_compressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # Two-factor balance estimator: input [length_ratio, density_ratio]
        self.balance_estimator = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2),  # output: [vision_weight, text_weight]
            nn.Softmax(dim=-1)
        )

        # Adaptive projection
        self.adaptive_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, vision_features, text_features, labels=None, attention_mask=None):
        B = vision_features.shape[0]
        device = vision_features.device

        # Masked pooling
        vision_pooled = vision_features.mean(dim=1)

        if attention_mask is not None:
            valid_lengths = attention_mask.sum(dim=1).float()
            text_pooled = (text_features * attention_mask.unsqueeze(-1)).sum(dim=1) / valid_lengths.unsqueeze(-1).clamp(
                min=1)
        else:
            valid_lengths = torch.full((B,), text_features.shape[1], dtype=torch.float, device=device)
            text_pooled = text_features.mean(dim=1)

        # Length imbalance
        length_ratio = valid_lengths / vision_features.shape[1]

        # Density imbalance
        vision_std = vision_features.std(dim=(1, 2)).mean()
        text_std = (text_features * attention_mask.unsqueeze(-1) if attention_mask is not None else text_features).std(
            dim=(1, 2)).mean()
        density_ratio = (vision_std / (text_std + 1e-6)).expand(B)

        # Compress vision (typically over-represented)
        vision_compressed = self.vision_compressor(vision_pooled)

        # Balance weights from two factors
        balance_input = torch.stack([
            length_ratio.clamp(0, 2),
            density_ratio.clamp(0, 10) / 10
        ], dim=1)  # [B, 2]

        balance_weights = self.balance_estimator(balance_input)  # [B, 2]

        # Adaptive projection with weighted features
        proj_input = torch.cat([
            vision_compressed * balance_weights[:, 0:1],
            text_pooled * balance_weights[:, 1:2],
            balance_input
        ], dim=1)

        aligned = self.adaptive_proj(proj_input)

        # Losses
        losses = {}
        losses['align'] = F.mse_loss(vision_compressed, text_pooled)

        # Adaptive loss scaled by imbalance magnitude
        imbalance_factor = (length_ratio * density_ratio).clamp(0.1, 10)
        losses['adaptive'] = F.mse_loss(aligned, text_pooled) * imbalance_factor.mean()

        # Prevent over-compression
        losses['preserve'] = F.mse_loss(vision_compressed, vision_pooled.detach()) * 0.1

        losses['total'] = sum(losses.values())

        # Overall imbalance score
        imbalance_score = (balance_weights[:, 0] - balance_weights[:, 1]).abs().unsqueeze(-1)  # [B, 1]

        return losses, imbalance_score


class MinimalHMB(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=4):
        super().__init__()
        self.align_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, vision_features, text_features, labels=None, attention_mask=None):
        """
        Same interface as HierarchicalModalityBalance.

        Args:
            vision_features: [B, N, D]
            text_features: [B, T, D]
            labels: [B, T] (unused, kept for interface compatibility)
            attention_mask: [B, T]
        """
        v_pooled = vision_features.mean(dim=1)  # [B, D]

        if attention_mask is not None:
            t_lengths = attention_mask.sum(dim=1, keepdim=True).float()
            t_pooled = (text_features * attention_mask.unsqueeze(-1)).sum(dim=1) / t_lengths.clamp(min=1)
        else:
            t_pooled = text_features.mean(dim=1)  # [B, D]

        v_proj = self.align_proj(v_pooled)

        loss = F.mse_loss(v_proj, t_pooled)

        losses = {
            'align': loss,
            'total': loss
        }

        dummy_score = torch.tensor([0.5], device=vision_features.device)

        return losses, dummy_score


def masked_std(tensor, mask):
    """Standard deviation over masked positions."""
    if mask is None:
        return tensor.std(dim=(1, 2))

    mask_expanded = mask.unsqueeze(-1).float()
    mean = (tensor * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
    var = ((tensor - mean.unsqueeze(1)) ** 2 * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
    return var.sqrt().mean(dim=1)


def compute_features(vision_features, text_features, attention_mask=None):
    """Extract pooled statistics from vision and text features (shared across CPA variants)."""
    B = vision_features.shape[0]
    device = vision_features.device

    v_mean = vision_features.mean(dim=1)
    v_std = vision_features.std(dim=(1, 2))
    v_max = vision_features.max(dim=1)[0].mean(dim=1)

    if attention_mask is not None:
        seq_lengths = attention_mask.sum(dim=1).float()
        t_masked = text_features * attention_mask.unsqueeze(-1)
        t_mean = t_masked.sum(dim=1) / seq_lengths.unsqueeze(-1).clamp(min=1)
        t_std = masked_std(text_features, attention_mask)
        norm_lengths = (seq_lengths / 1000).clamp(0, 1)
    else:
        seq_lengths = torch.full((B,), text_features.shape[1], dtype=torch.float, device=device)
        t_mean = text_features.mean(dim=1)
        t_std = text_features.std(dim=(1, 2))
        norm_lengths = torch.ones(B, device=device)

    return {
        'v_mean': v_mean, 'v_std': v_std, 'v_max': v_max,
        't_mean': t_mean, 't_std': t_std,
        'seq_lengths': seq_lengths, 'norm_lengths': norm_lengths
    }

class SparsePathway(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.projector = nn.Linear(hidden_dim, hidden_dim)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, vision_features, t_mean):
        B = vision_features.shape[0]
        device = vision_features.device

        # Top-k patch selection by norm
        v_norms = vision_features.norm(dim=-1)
        _, top_indices = v_norms.topk(k=min(64, vision_features.shape[1]), dim=1)
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, top_indices.shape[1])
        v_selected = vision_features[batch_indices, top_indices]  # [B, k, D]

        # Cross-attention: text queries into selected vision patches
        t_query = t_mean.unsqueeze(1)  # [B, 1, D]
        with torch.cuda.amp.autocast(enabled=False):  # fp32 for stability
            v_attended, _ = self.attention(query=t_query, key=v_selected, value=v_selected)

        v_sparse = self.projector(v_attended.squeeze(1)) / self.temperature
        return v_sparse


class MediumPathway(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.region_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=4)
        self.projector = nn.Linear(hidden_dim, hidden_dim)
        self.combiner = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, vision_features, v_mean):
        v_regions = self.region_conv(vision_features.transpose(1, 2)).transpose(1, 2)
        v_region_pool = v_regions.mean(dim=1)
        v_medium = self.combiner(torch.cat([v_region_pool, self.projector(v_mean)], dim=-1))
        return v_medium


class DensePathway(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.pool = nn.Linear(hidden_dim, hidden_dim)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, v_mean):
        return self.pool(v_mean) * self.scale

class CPA(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=4):
        super().__init__()

        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, 128),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

        # Three pathways
        self.sparse_pathway = SparsePathway(hidden_dim)
        self.medium_pathway = MediumPathway(hidden_dim)
        self.dense_pathway = DensePathway(hidden_dim)

        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

        # Alignment head
        self.final_aligner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Cross-complexity transfer projection
        self.cross_complexity_transfer = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, vision_features, text_features, labels=None, attention_mask=None):
        B = vision_features.shape[0]
        device = vision_features.device

        # Step 1: Complexity estimation
        v_mean = vision_features.mean(dim=1)
        v_std = vision_features.std(dim=(1, 2))
        v_max = vision_features.max(dim=1)[0].mean(dim=1)

        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1).float()
            t_masked = text_features * attention_mask.unsqueeze(-1)
            t_mean = t_masked.sum(dim=1) / seq_lengths.unsqueeze(-1).clamp(min=1)
            t_std = masked_std(text_features, attention_mask)
            norm_lengths = (seq_lengths / 1000).clamp(0, 1)
        else:
            seq_lengths = torch.full((B,), text_features.shape[1], dtype=torch.float, device=device)
            t_mean = text_features.mean(dim=1)
            t_std = text_features.std(dim=(1, 2))
            norm_lengths = torch.ones(B, device=device)

        complexity_features = torch.cat([
            v_mean, v_std.unsqueeze(-1), v_max.unsqueeze(-1),
            t_mean, t_std.unsqueeze(-1), norm_lengths.unsqueeze(-1)
        ], dim=-1)
        complexity_scores = self.complexity_estimator(complexity_features)

        # Step 2: Three pathways
        v_sparse = self.sparse_pathway(vision_features, t_mean)
        v_medium = self.medium_pathway(vision_features, v_mean)
        v_dense = self.dense_pathway(v_mean)

        pathway_outputs = [v_sparse, v_medium, v_dense]

        # Step 3: Fusion
        fusion_input = torch.cat(pathway_outputs + [complexity_scores], dim=-1)
        fusion_weights = self.fusion_gate(fusion_input)

        v_fused = sum(w.unsqueeze(-1) * out for w, out in zip(fusion_weights.unbind(dim=1), pathway_outputs))

        # Step 4: Alignment
        v_final = self.final_aligner(v_fused)
        t_final = self.final_aligner(t_mean)

        # Step 5: Losses
        losses = {}
        losses['main_align'] = F.mse_loss(v_final, t_final)

        for i, (out, name) in enumerate(zip(pathway_outputs, ['sparse', 'medium', 'dense'])):
            weight = complexity_scores[:, i].mean()
            losses[f'{name}_align'] = F.mse_loss(out, t_mean) * weight.detach()

        all_pathways = torch.cat(pathway_outputs, dim=-1)
        losses['transfer'] = F.mse_loss(self.cross_complexity_transfer(all_pathways), t_mean) * 0.1

        if B > 1:
            length_sim = 1.0 / (1.0 + (seq_lengths.unsqueeze(1) - seq_lengths.unsqueeze(0)).abs())
            complexity_sim = 1.0 - (complexity_scores.unsqueeze(1) - complexity_scores.unsqueeze(0)).abs().mean(dim=-1)
            losses['consistency'] = F.mse_loss(length_sim, complexity_sim) * 0.05

        entropy = -(fusion_weights * torch.log(fusion_weights + 1e-8)).sum(dim=1).mean()
        losses['entropy'] = -entropy * 0.1

        losses['total'] = (
            losses['main_align'] +
            sum(losses[f'{p}_align'] for p in ['sparse', 'medium', 'dense']) * 0.3 +
            losses['transfer'] +
            losses.get('consistency', 0) +
            losses['entropy']
        )

        imbalance_score = complexity_scores.std(dim=1, keepdim=True)

        return losses, imbalance_score


class SinglePathwayCPA(nn.Module):
    """Ablation: single pathway only (no complexity-aware routing)."""

    def __init__(self, hidden_dim=768, num_classes=4):
        super().__init__()

        # Kept for interface compatibility but output is unused
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, 128),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

        # Single generic pathway instead of sparse/medium/dense
        self.single_pathway = nn.Linear(hidden_dim, hidden_dim)

        self.final_aligner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, vision_features, text_features, labels=None, attention_mask=None):
        B = vision_features.shape[0]
        device = vision_features.device

        features = compute_features(vision_features, text_features, attention_mask)
        v_mean = features['v_mean']
        t_mean = features['t_mean']

        v_single = self.single_pathway(v_mean)

        v_final = self.final_aligner(v_single)
        t_final = self.final_aligner(t_mean)

        losses = {}
        losses['main_align'] = F.mse_loss(v_final, t_final)
        losses['total'] = losses['main_align']

        return losses, None


class UniformWeightsCPA(nn.Module):
    """Ablation: three pathways with fixed uniform weights (no learned routing)."""

    def __init__(self, hidden_dim=768, num_classes=4):
        super().__init__()

        self.register_buffer('uniform_weights', torch.tensor([0.333, 0.333, 0.334]))

        self.sparse_pathway = SparsePathway(hidden_dim)
        self.medium_pathway = MediumPathway(hidden_dim)
        self.dense_pathway = DensePathway(hidden_dim)

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

        self.final_aligner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.cross_complexity_transfer = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, vision_features, text_features, labels=None, attention_mask=None):
        B = vision_features.shape[0]
        device = vision_features.device

        features = compute_features(vision_features, text_features, attention_mask)
        v_mean = features['v_mean']
        t_mean = features['t_mean']

        # Fixed uniform weights instead of learned complexity scores
        complexity_scores = self.uniform_weights.unsqueeze(0).expand(B, -1)

        v_sparse = self.sparse_pathway(vision_features, t_mean)
        v_medium = self.medium_pathway(vision_features, v_mean)
        v_dense = self.dense_pathway(v_mean)

        pathway_outputs = [v_sparse, v_medium, v_dense]

        # Fuse with fixed weights
        v_fused = sum(w.unsqueeze(-1) * out for w, out in
                      zip(complexity_scores.unbind(dim=1), pathway_outputs))

        v_final = self.final_aligner(v_fused)
        t_final = self.final_aligner(t_mean)

        losses = {}
        losses['main_align'] = F.mse_loss(v_final, t_final)

        for i, (out, name) in enumerate(zip(pathway_outputs, ['sparse', 'medium', 'dense'])):
            losses[f'{name}_align'] = F.mse_loss(out, t_mean) * complexity_scores[0, i]

        all_pathways = torch.cat(pathway_outputs, dim=-1)
        losses['transfer'] = F.mse_loss(self.cross_complexity_transfer(all_pathways), t_mean) * 0.1

        fusion_weights = self.fusion_gate(torch.cat(pathway_outputs, dim=-1))
        entropy = -(fusion_weights * torch.log(fusion_weights + 1e-8)).sum(dim=1).mean()
        losses['entropy'] = -entropy * 0.1

        losses['total'] = (
                losses['main_align'] +
                sum(losses[f'{p}_align'] for p in ['sparse', 'medium', 'dense']) * 0.3 +
                losses['transfer'] +
                losses['entropy']
        )

        return losses, complexity_scores


class HardSelectionCPA(nn.Module):
    """Ablation: argmax pathway selection instead of soft fusion."""

    def __init__(self, hidden_dim=768, num_classes=4):
        super().__init__()

        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, 128),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

        self.sparse_pathway = SparsePathway(hidden_dim)
        self.medium_pathway = MediumPathway(hidden_dim)
        self.dense_pathway = DensePathway(hidden_dim)

        self.final_aligner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, vision_features, text_features, labels=None, attention_mask=None):
        B = vision_features.shape[0]
        device = vision_features.device

        features = compute_features(vision_features, text_features, attention_mask)
        v_mean = features['v_mean']
        t_mean = features['t_mean']
        v_std = features['v_std']
        v_max = features['v_max']
        t_std = features['t_std']
        norm_lengths = features['norm_lengths']

        complexity_features = torch.cat([
            v_mean, v_std.unsqueeze(-1), v_max.unsqueeze(-1),
            t_mean, t_std.unsqueeze(-1), norm_lengths.unsqueeze(-1)
        ], dim=-1)
        complexity_scores = self.complexity_estimator(complexity_features)

        # Hard selection: pick highest-scoring pathway per sample
        selected_pathways = complexity_scores.argmax(dim=1)  # [B]

        v_sparse = self.sparse_pathway(vision_features, t_mean)
        v_medium = self.medium_pathway(vision_features, v_mean)
        v_dense = self.dense_pathway(v_mean)

        pathway_outputs = torch.stack([v_sparse, v_medium, v_dense], dim=1)  # [B, 3, D]

        batch_indices = torch.arange(B, device=device)
        v_selected = pathway_outputs[batch_indices, selected_pathways]  # [B, D]

        v_final = self.final_aligner(v_selected)
        t_final = self.final_aligner(t_mean)

        losses = {}
        losses['main_align'] = F.mse_loss(v_final, t_final)

        # Only selected pathways contribute to their respective losses
        for i, name in enumerate(['sparse', 'medium', 'dense']):
            pathway_out = pathway_outputs[:, i]  # [B, D]
            pathway_loss = F.mse_loss(pathway_out, t_mean)
            mask = (selected_pathways == i).float().mean()
            losses[f'{name}_align'] = pathway_loss * mask

        losses['total'] = (
                losses['main_align'] +
                sum(losses[f'{p}_align'] for p in ['sparse', 'medium', 'dense']) * 0.3
        )

        selection_onehot = F.one_hot(selected_pathways, num_classes=3).float()

        return losses, selection_onehot

class RoutingOnlyCPA(nn.Module):
    """
    Ablation: routing losses only, no cross-modal alignment.

    Keeps complexity estimator + 3 pathways + fusion gate.
    Only uses entropy (pathway diversity) and consistency losses.
    Removes all alignment and transfer losses.

    Tests hypothesis: routing is useful, alignment is harmful.
    """

    def __init__(self, hidden_dim=768, num_classes=4):
        super().__init__()

        # Complexity estimator (same as CPA)
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, 128),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

        # Three pathways (same as CPA)
        self.sparse_pathway = SparsePathway(hidden_dim)
        self.medium_pathway = MediumPathway(hidden_dim)
        self.dense_pathway = DensePathway(hidden_dim)

        # Fusion gate (same as CPA)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

        # No final_aligner or cross_complexity_transfer (not needed without alignment loss)

    def forward(self, vision_features, text_features, labels=None, attention_mask=None):
        B = vision_features.shape[0]
        device = vision_features.device

        # Step 1: Complexity estimation
        v_mean = vision_features.mean(dim=1)
        v_std = vision_features.std(dim=(1, 2))
        v_max = vision_features.max(dim=1)[0].mean(dim=1)

        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1).float()
            t_masked = text_features * attention_mask.unsqueeze(-1)
            t_mean = t_masked.sum(dim=1) / seq_lengths.unsqueeze(-1).clamp(min=1)
            t_std = masked_std(text_features, attention_mask)
            norm_lengths = (seq_lengths / 1000).clamp(0, 1)
        else:
            seq_lengths = torch.full((B,), text_features.shape[1], dtype=torch.float, device=device)
            t_mean = text_features.mean(dim=1)
            t_std = text_features.std(dim=(1, 2))
            norm_lengths = torch.ones(B, device=device)

        complexity_features = torch.cat([
            v_mean, v_std.unsqueeze(-1), v_max.unsqueeze(-1),
            t_mean, t_std.unsqueeze(-1), norm_lengths.unsqueeze(-1)
        ], dim=-1)
        complexity_scores = self.complexity_estimator(complexity_features)

        # Step 2: Three pathways
        v_sparse = self.sparse_pathway(vision_features, t_mean)
        v_medium = self.medium_pathway(vision_features, v_mean)
        v_dense = self.dense_pathway(v_mean)

        pathway_outputs = [v_sparse, v_medium, v_dense]

        # Step 3: Fusion
        fusion_input = torch.cat(pathway_outputs + [complexity_scores], dim=-1)
        fusion_weights = self.fusion_gate(fusion_input)

        v_fused = sum(w.unsqueeze(-1) * out for w, out in zip(fusion_weights.unbind(dim=1), pathway_outputs))

        # Step 4: Routing-only losses (no alignment)
        losses = {}

        # Consistency: similar-length samples should get similar complexity scores
        if B > 1:
            length_sim = 1.0 / (1.0 + (seq_lengths.unsqueeze(1) - seq_lengths.unsqueeze(0)).abs())
            complexity_sim = 1.0 - (complexity_scores.unsqueeze(1) - complexity_scores.unsqueeze(0)).abs().mean(dim=-1)
            losses['consistency'] = F.mse_loss(length_sim, complexity_sim) * 0.05
        else:
            losses['consistency'] = torch.tensor(0.0, device=device)

        # Entropy: encourage pathway diversity (maximize entropy)
        entropy = -(fusion_weights * torch.log(fusion_weights + 1e-8)).sum(dim=1).mean()
        losses['entropy'] = -entropy * 0.1

        losses['total'] = losses['consistency'] + losses['entropy']

        imbalance_score = complexity_scores.std(dim=1, keepdim=True)

        return losses, imbalance_score
