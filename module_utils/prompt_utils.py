import torch
from torch import nn as neural_net
from timm.models.layers import trunc_normal_ as initialize_weights_with_truncated_normal

# --- Global device configuration ---
# Extract device detection logic to avoid redundancy across modules.
# This also makes the code more adaptable to different hardware environments.
COMPUTATION_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FastGELUActivation(neural_net.Module):
    """
    An efficient approximation of the GELU activation function.
    Formula: f(x) = x * sigmoid(1.702 * x)
    """
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor * torch.sigmoid(1.702 * input_tensor)

class AttentionMechanism(neural_net.Module):
    """
    A general multi-head attention mechanism.
    Supports both self-attention and cross-attention.
    """
    def __init__(self, embedding_dim, n_heads=8, use_bias_for_qkv=False, scale_factor=None, attention_dropout_rate=0., output_dropout_rate=0.):
        super().__init__()
        self.n_heads = n_heads
        head_dimension = embedding_dim // n_heads
        
        # Use a more descriptive name
        self.scaling_factor = scale_factor or head_dimension ** -0.5

        # Define separate linear layers for Q, K, V
        self.query_transform = neural_net.Linear(embedding_dim, embedding_dim, bias=use_bias_for_qkv, device=COMPUTATION_DEVICE)
        self.key_transform = neural_net.Linear(embedding_dim, embedding_dim, bias=use_bias_for_qkv, device=COMPUTATION_DEVICE)
        self.value_transform = neural_net.Linear(embedding_dim, embedding_dim, bias=use_bias_for_qkv, device=COMPUTATION_DEVICE)

        self.attention_dropout = neural_net.Dropout(attention_dropout_rate)
        
        # Output projection layer
        self.output_projection = neural_net.Linear(embedding_dim, embedding_dim, device=COMPUTATION_DEVICE)
        self.output_dropout = neural_net.Dropout(output_dropout_rate)

    def forward(self, query, key, value):
        batch_size_q, seq_len_q, channels_q = query.shape
        batch_size_k, seq_len_k, channels_k = key.shape

        # 1. Linear projections and reshape to multi-head format
        # (B, S, C) -> (B, S, H, D) -> (B, H, S, D)
        # B=batch size, S=sequence length, C=channels, H=heads, D=head dimension
        q_h = self.query_transform(query).view(batch_size_q, seq_len_q, self.n_heads, channels_q // self.n_heads).permute(0, 2, 1, 3)
        k_h = self.key_transform(key).view(batch_size_k, seq_len_k, self.n_heads, channels_k // self.n_heads).permute(0, 2, 1, 3)
        v_h = self.value_transform(value).view(batch_size_k, seq_len_k, self.n_heads, channels_k // self.n_heads).permute(0, 2, 1, 3)

        # 2. Compute attention scores
        attention_scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scaling_factor
        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # 3. Apply attention weights to value
        context_vector = torch.matmul(attention_weights, v_h)
        
        # 4. Reshape output and apply final projection
        context_vector = context_vector.transpose(1, 2).contiguous().view(batch_size_q, seq_len_q, channels_q)
        
        final_output = self.output_projection(context_vector)
        final_output = self.output_dropout(final_output)
        
        return final_output

class PromptRefinementBlock(neural_net.Module):
    """
    A decoder block for generating and refining prompts.
    Includes a cross-attention mechanism and a feed-forward network.
    """
    def __init__(self, model_dimension, num_attention_heads, dropout_rate=0.):
        super().__init__()
        # Use renamed attention class
        self.cross_attention = AttentionMechanism(model_dimension, num_attention_heads, output_dropout_rate=dropout_rate)

        # Use more descriptive layer names
        self.norm_for_attention = neural_net.LayerNorm(model_dimension, device=COMPUTATION_DEVICE)
        self.norm_for_ffn = neural_net.LayerNorm(model_dimension, device=COMPUTATION_DEVICE)
        self.dropout_layer = neural_net.Dropout(dropout_rate)

        # Define feed-forward network (MLP)
        self.feed_forward_net = neural_net.Sequential(
            neural_net.Linear(model_dimension, model_dimension * 4, device=COMPUTATION_DEVICE),
            FastGELUActivation(),
            neural_net.Dropout(dropout_rate),
            neural_net.Linear(model_dimension * 4, model_dimension, device=COMPUTATION_DEVICE)
        )

    def forward(self, text_features, visual_context):
        # Cross-attention part
        # Query comes from text, Key and Value come from visual features
        query_input = self.norm_for_attention(text_features)
        # Note: visual_context is assumed to be normalized already
        attended_features = self.cross_attention(query_input, visual_context, visual_context)
        # Residual connection
        text_features = text_features + attended_features

        # Feed-forward part
        ffn_input = self.norm_for_ffn(text_features)
        ffn_output = self.feed_forward_net(ffn_input)
        # Second residual connection
        refined_text_features = text_features + self.dropout_layer(ffn_output)
        
        return refined_text_features

class VisualPromptAdapter(neural_net.Module):
    """
    This module adjusts and generates text prompts based on visual features from video.
    """
    def __init__(self, num_blocks=2, feature_dim=512, initial_alpha=0.1):
        super().__init__()
        # Normalization layer for visual features
        self.visual_feature_normalizer = neural_net.LayerNorm(feature_dim, device=COMPUTATION_DEVICE)
        
        # Stack multiple prompt refinement blocks as a decoder
        self.refinement_blocks = neural_net.ModuleList(
            [PromptRefinementBlock(feature_dim, feature_dim // 64) for _ in range(num_blocks)]
        )
        
        # Learnable scaling factor alpha
        self.prompt_scaling_factor = neural_net.Parameter(torch.full((feature_dim,), initial_alpha))
        
        # Apply custom weight initialization
        self.apply(self._initialize_parameters)

    def _initialize_parameters(self, module):
        """Custom weight initialization function"""
        if isinstance(module, neural_net.Linear):
            # Use truncated normal distribution from timm
            initialize_weights_with_truncated_normal(module.weight, std=.02)
            if module.bias is not None:
                neural_net.init.constant_(module.bias, 0)
        elif isinstance(module, neural_net.LayerNorm):
            neural_net.init.constant_(module.bias, 0)
            neural_net.init.constant_(module.weight, 1.0)
    
    def forward(self, text_prompts, visual_features):
        batch_size, num_tokens, channels = visual_features.shape
        
        # Normalize visual features
        normalized_visual = self.visual_feature_normalizer(visual_features)
        
        # Iteratively refine text prompts through each block
        processed_prompts = text_prompts
        for block in self.refinement_blocks:
            processed_prompts = block(processed_prompts, normalized_visual)
            
        # Ensure scaling factor is on the same device as input
        scaling = self.prompt_scaling_factor.to(processed_prompts.device)
        
        # Apply scaling and return result
        return scaling * processed_prompts