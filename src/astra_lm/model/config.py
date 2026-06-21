from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    bias: bool = False
    tie_word_embeddings: bool = True
    activation: str = "silu"
    norm_type: str = "rmsnorm"

    # Attention Parameters
    attention_type: str = "sdpa" # "sdpa", "gqa" (legacy), or "chakra_legacy"
    use_flash_attention: bool = True
    local_window: int = 128
    sphere_buckets: int = 32
    nearby_buckets: int = 1
    use_exact_qk_after_routing: bool = True

    # VayuSphere Parameters
    use_vayusphere: bool = False
    vayusphere_mode: str = "scale" # "scale", "tangent", "tangent_scale"
    vayusphere_target: str = "qk" # "q", "k", or "qk"
    vayusphere_num_centroids: int = 32
    vayusphere_alpha: float = 0.1
    vayusphere_scale_alpha: float = 0.1
    vayusphere_topk_centroids: int = -1
    vayusphere_normalize: bool = True
    vayusphere_temperature: float = 1.0
    vayusphere_temperature_start: float = 1.0
    vayusphere_temperature_min: float = 1.0
    vayusphere_temperature_decay_steps: int = 0
    vayusphere_apply_stage: str = "post_rope" # "pre_rope", "post_rope"
    vayusphere_centroid_scope: str = "layer_shared"
    vayusphere_freeze_centroids: bool = False
    vayusphere_diagnostics_every_n_steps: int = 1
    vayusphere_enable_heavy_diagnostics: bool = True


    # Attention Modulation
    use_learned_attention_temp: bool = False

    # AKASHA Memory Parameters
    memory_window: int = 256
    anchor_interval: int = 16
    use_recent_bank: bool = True
    use_anchor_bank: bool = False
    use_summary_bank: bool = False
    use_latent_bank: bool = False
    use_sphere_bank: bool = False

    # Optional / Future Modules Config Gates
    use_surya: bool = False
    use_indra_phase: bool = False
    ffn_type: str = "swiglu" # "swiglu" or "fock"

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len ({self.max_seq_len}) must be positive")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers ({self.n_layers}) must be positive")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_kv_groups(self) -> int:
        return self.n_heads // self.n_kv_heads
