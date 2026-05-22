from transformers import PretrainedConfig


class ZeonConfig(PretrainedConfig):
    model_type = "zeon"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 4,
        num_recurrent_layers: int = 2,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 16384,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        # ----- Zeon-specific knobs -----
        max_recurrent_steps: int = 16,
        min_recurrent_steps: int = 1,
        halt_threshold: float = 0.9,
        ponder_loss_weight: float = 1e-2,
        ponder_lambda_p: float = 0.2,
        halt_entropy_weight: float = 1e-3,
        share_recurrent_weights: bool = True,
        recurrent_state_mix: str = "gated",  # residual | gated
        use_step_embedding: bool = True,
        cross_step_memory: bool = True,
        cross_step_memory_window: int = 4,
        # ----- Phase 1: Workspace Bank -----
        use_workspace: bool = True,
        workspace_num_slots: int = 16,
        workspace_num_heads: int = 4,
        workspace_diversity_weight: float = 1e-3,
        workspace_sticky_bias_init: float = 2.0,
        freeze_ffn: bool = True,
        freeze_embed: bool = True,
        freeze_lm_head: bool = True,
        base_model_name_or_path: str | None = None,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_recurrent_layers = num_recurrent_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range

        self.max_recurrent_steps = max_recurrent_steps
        self.min_recurrent_steps = min_recurrent_steps
        self.halt_threshold = halt_threshold
        self.ponder_loss_weight = ponder_loss_weight
        self.ponder_lambda_p = ponder_lambda_p
        self.halt_entropy_weight = halt_entropy_weight
        self.share_recurrent_weights = share_recurrent_weights
        self.recurrent_state_mix = recurrent_state_mix
        self.use_step_embedding = use_step_embedding
        self.cross_step_memory = cross_step_memory
        self.cross_step_memory_window = cross_step_memory_window
        self.use_workspace = use_workspace
        self.workspace_num_slots = workspace_num_slots
        self.workspace_num_heads = workspace_num_heads
        self.workspace_diversity_weight = workspace_diversity_weight
        self.workspace_sticky_bias_init = workspace_sticky_bias_init
        self.freeze_ffn = freeze_ffn
        self.freeze_embed = freeze_embed
        self.freeze_lm_head = freeze_lm_head
        self.base_model_name_or_path = base_model_name_or_path

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
