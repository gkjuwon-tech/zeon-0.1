from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from zeon.config import ZeonConfig
from zeon.modeling_zeon import (
    RecurrentCore,
    ZeonBlock,
    ZeonForCausalLM,
    ZeonModel,
)

# Register with HF Auto classes so that:
#   AutoConfig.from_pretrained("path/to/zeon")
#   AutoModelForCausalLM.from_pretrained("path/to/zeon")
# work the moment `import zeon` happens — no extra glue at the call site.
AutoConfig.register("zeon", ZeonConfig, exist_ok=True)
AutoModel.register(ZeonConfig, ZeonModel, exist_ok=True)
AutoModelForCausalLM.register(ZeonConfig, ZeonForCausalLM, exist_ok=True)

__all__ = [
    "ZeonConfig",
    "RecurrentCore",
    "ZeonBlock",
    "ZeonModel",
    "ZeonForCausalLM",
]

__version__ = "0.1.0"
