# Keep existing imports and add new ones
from ..maddpg import MADDPGTrainer
from ..attention import AttentionTrainer
from .pso_baseline import PSOBaseline
from .baseline_trainer import BaselineTrainer

__all__ = [
    'MADDPGTrainer',
    'AttentionTrainer', 
    'PSOBaseline',
    'BaselineTrainer'
]