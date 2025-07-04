from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.modules.trainer.models.base import ArchitectureConfig, CheckpointConfig, ModelConfig


@dataclass
class MixNetSmallArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "mixnet",
            "params": {
                "stem_channels": 16,
                "wid_mul": 1.0,
                "dep_mul": 1.0,
                "dropout_rate": 0.0,
            },
            "stage_params": [
                {
                    "expansion_ratio": [1, 6, 3],
                    "out_channels": [16, 24, 24],
                    "num_blocks": [1, 1, 1],
                    "kernel_sizes": [[3], [3], [3]],
                    "num_exp_groups": [1, 2, 2],
                    "num_poi_groups": [1, 2, 2],
                    "stride": [1, 2, 1],
                    "act_type": ["relu", "relu", "relu"],
                    "se_reduction_ratio": [None, None, None],
                },
                {
                    "expansion_ratio": [6, 6],
                    "out_channels": [40, 40],
                    "num_blocks": [1, 3],
                    "kernel_sizes": [[3, 5, 7], [3, 5]],
                    "num_exp_groups": [1, 2],
                    "num_poi_groups": [1, 2],
                    "stride": [2, 1],
                    "act_type": ["swish", "swish"],
                    "se_reduction_ratio": [2, 2],
                },
                {
                    "expansion_ratio": [6, 6, 6, 3],
                    "out_channels": [80, 80, 120, 120],
                    "num_blocks": [1, 2, 1, 2],
                    "kernel_sizes": [[3, 5, 7], [3, 5], [3, 5, 7], [3, 5, 7, 9]],
                    "num_exp_groups": [1, 1, 2, 2],
                    "num_poi_groups": [2, 2, 2, 2],
                    "stride": [2, 1, 1, 1],
                    "act_type": ["swish", "swish", "swish", "swish"],
                    "se_reduction_ratio": [4, 4, 2, 2],
                },
                {
                    "expansion_ratio": [6, 6],
                    "out_channels": [200, 200],
                    "num_blocks": [1, 2],
                    "kernel_sizes": [[3, 5, 7, 9, 11], [3, 5, 7, 9]],
                    "num_exp_groups": [1, 1],
                    "num_poi_groups": [1, 2],
                    "stride": [2, 1],
                    "act_type": ["swish", "swish"],
                    "se_reduction_ratio": [2, 2],
                },
            ],
        }
    )


@dataclass
class MixNetMediumArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "mixnet",
            "params": {
                "stem_channels": 24,
                "wid_mul": 1.0,
                "dep_mul": 1.0,
                "dropout_rate": 0.0,
            },
            "stage_params": [
                {
                    "expansion_ratio": [1, 6, 3],
                    "out_channels": [24, 32, 32],
                    "num_blocks": [1, 1, 1],
                    "kernel_sizes": [[3], [3, 5, 7], [3]],
                    "num_exp_groups": [1, 2, 2],
                    "num_poi_groups": [1, 2, 2],
                    "stride": [1, 2, 1],
                    "act_type": ["relu", "relu", "relu"],
                    "se_reduction_ratio": [None, None, None],
                },
                {
                    "expansion_ratio": [6, 6],
                    "out_channels": [40, 40],
                    "num_blocks": [1, 3],
                    "kernel_sizes": [[3, 5, 7, 9], [3, 5]],
                    "num_exp_groups": [1, 2],
                    "num_poi_groups": [1, 2],
                    "stride": [2, 1],
                    "act_type": ["swish", "swish"],
                    "se_reduction_ratio": [2, 2],
                },
                {
                    "expansion_ratio": [6, 6, 6, 3],
                    "out_channels": [80, 80, 120, 120],
                    "num_blocks": [1, 3, 1, 3],
                    "kernel_sizes": [[3, 5, 7], [3, 5, 7, 9], [3], [3, 5, 7, 9]],
                    "num_exp_groups": [1, 2, 1, 2],
                    "num_poi_groups": [1, 2, 1, 2],
                    "stride": [2, 1, 1, 1],
                    "act_type": ["swish", "swish", "swish", "swish"],
                    "se_reduction_ratio": [4, 4, 2, 2],
                },
                {
                    "expansion_ratio": [6, 6],
                    "out_channels": [200, 200],
                    "num_blocks": [1, 3],
                    "kernel_sizes": [[3, 5, 7, 9], [3, 5, 7, 9]],
                    "num_exp_groups": [1, 1],
                    "num_poi_groups": [1, 2],
                    "stride": [2, 1],
                    "act_type": ["swish", "swish"],
                    "se_reduction_ratio": [2, 2],
                },
            ],
        }
    )


@dataclass
class MixNetLargeArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "mixnet",
            "params": {
                "stem_channels": 24,
                "wid_mul": 1.3,
                "dep_mul": 1.0,
                "dropout_rate": 0.0,
            },
            "stage_params": [
                {
                    "expansion_ratio": [1, 6, 3],
                    "out_channels": [24, 32, 32],
                    "num_blocks": [1, 1, 1],
                    "kernel_sizes": [[3], [3, 5, 7], [3]],
                    "num_exp_groups": [1, 2, 2],
                    "num_poi_groups": [1, 2, 2],
                    "stride": [1, 2, 1],
                    "act_type": ["relu", "relu", "relu"],
                    "se_reduction_ratio": [None, None, None],
                },
                {
                    "expansion_ratio": [6, 6],
                    "out_channels": [40, 40],
                    "num_blocks": [1, 3],
                    "kernel_sizes": [[3, 5, 7, 9], [3, 5]],
                    "num_exp_groups": [1, 2],
                    "num_poi_groups": [1, 2],
                    "stride": [2, 1],
                    "act_type": ["swish", "swish"],
                    "se_reduction_ratio": [2, 2],
                },
                {
                    "expansion_ratio": [6, 6, 6, 3],
                    "out_channels": [80, 80, 120, 120],
                    "num_blocks": [1, 3, 1, 3],
                    "kernel_sizes": [[3, 5, 7], [3, 5, 7, 9], [3], [3, 5, 7, 9]],
                    "num_exp_groups": [1, 2, 1, 2],
                    "num_poi_groups": [1, 2, 1, 2],
                    "stride": [2, 1, 1, 1],
                    "act_type": ["swish", "swish", "swish", "swish"],
                    "se_reduction_ratio": [4, 4, 2, 2],
                },
                {
                    "expansion_ratio": [6, 6],
                    "out_channels": [200, 200],
                    "num_blocks": [1, 3],
                    "kernel_sizes": [[3, 5, 7, 9], [3, 5, 7, 9]],
                    "num_exp_groups": [1, 1],
                    "num_poi_groups": [1, 2],
                    "stride": [2, 1],
                    "act_type": ["swish", "swish"],
                    "se_reduction_ratio": [2, 2],
                },
            ],
        }
    )


@dataclass
class ClassificationMixNetSmallModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "mixnet_s"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: MixNetSmallArchitectureConfig(
            head={
                "name": "fc",
                "params": {
                    "num_layers": 1,
                    "intermediate_channels": None,
                    "act_type": None,
                    "dropout_prob": 0.0,
                },
            }
        )
    )
    postprocessor: Optional[Dict[str, Any]] = None
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}]
    )


@dataclass
class SegmentationMixNetSmallModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "mixnet_s"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: MixNetSmallArchitectureConfig(
            head={
                "name": "all_mlp_decoder",
                "params": {
                    "intermediate_channels": 256,
                    "classifier_dropout_prob": 0.0,
                },
            }
        )
    )
    postprocessor: Optional[Dict[str, Any]] = None
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "seg_cross_entropy", "ignore_index": 255, "weight": None}]
    )


@dataclass
class DetectionMixNetSmallModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "mixnet_s"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: MixNetSmallArchitectureConfig(
            neck={
                "name": "fpn",
                "params": {
                    "num_outs": 4,
                    "start_level": 0,
                    "end_level": -1,
                    "add_extra_convs": False,
                    "relu_before_extra_convs": False,
                },
            },
            head={
                "name": "anchor_decoupled_head",
                "params": {
                    # Anchor parameters
                    "anchor_sizes": [
                        [
                            32,
                        ],
                        [
                            64,
                        ],
                        [
                            128,
                        ],
                        [
                            256,
                        ],
                    ],
                    "aspect_ratios": [0.5, 1.0, 2.0],
                    "num_layers": 1,
                    "norm_type": "batch_norm",
                },
            },
        )
    )
    postprocessor: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "params": {
                # postprocessor - decode
                "topk_candidates": 1000,
                "score_thresh": 0.05,
                # postprocessor - nms
                "nms_thresh": 0.45,
                "class_agnostic": False,
            },
        }
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"criterion": "retinanet_loss", "weight": None},
        ]
    )


@dataclass
class ClassificationMixNetMediumModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "mixnet_m"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: MixNetMediumArchitectureConfig(
            head={
                "name": "fc",
                "params": {
                    "num_layers": 1,
                    "intermediate_channels": None,
                    "act_type": None,
                    "dropout_prob": 0.0,
                },
            }
        )
    )
    postprocessor: Optional[Dict[str, Any]] = None
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}]
    )


@dataclass
class SegmentationMixNetMediumModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "mixnet_m"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: MixNetMediumArchitectureConfig(
            head={
                "name": "all_mlp_decoder",
                "params": {
                    "intermediate_channels": 256,
                    "classifier_dropout_prob": 0.0,
                },
            }
        )
    )
    postprocessor: Optional[Dict[str, Any]] = None
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "seg_cross_entropy", "ignore_index": 255, "weight": None}]
    )


@dataclass
class DetectionMixNetMediumModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "mixnet_m"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: MixNetMediumArchitectureConfig(
            neck={
                "name": "fpn",
                "params": {
                    "num_outs": 4,
                    "start_level": 0,
                    "end_level": -1,
                    "add_extra_convs": False,
                    "relu_before_extra_convs": False,
                },
            },
            head={
                "name": "anchor_decoupled_head",
                "params": {
                    # Anchor parameters
                    "anchor_sizes": [
                        [
                            32,
                        ],
                        [
                            64,
                        ],
                        [
                            128,
                        ],
                        [
                            256,
                        ],
                    ],
                    "aspect_ratios": [0.5, 1.0, 2.0],
                    "num_layers": 1,
                    "norm_type": "batch_norm",
                },
            },
        )
    )
    postprocessor: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "params": {
                # postprocessor - decode
                "topk_candidates": 1000,
                "score_thresh": 0.05,
                # postprocessor - nms
                "nms_thresh": 0.45,
                "class_agnostic": False,
            },
        }
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"criterion": "retinanet_loss", "weight": None},
        ]
    )


@dataclass
class ClassificationMixNetLargeModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "mixnet_l"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: MixNetLargeArchitectureConfig(
            head={
                "name": "fc",
                "params": {
                    "num_layers": 1,
                    "intermediate_channels": None,
                    "act_type": None,
                    "dropout_prob": 0.0,
                },
            }
        )
    )
    postprocessor: Optional[Dict[str, Any]] = None
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}]
    )


@dataclass
class SegmentationMixNetLargeModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "mixnet_l"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: MixNetLargeArchitectureConfig(
            head={
                "name": "all_mlp_decoder",
                "params": {
                    "intermediate_channels": 256,
                    "classifier_dropout_prob": 0.0,
                },
            }
        )
    )
    postprocessor: Optional[Dict[str, Any]] = None
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "seg_cross_entropy", "ignore_index": 255, "weight": None}]
    )


@dataclass
class DetectionMixNetLargeModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "mixnet_l"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: MixNetLargeArchitectureConfig(
            neck={
                "name": "fpn",
                "params": {
                    "num_outs": 4,
                    "start_level": 0,
                    "end_level": -1,
                    "add_extra_convs": False,
                    "relu_before_extra_convs": False,
                },
            },
            head={
                "name": "anchor_decoupled_head",
                "params": {
                    # Anchor parameters
                    "anchor_sizes": [
                        [
                            32,
                        ],
                        [
                            64,
                        ],
                        [
                            128,
                        ],
                        [
                            256,
                        ],
                    ],
                    "aspect_ratios": [0.5, 1.0, 2.0],
                    "num_layers": 1,
                    "norm_type": "batch_norm",
                },
            },
        )
    )
    postprocessor: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "params": {
                # postprocessor - decode
                "topk_candidates": 1000,
                "score_thresh": 0.05,
                # postprocessor - nms
                "nms_thresh": 0.45,
                "class_agnostic": False,
            },
        }
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"criterion": "retinanet_loss", "weight": None},
        ]
    )
