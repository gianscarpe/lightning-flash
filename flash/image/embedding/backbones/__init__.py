from flash.core.registry import FlashRegistry  # noqa: F401
from flash.image.classification.backbones import IMAGE_CLASSIFIER_BACKBONES  # noqa: F401
from flash.image.embedding.backbones.vissl_backbones import register_vissl_backbones  # noqa: F401

register_vissl_backbones(IMAGE_CLASSIFIER_BACKBONES)
