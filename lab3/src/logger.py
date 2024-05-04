import logging


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s',
)

__all__ = [
    'logger',
]
