import logging

from app.core.config import Settings, settings


def configure_logging(config: Settings = settings) -> None:
    logging.basicConfig(
        level=config.log_level,
        format=config.log_format,
    )
