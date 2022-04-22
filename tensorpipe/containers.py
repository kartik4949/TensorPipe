from bunch import Bunch
from dependency_injector import containers, providers

from .augment import augment

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    external_config = providers.Factory(Bunch, config.external_config)
    augmenter = providers.Factory(
        augment.Augment,
        config=external_config,
        datatype=config.datatype
    )
