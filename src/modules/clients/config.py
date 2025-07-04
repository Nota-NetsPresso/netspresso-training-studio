import configparser
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from src.modules.clients.enums.config import EndPointProperty, EnvironmentType, ServiceModule, ServiceName

BASE_DIR = Path(__file__).resolve().parent
config_parser = configparser.ConfigParser()
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "v2-prod-cloud")
config_parser.read(f"{BASE_DIR}/configs/config-{DEPLOYMENT_MODE.lower()}.ini")


class Config:
    _printed = False

    def __init__(self, service_name: ServiceName, module: ServiceModule):
        self.ENVIRONMENT_TYPE = EnvironmentType(DEPLOYMENT_MODE.lower())
        self.SERVICE_NAME = service_name
        self.MODULE = module

        dotenv_path = find_dotenv(filename="netspresso.env")
        if dotenv_path:
            load_dotenv(dotenv_path)

        if self.SERVICE_NAME == ServiceName.DATAFORGE:
            self.HOST = os.environ.get("DATAFORGE_HOST")
            self.PORT = int(os.environ.get("DATAFORGE_PORT"))
            self.URI_PREFIX = os.environ.get("DATAFORGE_URI_PREFIX")
        else:
            self.HOST = os.environ.get("HOST", config_parser[ServiceName.NP][EndPointProperty.HOST])
            self.PORT = int(os.environ.get("PORT", config_parser[ServiceName.NP][EndPointProperty.PORT]))
            self.URI_PREFIX = config_parser[f"NP.{self.MODULE}"][EndPointProperty.URI_PREFIX]

        self._print_host_and_port()

    def _print_host_and_port(self):
        if not Config._printed and self.SERVICE_NAME == ServiceName.NP:
            logger.info(f"Host: {self.HOST}, Port: {self.PORT}")
            Config._printed = True

    def is_cloud(self) -> bool:
        return self.ENVIRONMENT_TYPE in [
            EnvironmentType.V2_PROD_CLOUD,
            EnvironmentType.V2_STAGING_CLOUD,
            EnvironmentType.V2_DEV_CLOUD,
        ]
