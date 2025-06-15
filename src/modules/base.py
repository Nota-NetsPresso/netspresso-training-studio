from loguru import logger

from src.exceptions.common import NotEnoughCreditException
from src.modules.clients.auth import TokenHandler, auth_client
from src.modules.enums.credit import ServiceCredit, ServiceTask


class NetsPressoBase:
    def __init__(self, token_handler: TokenHandler) -> None:
        self.token_handler = token_handler
        self.auth_client = auth_client

    def check_credit_balance(self, service_task: ServiceTask):
        current_credit = self.auth_client.get_credit(
            access_token=self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
        )
        service_credit = ServiceCredit.get_credit(service_task)
        service_task_name = service_task.name.replace("_", " ").lower()
        if current_credit < service_credit:
            logger.error(
                f"Insufficient balance: {current_credit} credits available, but {service_credit} credits required for {service_task_name} task."
            )
            raise NotEnoughCreditException(current_credit, service_credit, service_task_name)

    def print_remaining_credit(self, service_task):
        if self.auth_client.is_cloud():
            self.token_handler.validate_token()
            service_credit = ServiceCredit.get_credit(service_task)
            remaining_credit = self.auth_client.get_credit(
                self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )
            logger.info(f"{service_credit} credits have been consumed. Remaining Credit: {remaining_credit}")

    def validate_token_and_check_credit(self, service_task: ServiceTask):
        self.token_handler.validate_token()
        self.check_credit_balance(service_task=service_task)
