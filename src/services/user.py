from src.api.v1.schemas.user import CreditInfo, DetailData, UserPayload
from src.clients.auth import auth_client


class UserService:
    def get_user_info(self, token: str) -> UserPayload:
        user_info = auth_client.get_user_info(access_token=token)

        return UserPayload(
            user_id=user_info.user_id,
            email=user_info.email,
            detail_data=DetailData(
                first_name=user_info.detail_data.first_name,
                last_name=user_info.detail_data.last_name,
                company=user_info.detail_data.company,
            ),
            credit_info=CreditInfo(
                free=user_info.credit_info.free,
                reward=user_info.credit_info.reward,
                contract=user_info.credit_info.contract,
                paid=user_info.credit_info.paid,
                total=user_info.credit_info.total,
            ),
        )


user_service = UserService()
