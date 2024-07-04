from datetime import datetime, timedelta
from typing import List
from pydantic import BaseModel


class BasicInfo(BaseModel):
    host: str = "https://hsmoa.com"
    date: str = datetime.today().strftime("%Y%m%d")
    request_url: str = f"{host}/?date={date}"

    @classmethod
    def get_30days(cls) -> List[str]:
        today = datetime.today()
        return [(today - timedelta(days=i)).strftime("%Y%m%d") for i in range(30)]


class ItemInfo(BaseModel):
    item_name: str
    item_price: str
    item_image: str
    item_url: str
    shopping_mall_name: str
