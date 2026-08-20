from decimal import Decimal

from pydantic import BaseModel


class Item(BaseModel):
    title: str
    category: str
    price: Decimal
