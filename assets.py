from enum import Enum
from typing import Optional

class AssetClass(Enum):
    EQUITY = "equity"
    BOND = "bond"
    CRYPTO = "crypto"
    METALS = "metals"
    AI = "ai"
    SPECULATION = "speculation"

class Position:
    ticker: str
    weight: float
    asset_class: AssetClass
    sector: Optional[str]
    strategy: Optional[str]