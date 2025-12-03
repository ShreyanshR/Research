from typing import List
from assets import Position, AssetClass

class Portfolio:
    positions: List[Position]
    target_vol: float
    MAX_INTERPO