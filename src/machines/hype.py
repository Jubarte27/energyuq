from typing import ClassVar
from .machine import Machine


# Todo: Consultar essas informações programaticamente, sem necessidade de fazer estes arquivos
class Hype(Machine):
    name: ClassVar[str] = "Hype"
    freq: ClassVar[list[int]] = [
        1200000,
        1300000,
        1400000,
        1500000,
        1600000,
        1700000,
        1800000,
        1900000,
        2000000,
        2100000,
        2200000,
        2300000,
        2301000,
    ]
    max_threads: ClassVar[int] = 40
    package = [0, 1]
    sub_package = [-1,0] # nome ruim, valores ruins
    places = ["threads", "core", "sockets"]
    proc_bind = ["true", "close", "spread", "false"]
