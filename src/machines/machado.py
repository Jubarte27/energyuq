from typing import ClassVar
from .machine import Machine


# Todo: Consultar essas informações programaticamente, sem necessidade de fazer estes arquivos
class Machado(Machine):
    name: ClassVar[str] = "Machado"
    freq: ClassVar[list[int]] = [
        1400000,
        1700000,
        3700000,
    ]
    max_threads: ClassVar[int] = 12
    places = ["threads", "cores", "sockets"]
    proc_bind = ["true", "close", "spread", "false"]
