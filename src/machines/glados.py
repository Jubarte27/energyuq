from typing import ClassVar
from .machine import Machine


# Todo: Consultar essas informações programaticamente, sem necessidade de fazer estes arquivos
class Glados(Machine):
    name: ClassVar[str] = "Glados"
    freq: ClassVar[list[int]] = [
        2200000,
        2800000,
        3300000,
    ]
    max_threads: ClassVar[int] = 16 ## mentira
    places = ["threads", "cores", "sockets"]
    proc_bind = ["true", "close", "spread", "false"]
