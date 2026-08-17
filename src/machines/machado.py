from .machine import Machine


# Todo: Consultar essas informações programaticamente, sem necessidade de fazer estes arquivos
Machado = Machine(
    name="Machado",
    freq=[
        1400000,
        1700000,
        3700000,
    ],
    max_threads=12,
    places=["threads", "cores", "sockets"],
    proc_bind=["true", "close", "spread", "false"],
)
