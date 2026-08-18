from .machine import Machine


# Todo: Consultar essas informações programaticamente, sem necessidade de fazer estes arquivos
Glados = Machine(
    name="Glados",
    freq=[
        2200000,
        2800000,
        3300000,
    ],
    max_threads=16, ## mentira
    places=["threads", "cores", "sockets"],
    proc_bind=["true", "close", "spread", "false"],
)
