from .machine import Machine


class Glados(Machine):

    @staticmethod
    def name() -> str:
        return "Glados"

    @staticmethod
    def freq() -> list[int]:
        return [
            2200000,
            2800000,
            3300000,
        ]
