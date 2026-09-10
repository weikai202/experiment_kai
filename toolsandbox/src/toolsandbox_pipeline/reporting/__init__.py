"""Offline-safe final evaluation and reporting package.

Public implementations are imported from their concrete modules.  Keeping this
initializer inert prevents provider/accounting imports from creating cycles.
"""

__all__: list[str] = []
