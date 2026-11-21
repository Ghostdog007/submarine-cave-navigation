"""Route memory for the navigator: retention (multi-scale decaying attention) and junction (position-keyed) memory.

Kept free of PyTorch at import time so the torch-free env workers can import submarine.* safely; import
`submarine.memory.modules` / `submarine.memory.policies` explicitly where torch is wanted.
"""
