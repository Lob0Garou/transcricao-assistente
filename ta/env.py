"""Prepara o ambiente: dependências locais em .deps e DLLs CUDA para a GPU.

Aceita uma instalação normal/venv e, opcionalmente, pacotes em .deps.
"""
import glob
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS = os.path.join(RAIZ, ".deps")
_DLLS = []  # Mantém os diretórios registrados durante a execução.


def preparar():
    if os.path.isdir(DEPS) and DEPS not in sys.path:
        sys.path.insert(0, DEPS)
    if os.name == "nt":
        for base in list(sys.path):
            for d in glob.glob(os.path.join(base, "nvidia", "*", "bin")):
                _DLLS.append(os.add_dll_directory(d))
                os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
