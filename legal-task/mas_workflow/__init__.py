from mas.mas import MetaMAS

from .macnet.graph_mas import MacNet

MAS = {
    'macnet': MacNet,
}


def get_mas(mas_type: str) -> MetaMAS:
    if MAS.get(mas_type) is None:
        raise ValueError('Unsupported mas type for legal-task.')
    return MAS.get(mas_type)()