"""Dataset for 5G neural receiver."""

from torch.utils.data import Dataset


class NeuralReceiverDataset(Dataset):
    """Dataset for 5G neural receiver - to be implemented."""

    def __init__(self, num_samples: int = 10000, **kwargs):
        # TODO: Implement dataset with Sionna integration
        raise NotImplementedError("Dataset not yet implemented")

    def __len__(self):
        # TODO: Return dataset length
        raise NotImplementedError("__len__ not yet implemented")

    def __getitem__(self, idx):
        # TODO: Return (received_signal, bits) tuple
        raise NotImplementedError("__getitem__ not yet implemented")
