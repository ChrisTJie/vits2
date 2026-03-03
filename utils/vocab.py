from typing import List, Dict, Optional


class Vocab:
    def __init__(self, stoi: Dict[str, int], specials: Optional[List[str]] = None):
        self._stoi = stoi.copy()
        if specials:
            # Note: in torchtext, specials are usually prepended if they don't exist
            # but here we follow the order in the provided stoi or add them at the end
            for s in specials:
                if s not in self._stoi:
                    # In this project, indices are usually pre-defined in vocab.txt
                    # If a special token is missing, we add it with a new index
                    self._stoi[s] = max(self._stoi.values()) + 1 if self._stoi else 0

        self._itos = {v: k for k, v in self._stoi.items()}
        self._default_index = None

    def set_default_index(self, index: int):
        self._default_index = index

    def __call__(self, tokens: List[str]) -> List[int]:
        if self._default_index is not None:
            return [self._stoi.get(t, self._default_index) for t in tokens]
        return [self._stoi[t] for t in tokens]

    def get_stoi(self) -> Dict[str, int]:
        return self._stoi

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        return [self._itos.get(i, "<UNK>") for i in indices]

    def __len__(self):
        return len(self._stoi)


def build_vocab(stoi: Dict[str, int], specials: Optional[List[str]] = None) -> Vocab:
    return Vocab(stoi, specials=specials)
