import numpy as np, torch
from torch.utils.data import Dataset
from common import make_3ch, normalise

class CycloneSet(Dataset):
    def __init__(self, X, y, norm, train=False):
        self.X, self.y, self.norm, self.train = X, y, norm, train
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        x = make_3ch(self.X[i].astype(np.float32))
        if self.train: x = self._augment(x)
        x = normalise(x, self.norm)
        return torch.from_numpy(x), torch.tensor(float(self.y[i]))
    @staticmethod
    def _augment(x):
        # NO flips/90-degree rotations: cyclones spin counter-clockwise in the Northern Hemisphere,
        # a mirrored image would show a spin direction that never occurs -> would teach wrong physics.
        dx, dy = np.random.randint(-16, 17, 2)                    # small shift (cyclone not always centred)
        x = np.roll(x, (dy, dx), axis=(1, 2))
        x[:2] *= np.random.uniform(0.97, 1.03)                    # tiny calibration difference between satellites
        x[:2] += np.random.uniform(-10, 10)                       # tiny offset (counts)
        x[2] = x[0] - x[1]                                        # keep diff channel consistent
        return x
