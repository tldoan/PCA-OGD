
from dataloaders.wrapper import Storage

class Memory(Storage):
    def reduce(self, m):
        self.storage = self.storage[:m]