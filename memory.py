from collections import deque
import random
import numpy as np


class Memory(object):
    def __init__(self, buffer_size: int) -> None:
        """Init curiosity memory(deque) with maximum size buffer_size.
        Args:
            buffer_size: memory size

        """
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size: int) -> list:
        """Randomly sample batch_size examples.
        Args:
            batch_size: memory sample size

        """
        return random.sample(self.buffer, batch_size)

    def add(self, action: np.ndarray):
        """Add exploration into memory.
        Args:
            action: exploration

        """
        experience = action
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            # When the memory is full, pop up the first exploration.
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self) -> int:
        """Return the number of exploration.
        Return: num_experiences

        """
        return self.num_experiences
