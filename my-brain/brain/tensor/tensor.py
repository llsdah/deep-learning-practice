class Tensor:

    def __init__(
        self,
        data,
        shape
    ):
        self.data = data
        self.shape = shape
        self.grad = None