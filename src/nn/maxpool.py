import numpy as np

from .module import Module


class MaxPool(Module):
    def __init__(self, kernel_size: int = 2) -> None:
        """
        This class handles the forward and backward pass of the max pooling layer.

        Args:
            kernel_size (int, optional): The kernel size used to reduce the dimensions in the max pooling layer. Default is `2`.
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.mask = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        This method calculates the forward pass of the max pooling layer.

        Args:
            input (np.ndarray): The input feature matrix for this layer (shape = (batch_size, num_filters, height, width)).

        Returns:
            The reduced input matrix by the kernel size (shape = (batch_size, num_filters, height // kernel_size, width // kernel_size)).
        """
        batch_size, num_filters, height, width = input.shape

        pad_h_total = height % self.kernel_size
        self.pad_h = (int(pad_h_total / self.kernel_size), int(pad_h_total /
                                                               self.kernel_size) + pad_h_total % self.kernel_size)
        pad_w_total = width % self.kernel_size
        self.pad_w = (int(pad_w_total / self.kernel_size), int(pad_w_total /
                                                               self.kernel_size) + pad_w_total % self.kernel_size)

        input_padded = np.pad(input,
                              ((0, 0), (0, 0), self.pad_h, self.pad_w),
                              mode='constant', constant_values=-np.inf)
        self.padded_shape = input_padded.shape

        padded_h, padded_w = input_padded.shape[2], input_padded.shape[3]

        input_r = input_padded.reshape(batch_size, num_filters, padded_h // self.kernel_size,
                                       self.kernel_size, padded_w // self.kernel_size, self.kernel_size)

        output = np.max(input_r, axis=(3, 5))

        self.mask = (input_r == output[:, :, :, np.newaxis, :, np.newaxis])

        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        This method calculates the backward pass of the max pooling layer.

        Args:
            dout (np.ndarrray): The cotangent of the previous layers in the backward pass 
                                (shape = (batch_size, num_filters, height // kernel_size, width // kernel_size)).

        Returns:
            The cotangent of this layer (shape = (batch_size, num_filters, height, width)).

        Raises:
            RunTimeError: If the forward pass wasn't called before hand.
        """
        if self.mask is None:
            raise RuntimeError("The forward pass hasn't been executed prior.")

        dX_padded = (self.mask * dout[:, :, :, np.newaxis, :, np.newaxis]).reshape(self.padded_shape)

        ph, pw = self.padded_shape[2:]

        dX = dX_padded[:, :, self.pad_h[0]:ph - self.pad_h[1], self.pad_w[0]:pw - self.pad_w[1]]

        return dX
