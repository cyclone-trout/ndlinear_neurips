
import torch
import torch.nn as nn
import torch.optim as optim


class NdLinear(nn.Module):
    def __init__(self, input_dims: tuple, hidden_size: tuple, transform_outer=True):
        """
        NdLinear: A PyTorch layer for projecting tensors into multi-space representations.
        
        Unlike conventional embedding layers that map into a single vector space, NdLinear 
        transforms tensors across a collection of vector spaces, capturing multivariate structure 
        and topical information that standard deep learning architectures typically lose.

        Args:
            input_dims (tuple): Shape of input tensor (excluding batch dimension).
            hidden_size (tuple): Target hidden dimensions after transformation.
        """
        super(NdLinear, self).__init__()

        if len(input_dims) != len(hidden_size):
            raise Exception("Input shape and hidden shape do not match.")

        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.num_layers = len(input_dims)  # Must match since dims are equal
        # self.relu = nn.ReLU() # self.relu is not being used.
        self.transform_outer = transform_outer

        # Define transformation layers per dimension
        self.align_layers = nn.ModuleList([
            nn.Linear(input_dims[i], hidden_size[i]) for i in range(self.num_layers)
        ])


    def forward(self, X):
        """
        Forward pass to project input tensor into a new multi-space representation.
        - Incrementally transposes, flattens, applies linear layers, and restores shape.

        Expected Input Shape: [batch_size, *input_dims]
        Output Shape: [batch_size, *hidden_size]

        Args:
            X (torch.Tensor): Input tensor with shape [batch_size, *input_dims]

        Returns:
            torch.Tensor: Output tensor with shape [batch_size, *hidden_size]
        """
        num_transforms = self.num_layers  # Number of transformations
        
        # Define iteration order
        # transform_indices = range(num_transforms) if transform_outer else reversed(range(num_transforms))

        for i in range(num_transforms):
            if self.transform_outer:
                layer = self.align_layers[i]
                transpose_dim = i + 1
            else:
                layer = self.align_layers[num_transforms - (i+1)]
                transpose_dim = num_transforms - i

            # Transpose the selected dimension to the last position
            X = torch.transpose(X, transpose_dim, num_transforms).contiguous()

            # Store original shape before transformation
            X_size = X.shape[:-1]

            # Flatten everything except the last dimension
            X = X.view(-1, X.shape[-1])

            # Apply transformation
            X = layer(X)
            
            # Reshape back to the original spatial structure (with new embedding dim)
            X = X.view(*X_size, X.shape[-1])

            # Transpose the dimension back to its original position
            X = torch.transpose(X, transpose_dim, num_transforms).contiguous()

        return X