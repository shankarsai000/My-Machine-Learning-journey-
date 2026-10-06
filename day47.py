# ============================================================
# DAY 47 — CONVOLUTIONAL NEURAL NETWORKS (CNNs)
# NumPy From Scratch → PyTorch
# ============================================================

import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 1. BASIC CNN DIMENSIONS
# ============================================================

def calculate_output_size(input_size, kernel_size, padding=0, stride=1):
    """
    Calculate spatial output size of a convolution.

    Formula:
        O = floor((N + 2P - K) / S) + 1
    """
    return ((input_size + 2 * padding - kernel_size) // stride) + 1


# Example:
input_size = 28
kernel_size = 3
padding = 1
stride = 1

output_size = calculate_output_size(
    input_size,
    kernel_size,
    padding,
    stride
)

print("=" * 60)
print("CNN OUTPUT SIZE")
print("=" * 60)
print(f"Input size  : {input_size} × {input_size}")
print(f"Kernel      : {kernel_size} × {kernel_size}")
print(f"Padding     : {padding}")
print(f"Stride      : {stride}")
print(f"Output size : {output_size} × {output_size}")


# ============================================================
# 2. CONVOLUTION FROM SCRATCH — SINGLE CHANNEL
# ============================================================

def conv2d_single_channel(
    image,
    kernel,
    stride=1,
    padding=0
):
    """
    Perform 2D convolution/cross-correlation on a single-channel image.

    image  : (H, W)
    kernel : (K, K)
    """

    # Add zero padding around the image
    if padding > 0:
        image = np.pad(
            image,
            ((padding, padding), (padding, padding)),
            mode="constant"
        )

    h, w = image.shape
    k = kernel.shape[0]

    # Calculate output dimensions
    output_h = ((h - k) // stride) + 1
    output_w = ((w - k) // stride) + 1

    output = np.zeros((output_h, output_w))

    # Slide the kernel over the image
    for i in range(0, output_h):
        for j in range(0, output_w):

            row_start = i * stride
            col_start = j * stride

            region = image[
                row_start:row_start + k,
                col_start:col_start + k
            ]

            # Element-wise multiplication + summation
            output[i, j] = np.sum(region * kernel)

    return output


# Small example image
image = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32)


# Vertical-edge detector
vertical_edge_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)


feature_map = conv2d_single_channel(
    image,
    vertical_edge_kernel
)

print("\n" + "=" * 60)
print("CONVOLUTION FROM SCRATCH")
print("=" * 60)
print("Input image:")
print(image)

print("\nKernel:")
print(vertical_edge_kernel)

print("\nFeature map:")
print(feature_map)


# ============================================================
# 3. MULTI-CHANNEL CONVOLUTION
# ============================================================

def conv2d_multi_channel(
    image,
    kernels,
    biases=None,
    stride=1,
    padding=0
):
    """
    Multi-channel convolution.

    image   : (C_in, H, W)
    kernels : (C_out, C_in, K, K)
    biases  : (C_out,)

    output  : (C_out, H_out, W_out)
    """

    c_in, h, w = image.shape
    c_out, kernel_c_in, k, _ = kernels.shape

    assert c_in == kernel_c_in, (
        "Number of input channels must match kernel channels."
    )

    # Pad every input channel
    if padding > 0:
        image = np.pad(
            image,
            (
                (0, 0),
                (padding, padding),
                (padding, padding)
            ),
            mode="constant"
        )

    _, padded_h, padded_w = image.shape

    output_h = ((padded_h - k) // stride) + 1
    output_w = ((padded_w - k) // stride) + 1

    output = np.zeros(
        (c_out, output_h, output_w),
        dtype=np.float32
    )

    if biases is None:
        biases = np.zeros(c_out)

    # Loop over filters
    for out_channel in range(c_out):

        # Slide filter over image
        for i in range(output_h):
            for j in range(output_w):

                row_start = i * stride
                col_start = j * stride

                # Extract region from ALL input channels
                region = image[
                    :,
                    row_start:row_start + k,
                    col_start:col_start + k
                ]

                # One scalar for this output location
                output[out_channel, i, j] = (
                    np.sum(
                        region * kernels[out_channel]
                    )
                    + biases[out_channel]
                )

    return output


# ============================================================
# 4. RGB IMAGE + MULTIPLE FILTERS
# ============================================================

# RGB image:
# Channels = 3
# Height   = 5
# Width    = 5

rgb_image = np.random.randn(
    3, 5, 5
).astype(np.float32)


# 4 filters
# Each filter = 3 input channels × 3 × 3 kernel
kernels = np.random.randn(
    4, 3, 3, 3
).astype(np.float32)

biases = np.zeros(4, dtype=np.float32)


multi_channel_output = conv2d_multi_channel(
    rgb_image,
    kernels,
    biases,
    stride=1,
    padding=1
)

print("\n" + "=" * 60)
print("MULTI-CHANNEL CONVOLUTION")
print("=" * 60)

print(f"Input shape   : {rgb_image.shape}")
print(f"Kernel shape  : {kernels.shape}")
print(f"Output shape  : {multi_channel_output.shape}")

# Expected:
# Input  = (3, 5, 5)
# Filters = 4
# Padding = 1
# Stride  = 1
# Output  = (4, 5, 5)


# ============================================================
# 5. PARAMETER COUNT
# ============================================================

def count_conv_parameters(
    in_channels,
    out_channels,
    kernel_size,
    bias=True
):
    """
    Count trainable parameters in a Conv2D layer.
    """

    weights = (
        in_channels
        * out_channels
        * kernel_size
        * kernel_size
    )

    bias_parameters = out_channels if bias else 0

    return weights + bias_parameters


parameter_count = count_conv_parameters(
    in_channels=3,
    out_channels=64,
    kernel_size=3
)

print("\n" + "=" * 60)
print("CONVOLUTION PARAMETER COUNT")
print("=" * 60)

print("Input channels : 3")
print("Filters        : 64")
print("Kernel         : 3 × 3")
print(f"Parameters     : {parameter_count}")

# 3 × 3 × 3 × 64 + 64 = 1,792


# ============================================================
# 6. MAX POOLING FROM SCRATCH
# ============================================================

def max_pool2d(
    feature_map,
    pool_size=2,
    stride=2
):
    """
    Max pooling for a single feature map.

    feature_map : (H, W)
    """

    h, w = feature_map.shape

    output_h = ((h - pool_size) // stride) + 1
    output_w = ((w - pool_size) // stride) + 1

    output = np.zeros(
        (output_h, output_w),
        dtype=feature_map.dtype
    )

    for i in range(output_h):
        for j in range(output_w):

            row_start = i * stride
            col_start = j * stride

            region = feature_map[
                row_start:row_start + pool_size,
                col_start:col_start + pool_size
            ]

            output[i, j] = np.max(region)

    return output


pool_input = np.array([
    [1, 5, 2, 4],
    [8, 3, 7, 6],
    [2, 9, 1, 5],
    [4, 6, 8, 2]
], dtype=np.float32)


pool_output = max_pool2d(
    pool_input,
    pool_size=2,
    stride=2
)

print("\n" + "=" * 60)
print("MAX POOLING")
print("=" * 60)

print("Input:")
print(pool_input)

print("\nAfter 2×2 MaxPool:")
print(pool_output)


# ============================================================
# 7. PYTORCH Conv2D
# ============================================================

torch_conv = nn.Conv2d(
    in_channels=3,
    out_channels=4,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True
)

print("\n" + "=" * 60)
print("PYTORCH Conv2D")
print("=" * 60)

print(torch_conv)

print(f"\nWeight shape : {tuple(torch_conv.weight.shape)}")
print(f"Bias shape   : {tuple(torch_conv.bias.shape)}")

pytorch_parameters = sum(
    parameter.numel()
    for parameter in torch_conv.parameters()
)

print(f"Total parameters : {pytorch_parameters}")


# ============================================================
# 8. PYTORCH FORWARD PASS
# ============================================================

# Batch of 2 RGB images
x = torch.randn(
    2,       # batch size
    3,       # RGB channels
    32,      # height
    32       # width
)

conv_output = torch_conv(x)

print("\n" + "=" * 60)
print("PYTORCH FORWARD PASS")
print("=" * 60)

print(f"Input shape  : {tuple(x.shape)}")
print(f"Output shape : {tuple(conv_output.shape)}")


# ============================================================
# 9. COMPLETE CNN ARCHITECTURE
# ============================================================

class SimpleCNN(nn.Module):
    """
    Small CNN demonstrating the standard image pipeline:

    Image
      ↓
    Conv2D
      ↓
    BatchNorm
      ↓
    ReLU
      ↓
    MaxPool
      ↓
    Conv2D
      ↓
    BatchNorm
      ↓
    ReLU
      ↓
    MaxPool
      ↓
    Flatten
      ↓
    Fully Connected
    """

    def __init__(self, num_classes=10):

        super().__init__()

        self.features = nn.Sequential(

            # 3 × 32 × 32
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            # Normalize channels
            nn.BatchNorm2d(32),

            # Non-linearity
            nn.ReLU(),

            # 32 × 32 × 32
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # 32 × 16 × 16
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 64 × 16 × 16
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(

            # 64 × 8 × 8 = 4096
            nn.Flatten(),

            nn.Linear(
                64 * 8 * 8,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# Create model
model = SimpleCNN(num_classes=10)

print("\n" + "=" * 60)
print("COMPLETE CNN")
print("=" * 60)

print(model)


# ============================================================
# 10. CNN FORWARD PASS
# ============================================================

dummy_images = torch.randn(
    4,      # batch
    3,      # RGB
    32,     # height
    32      # width
)

logits = model(dummy_images)

print("\n" + "=" * 60)
print("COMPLETE CNN FORWARD PASS")
print("=" * 60)

print(f"Input shape : {tuple(dummy_images.shape)}")
print(f"Logits shape: {tuple(logits.shape)}")

print("\nRaw logits:")
print(logits)


# ============================================================
# 11. PREDICTIONS
# ============================================================

predictions = torch.argmax(
    logits,
    dim=1
)

print("\nPredicted classes:")
print(predictions)


# ============================================================
# 12. INSPECT FEATURE MAP SHAPES
# ============================================================

with torch.no_grad():

    x = dummy_images

    print("\n" + "=" * 60)
    print("FEATURE MAP SHAPES")
    print("=" * 60)

    for layer in model.features:

        x = layer(x)

        print(
            f"{layer.__class__.__name__:15s}"
            f" → {tuple(x.shape)}"
        )


# ============================================================
# 13. MODEL PARAMETER COUNT
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\n" + "=" * 60)
print("MODEL PARAMETERS")
print("=" * 60)

print(f"Total parameters     : {total_parameters:,}")
print(f"Trainable parameters : {trainable_parameters:,}")


# ============================================================
# 14. CNN MENTAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("CNN PIPELINE")
print("=" * 60)

print(
    """
Image
  ↓
Conv2D
  ↓
Feature Maps
  ↓
BatchNorm
  ↓
ReLU
  ↓
MaxPool
  ↓
Conv2D
  ↓
Feature Maps
  ↓
BatchNorm
  ↓
ReLU
  ↓
MaxPool
  ↓
Flatten
  ↓
Linear
  ↓
Logits
  ↓
Prediction
"""
)

print("Day 47 complete! 🚀")