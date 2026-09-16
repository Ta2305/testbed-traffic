"""
Quantization utilities for efficient WAFL model exchange.

Sign and magnitude are packed together in the same bytes, avoiding a
separate sign array:
- Q=8: 1-bit sign + 7-bit magnitude in 1 byte
- Q=4: 1-bit sign + 3-bit magnitude = 4 bits, 2 values per byte
- Q=2: 1-bit sign + 1-bit magnitude = 2 bits, 4 values per byte
- Q=1: Sign only, 8 values per byte
- Q=16: 1-bit sign + 15-bit magnitude in 2 bytes
- Q=32 or Q=0: No quantization (float32)

Provides:
- Combined sign-magnitude quantization
- Efficient bit-packing
- Bitmap encoding/decoding for sparse indices
"""

from typing import Any, Dict

import numpy as np
import torch


class Quantizer3:
    """
    Optimized quantizer with combined sign-magnitude packing.
    Supports 0 (no quantization), 1, 2, 4, 8, 16, and 32-bit modes.
    """

    def __init__(self, bits: int):
        """
        Initialize the quantizer.

        Args:
            bits: Number of bits for quantization (0, 1, 2, 4, 8, 16, or 32)
                  0 or 32 means no quantization (float32)
        """
        if bits not in [0, 1, 2, 4, 8, 16, 32]:
            raise ValueError(f"Unsupported bit width: {bits}. Must be 0, 1, 2, 4, 8, 16, or 32.")

        self.bits = bits

        # Calculate magnitude bits (total bits - 1 for sign, except special cases)
        if bits == 0 or bits == 32:
            self.magnitude_bits = 32  # float32, no quantization
            self.qmax = 0  # Not used
        elif bits == 1:
            self.magnitude_bits = 0  # Sign only
            self.qmax = 0
        else:
            self.magnitude_bits = bits - 1
            self.qmax = (2 ** self.magnitude_bits) - 1

    def quantize(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Quantize a tensor with combined sign-magnitude packing.

        Args:
            tensor: Input tensor (CPU, float32)

        Returns:
            Dict containing packed_data, scale, zero_point, shape, bits, numel
        """
        data = tensor.detach().cpu().numpy().astype(np.float32)
        original_shape = list(tensor.shape)
        numel = int(np.prod(original_shape))

        # Handle no quantization case (Q=0 or Q=32)
        if self.bits == 0 or self.bits == 32:
            return {
                "packed_data": data.tobytes(),
                "scale": 1.0,
                "zero_point": 0.0,
                "shape": original_shape,
                "bits": self.bits,
                "numel": numel,
                "all_zero": False
            }

        # Extract signs and absolute values
        signs = np.sign(data)
        abs_data = np.abs(data)

        # Handle all-zero case
        if np.all(data == 0):
            return {
                "packed_data": np.zeros(max(1, (numel * self.bits + 7) // 8), dtype=np.uint8).tobytes(),
                "scale": 1.0,
                "zero_point": 0.0,
                "shape": original_shape,
                "bits": self.bits,
                "numel": numel,
                "all_zero": True
            }

        # Get min/max of absolute values
        max_abs = np.max(abs_data)
        nonzero_mask = data != 0
        if np.any(nonzero_mask):
            min_abs = np.min(abs_data[nonzero_mask])
        else:
            min_abs = max_abs

        # Q=1: Sign only, use mean_abs for reconstruction
        if self.bits == 1:
            mean_abs = np.mean(abs_data[nonzero_mask]) if np.any(nonzero_mask) else 0.0

            # Pack signs: 1 = positive/zero, 0 = negative
            sign_bits = (signs.flatten() >= 0).astype(np.uint8)
            packed_data = np.packbits(sign_bits)

            return {
                "packed_data": packed_data.tobytes(),
                "scale": float(mean_abs),
                "zero_point": 0.0,
                "shape": original_shape,
                "bits": self.bits,
                "numel": numel,
                "all_zero": False
            }

        # Q >= 2: Combined sign-magnitude packing
        if max_abs == min_abs:
            step = 1.0
        else:
            step = (max_abs - min_abs) / self.qmax

        # Quantize magnitudes
        q_magnitudes = np.round((abs_data - min_abs) / step)
        q_magnitudes = np.clip(q_magnitudes, 0, self.qmax).astype(np.uint32)

        # Get sign bits (1 = positive/zero, 0 = negative)
        sign_bits = (signs.flatten() >= 0).astype(np.uint32)

        # Combine sign and magnitude
        # Format: [sign_bit][magnitude_bits]
        combined = (sign_bits << self.magnitude_bits) | q_magnitudes.flatten()

        # Pack according to bit width
        if self.bits == 16:
            # 2 bytes per value (big-endian)
            packed_data = combined.astype(np.uint16).tobytes()
        elif self.bits == 8:
            # 1 byte per value
            packed_data = combined.astype(np.uint8).tobytes()
        elif self.bits == 4:
            # 2 values per byte
            flat = combined.astype(np.uint8)
            if flat.size % 2 != 0:
                flat = np.pad(flat, (0, 1), 'constant')
            reshaped = flat.reshape(-1, 2)
            packed = ((reshaped[:, 0] << 4) | reshaped[:, 1]).astype(np.uint8)
            packed_data = packed.tobytes()
        elif self.bits == 2:
            # 4 values per byte
            flat = combined.astype(np.uint8)
            pad_len = (4 - flat.size % 4) % 4
            if pad_len > 0:
                flat = np.pad(flat, (0, pad_len), 'constant')
            reshaped = flat.reshape(-1, 4)
            packed = ((reshaped[:, 0] << 6) | (reshaped[:, 1] << 4) |
                      (reshaped[:, 2] << 2) | reshaped[:, 3]).astype(np.uint8)
            packed_data = packed.tobytes()
        else:
            packed_data = combined.astype(np.uint8).tobytes()

        return {
            "packed_data": packed_data,
            "scale": float(step),
            "zero_point": float(min_abs),
            "shape": original_shape,
            "bits": self.bits,
            "numel": numel,
            "all_zero": False
        }

    def dequantize(self, packed_bytes: bytes, scale: float, zero_point: float,
                   numel: int, all_zero: bool = False) -> torch.Tensor:
        """
        Dequantize packed bytes back to a tensor.

        Args:
            packed_bytes: Packed quantized data (combined sign-magnitude)
            scale: step (for Q>=2) or mean_abs (for Q==1)
            zero_point: min_abs
            numel: Number of elements in original tensor
            all_zero: Flag indicating if original tensor was all zeros

        Returns:
            Dequantized tensor (1D, float32)
        """
        if all_zero:
            return torch.zeros(numel)

        # No quantization case
        if self.bits == 0 or self.bits == 32:
            data = np.frombuffer(packed_bytes, dtype=np.float32)[:numel]
            return torch.from_numpy(data.copy())

        # Q=1: Sign only
        if self.bits == 1:
            unpacked_signs = np.unpackbits(np.frombuffer(packed_bytes, dtype=np.uint8))[:numel]
            signs = np.where(unpacked_signs, 1.0, -1.0).astype(np.float32)
            dequantized = signs * scale  # scale contains mean_abs
            return torch.from_numpy(dequantized)

        # Unpack based on bit width
        packed_data = np.frombuffer(packed_bytes, dtype=np.uint8)

        if self.bits == 16:
            combined = np.frombuffer(packed_bytes, dtype=np.uint16)[:numel]
        elif self.bits == 8:
            combined = packed_data[:numel]
        elif self.bits == 4:
            val0 = (packed_data >> 4) & 0x0F
            val1 = packed_data & 0x0F
            combined = np.stack([val0, val1], axis=1).flatten()[:numel]
        elif self.bits == 2:
            val0 = (packed_data >> 6) & 0x03
            val1 = (packed_data >> 4) & 0x03
            val2 = (packed_data >> 2) & 0x03
            val3 = packed_data & 0x03
            combined = np.stack([val0, val1, val2, val3], axis=1).flatten()[:numel]
        else:
            combined = packed_data[:numel]

        combined = combined.astype(np.uint32)

        # Extract sign and magnitude
        magnitude_mask = (1 << self.magnitude_bits) - 1
        q_magnitudes = combined & magnitude_mask
        sign_bits = (combined >> self.magnitude_bits) & 1

        # Convert to float
        signs = np.where(sign_bits, 1.0, -1.0).astype(np.float32)
        abs_values = q_magnitudes.astype(np.float32) * scale + zero_point

        dequantized = signs * abs_values
        return torch.from_numpy(dequantized)

    @staticmethod
    def encode_bitmap(mask: torch.Tensor) -> bytes:
        """
        Encode a boolean mask tensor to packed bytes.
        """
        return np.packbits(mask.cpu().numpy().flatten()).tobytes()

    @staticmethod
    def decode_bitmap(bitmap_bytes: bytes, numel: int) -> torch.Tensor:
        """
        Decode packed bytes back to a boolean mask tensor.
        """
        unpacked = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
        return torch.from_numpy(unpacked[:numel].astype(bool))


def compute_topk_threshold(diff_dict: Dict[str, torch.Tensor], K: float) -> float:
    """
    Compute the threshold for top-K% sparsification across all layers.

    Args:
        diff_dict: Dictionary of layer name -> differential tensor
        K: Sparsification ratio (0.0 to 1.0, e.g., 0.1 = top 10%)

    Returns:
        Threshold value (parameters with |value| >= threshold are kept)
    """
    if K >= 1.0:
        return 0.0  # Keep all parameters
    if K <= 0.0:
        return float('inf')  # Keep no parameters

    # Concatenate absolute values of all parameters
    all_params = torch.cat([v.view(-1).abs() for v in diff_dict.values()])
    numel = all_params.numel()
    k_num = max(1, int(numel * K))

    # Find the k-th largest value
    values, _ = torch.topk(all_params, k_num)
    return float(values[-1])
