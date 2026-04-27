"""
Transmitter side of MU-MIO systems, including:
- Bit generation: Randomly generate bits for each user based on the modulation scheme.
- Modulation: Map the generated bits to symbols based on the specified modulation scheme (e.g., QPSK, 16-QAM, 64-QAM).
- Precoding: Implement various precoding techniques (e.g., BD) to mitigate inter-user interference and improve signal quality at the receivers.
- Power allocation: Allocate power to different data streams based on channel conditions and system requirements.
"""

import numpy as np
from constellation import constellation

# generate bit sequence for modulation

def generate_bits(modulation, num_samples, K, S):

    """
    generate bit sequences of MU-MIMO, support 2 types of input.
    Input:
    - modulation: modulation scheme
    - num_samples: number of OFDM symbols
    - K: number of users
    - number of data streams of each user
    Output:
    - bits: num_samples x S x K x number of bits per symbol
    """

    if modulation == 'QPSK' or modulation == 2:

        bits = np.random.randint(0, 2, size=(num_samples, S, K, 2)) # Generate random bits for modulation (2 bits per symbol for QPSK)

    elif modulation == '16QAM' or modulation == 4:

        bits = np.random.randint(0, 2, size=(num_samples, S, K, 4)) # Generate random bits for modulation (4 bits per symbol for 16-QAM)

    elif modulation == '64QAM' or modulation == 6:

        bits = np.random.randint(0, 2, size=(num_samples, S, K, 6)) # Generate random bits for modulation (6 bits per symbol for 64-QAM)

    else:

        raise ValueError("Unsupported modulation scheme. Choose 'QPSK', '16QAM', or '64QAM'.")

    return bits





def modulate(bits, modulation):

    """
    Modulate the input bits into symbols based on the modulation scheme.
    Input: 
    - bits: (num_samples x S x K x num_bits of per symbol). 
    - modulation: modulation scheme, e.g., QPSK, 16QAM, 64QAM
    - constellation_points: constellation_points corresponding to modulation
    Output: 
    - symbols: (num_samples x S x K), each symbol in the range of constellation points. 
    """

    if modulation == 'QPSK' or modulation == 2:
        # Map 2 bits to 1 symbol
        if bits.shape[3] != 2:
            raise ValueError("Number of bits must be a multiple of 2 for QPSK modulation.")
        constellation_points = constellation(modulation)
        symbol_indices = bits[:, :, :, 0] * 2 + bits[:, :, :, 1]  # Convert bits to symbol indices
        symbols = constellation_points[symbol_indices]  # Map symbol indices to constellation points

    elif modulation == '16QAM' or modulation == 4:
        # Map 4 bits to 1 symbol
        if bits.shape[3] != 4:
            raise ValueError("Number of bits must be a multiple of 4 for 16-QAM modulation.")
        constellation_points = constellation(modulation)

        symbol_indices = bits[:, :, :, 0] * 8 + bits[:, :, :, 1] * 4 + bits[:, :, :, 2] * 2 + bits[:, :, :, 3]  # Convert bits to symbol
        symbols = constellation_points[symbol_indices]  # Map symbol indices to constellation points

    elif modulation == '64QAM' or modulation == 6:
        if bits.shape[3] != 6:
            raise ValueError("Number of bits must be a multiple of 6 for 64-QAM modulation.")
        constellation_points = constellation(modulation)
        symbol_indices = (bits[:, :, :, 0] * 32 + bits[:, :, :, 1] * 16 + bits[:, :, :, 2] * 8 +
                            bits[:, :, :, 3] * 4 + bits[:, :, :, 4] * 2 + bits[:, :, :, 5])  # Convert bits to symbol indices
        symbols = constellation_points[symbol_indices]  # Map symbol indices to constellation points

    else:
        raise ValueError("Unsupported modulation scheme. Choose 'QPSK', '16QAM', or '64QAM'.")
    
    return symbols


def demodulate(X_hat, constellation_points):
    """
    Decode the detected symbols back to bits based on the modulation scheme.
    Input:
    - X_hat: Detected symbol matrix of shape (num_samples x S x K), where each column represents the detected symbols for a specific user.
    - constellation_points: The constellation points for the modulation scheme.
    Output:
    - bits_hat: Decoded bit matrix of shape (num_samples x S x K x num_bits), where each column represents the decoded bits for a specific user.
    """
    # 获取输入形状
    num_samples, S, K = X_hat.shape
    num_constellation_points = len(constellation_points)
    
    # 处理边界情况：如果星座点为空，无法解调
    if num_constellation_points == 0:
        raise ValueError("Constellation points cannot be empty")
        
    num_bits_per_symbol = int(np.log2(num_constellation_points))
    
    # 验证 log2 结果是否为整数，避免浮点误差导致的问题
    if 2**num_bits_per_symbol != num_constellation_points:
        raise ValueError("Number of constellation points must be a power of 2")

    # 1. 向量化计算最近邻索引
    # X_hat shape: (num_samples, S, K)
    # constellation_points shape: (M,)
    # 我们需要计算每个 X_hat 元素到每个 constellation_point 的距离
    
    # 重塑 X_hat 以便广播: (num_samples, S, K, 1)
    # 维度为1，就可以广播，其实就是把为1的维度扩展为和另一个数组相同维度的元素，所谓扩展就是复制。
    X_expanded = X_hat[:, :, :, np.newaxis]
    
    # 重塑 constellation_points 以便广播: (1, 1, 1, M)
    C_expanded = constellation_points[np.newaxis, np.newaxis, np.newaxis, :]
    
    # 计算距离: (num_samples, S, K, M)
    distances = np.abs(X_expanded - C_expanded)
    
    # 找到最小距离的索引: (num_samples, S, K)
    closest_indices = np.argmin(distances, axis=-1)
    
    # 2. 将索引转换为比特
    # 原代码使用 np.binary_repr，它是 MSB-first (最高位在左边/索引0)
    # 例如: index=1, width=2 -> "01" -> [0, 1]
    # 我们可以使用位运算来向量化这个过程
    
    # 创建一个数组保存比特 (num_samples, S, K, num_bits_per_symbol)
    bits_hat = np.zeros((num_samples, S, K, num_bits_per_symbol), dtype=int)
    
    # 对于每一个比特位置 i (从最高位 MSB 到最低位 LSB)
    # MSB 对应的是 2^(num_bits-1)，LSB 对应的是 2^0
    # 在原代码中: binary_repr 生成的字符串第0个字符是 MSB
    # 所以 bits_hat[..., 0] 应该是 MSB
    
    for i in range(num_bits_per_symbol):
        # 计算当前位的权重 (MSB first)
        # i=0 -> shift = num_bits - 1
        # i=1 -> shift = num_bits - 2
        shift = num_bits_per_symbol - 1 - i
        # 提取该比特: (index >> shift) & 1
        bits_hat[..., i] = (closest_indices >> shift) & 1

    return bits_hat

def test_transmitter():

    """

    test transmitter, including:
    - bit generation module: generate bit sequences for MU-MIMO
    - modulation module: modulate bits to modulated symbol
    """

    K = 4  # number of users

    S = 2  # number of data streams of each user

    num_samples = 100  # number of data symbols

    modulation = 4  # "16QAM"

    constellation_points = constellation(modulation)

    bits = generate_bits(modulation, num_samples, K, S)

    print("bits' shape:", bits.shape)

    symbols = modulate(bits, modulation)

    print("symbols' shape:", symbols.shape)

    decoded_bits = demodulate(symbols, constellation_points)

    from main import biterror
    ber = biterror(bits, decoded_bits)
    print("BER:", ber)





if __name__ == "__main__":

    test_transmitter()
