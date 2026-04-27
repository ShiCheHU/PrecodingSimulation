"""
MU-MIMO system with BD precoding
"""

import numpy as np
import matplotlib.pyplot as plt

from transmitter import generate_bits, modulate, demodulate
from channel import generate_channel, channel_to_received_signal, obtain_precoding_matrix
from receiver import receiver_combining, signal_detection
from constellation import constellation


import numpy as np

def biterror(transmitted_bits, received_bits):
    """计算BER
    Args:
    - transmitted_bits: 传输的比特流 (num_samples x S x K x num_bits)
    - received_bits: 接收到的比特流 (num_samples x S x K x num_bits)
    Returns:
    - ber: 误码率 (1)
    """
    # 检查输入是否为空
    if transmitted_bits.size == 0 or received_bits.size == 0:
        raise ValueError("输入数组不能为空")
    
    # 检查两个数组的形状是否一致，避免广播导致的意外行为
    if transmitted_bits.shape != received_bits.shape:
        raise ValueError(f"输入数组形状不匹配: {transmitted_bits.shape} vs {received_bits.shape}")
    
    # 计算错误的比特数
    num_errors = np.sum(transmitted_bits != received_bits)
    
    # 计算总比特数，使用 shape 属性安全地获取各维度大小
    total_bits = transmitted_bits.shape[0] * transmitted_bits.shape[1] * transmitted_bits.shape[2] * transmitted_bits.shape[3]
    
    # 防止除零（虽然前面检查了 size，但为了双重保险）
    if total_bits == 0:
        return 0.0
        
    return float(num_errors) / total_bits

def run_simulation(mod='64QAM', method='ZF', num_samples=2000):
    """
    Run the MU-MIMO BER simulation for a given modulation and detector.

    Args:
        mod: Modulation scheme, e.g. 'QPSK', '16QAM', '64QAM'.
        method: Detection method, 'ZF' or 'MMSE'.
        num_samples: Number of transmitted symbol samples.

    Returns:
        SNR_list: Evaluated SNR points in dB.
        results: BER values corresponding to SNR_list.
    """
    Nt = 32
    Nr = 4
    S = 2
    K = 4

    SNR_list = np.arange(-10, 20, 5)

    const_points = constellation(mod)
    bits = generate_bits(mod, num_samples, K, S)
    symbols = modulate(bits, mod)
    channels = generate_channel(num_samples, Nt, Nr, K, 'Rayleigh')

    results = []

    print("Start simulation...")
    print(f"Configuration: Nt={Nt}, Nr={Nr}, K={K}, S={S}, modulation={mod}, detector={method}")
    print("-" * 70)

    for snr_db in SNR_list:
        noise_power = 10 ** (-snr_db / 10)
        x_hats = []

        for n in range(num_samples):
            H = channels[n, :, :, :]
            F, W, P, Hp = obtain_precoding_matrix(H, S)
            Y = channel_to_received_signal(H, F, P, symbols[n, :, :], noise_power)
            Z = receiver_combining(Y, W)
            x_hat = signal_detection(Z, Hp, method=method, noise_power=noise_power)
            x_hats.append(x_hat)

        rx_bits = demodulate(np.asarray(x_hats), const_points)
        ber = biterror(bits, rx_bits)
        results.append(ber)
        print(f"SNR = {snr_db:>3} dB: BER = {ber:.6e}")

    # plt.figure(figsize=(10, 6))
    # plt.semilogy(SNR_list, results, 'b-o', linewidth=2, markersize=6)
    # plt.grid(True, which='both', alpha=0.3)
    # plt.xlabel('SNR (dB)', fontsize=12)
    # plt.ylabel('BER', fontsize=12)
    # plt.title(f'MU-MIMO with BD Precoding ({mod}, {method})', fontsize=14)
    # out_name = f'bd_ber_{mod}_{Nt}T_{Nr}R_{S}S_{K}K_{method}.png'
    # plt.savefig(out_name, dpi=150, bbox_inches='tight')
    # plt.show()

    return SNR_list, results


if __name__ == "__main__":
    np.random.seed(42)

    res = {}

    mods = ['QPSK', '16QAM', '64QAM']
    methods = ['ZF', 'MMSE']

    for mod in mods:
        for method in methods:
            SNR_list, result = run_simulation(mod, method)
            res[f'{mod}_{method}'] = (SNR_list, result)

    plt.figure(figsize=(10, 6))

    markers = {
        'QPSK_ZF': 'o-',
        'QPSK_MMSE': 's-',
        '16QAM_ZF': '^-',
        '16QAM_MMSE': 'd-',
        '64QAM_ZF': 'x-',
        '64QAM_MMSE': '*-'
    }

    for mod in mods:
        for method in methods:
            key = f'{mod}_{method}'
            SNR_list, result = res[key]
            plt.semilogy(SNR_list, result, markers[key], linewidth=2, markersize=6, label=key)

    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('BER', fontsize=12)
    plt.title('MU-MIMO with BD Precoding', fontsize=14)
    plt.legend()
    plt.tight_layout()

    out_name = 'bd_ber_all_mods_detectors.png'
    plt.savefig(out_name, dpi=150, bbox_inches='tight')
    plt.show()