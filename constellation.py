"""
Constellation: generate constellation points for different modulation schemes
"""

import numpy as np


def constellation(modulation):
    """
    返回归一化星座点, with dimension 2**modulation
    支持: QPSK, 16-QAM, 64-QAM
    """
    if modulation == 'QPSK' or modulation == 2:
        # 按格雷码顺序: 00,01,11,10
        constellation_points = np.array([
            (1 + 1j) / np.sqrt(2),   # 00
            (1 - 1j) / np.sqrt(2),   # 01
            (-1 - 1j) / np.sqrt(2),   # 11
            (-1 + 1j) / np.sqrt(2)    # 10
        ])
    
    elif modulation == '16QAM' or modulation == 4:
        # 16-QAM: 按格雷码排列
        real = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        imag = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        constellation_points = np.array([r + 1j*c for r in real for c in imag])
    
    elif modulation == '64QAM' or modulation == 6:
        real = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42)
        imag = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42)
        constellation_points = np.array([r + 1j*c for r in real for c in imag])
    
    else:
        raise ValueError("Unsupported modulation scheme")
    
    return constellation_points


def get_bits_per_symbol(modulation):
    """获取每符号比特数"""
    if modulation == 'QPSK' or modulation == 2:
        return 2
    elif modulation == '16QAM' or modulation == 4:
        return 4
    elif modulation == '64QAM' or modulation == 6:
        return 6
    else:
        raise ValueError("Unsupported modulation")