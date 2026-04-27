"""
Channel related processing
"""

import numpy as np


def generate_channel(num_samples, Nt, Nr, K, model='Rayleigh'):
    """
    生成信道矩阵
    
    Args:
        num_samples: 样本数
        Nt: 发射天线数
        Nr: 接收天线数
        K: 用户数
        model: 'Rayleigh' or 'Rician'
    
    Returns:
        H: 信道矩阵 (num_samples x Nr x Nt x K)
    """
    if model == 'Rayleigh':
        H = (1/np.sqrt(2)) * (
            np.random.randn(num_samples, Nr, Nt, K) + 
            1j * np.random.randn(num_samples, Nr, Nt, K)
        )
    
    elif model == 'Rician':
        K_factor = 10
        
        # 正确的LOS分量：基于天线相位偏移
        d_lambda = 0.5  # 半波长
        
        # 生成LOS相位矩阵
        phase_shifts = np.exp(1j * 2 * np.pi * d_lambda * np.arange(Nt))
        LOS_component = np.outer(np.ones(Nr), phase_shifts)  # (Nr x Nt)
        LOS_component = LOS_component / np.sqrt(Nr * Nt)  # 归一化
        
        # NLOS分量
        NLOS_component = (1/np.sqrt(2 * (K_factor + 1))) * (
            np.random.randn(num_samples, Nr, Nt, K) + 
            1j * np.random.randn(num_samples, Nr, Nt, K)
        )
        
        # Rician信道
        H = np.sqrt(K_factor / (K_factor + 1)) * LOS_component + \
            np.sqrt(1 / (K_factor + 1)) * NLOS_component
    
    else:
        raise ValueError("Unsupported channel model")
    
    return H


def channel_to_received_signal(H, F, P, X, noise_power):
    """
    计算接收信号
    
    Args:
        H: 信道 (Nr x Nt x K)
        F: 预编码矩阵 (Nt x S x K)
        P: 功率矩阵 (S x K)
        X: 发送信号 (S x K)
        noise_power: 噪声功率
    
    Returns:
        Y: 接收信号 (Nr x K)
    """
    Nr, Nt, K = H.shape
    S = X.shape[0]
    
    # 噪声
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(Nr, K) + 1j * np.random.randn(Nr, K)
    )
    
    Y = np.zeros((Nr, K), dtype=complex)
    
    for k in range(K):
        # Y_k = H_k @ F_k @ sqrt(P_k) @ X_k + noise
        F_k = F[:, :, k]  # (Nt x S)
        P_k = np.sqrt(P[:, k])  # (S,)
        D_k = np.diag(P_k)  # (S x S)
        X_k = X[:, k]  # (S,)
        
        Y[:, k] = H[:, :, k] @ F_k @ D_k @ X_k + noise[:, k]
    
    return Y


def obtain_precoding_matrix(H, S):
    """
    获取预编码矩阵
    
    Args:
        H: 信道 (Nr x Nt x K)
        S: 每用户数据流数
    
    Returns:
        F: 预编码矩阵 (Nt x S x K)
        W: 合并矩阵 (S x Nr x K)
        P: 功率矩阵 (S x K)
        Hp: 有效信道 (S x S x K)
    """
    from BlockDiagonalization import BD
    
    Nr, Nt, K = H.shape
    
    # BD预编码
    F, W, P = BD(H, S)
    
    # 有效信道
    Hp = np.zeros((S, S, K), dtype=complex)
    for k in range(K):
        Hp[:, :, k] = W[:, :, k] @ H[:, :, k] @ F[:, :, k] @ np.diag(np.sqrt(P[:, k]))
    
    return F, W, P, Hp