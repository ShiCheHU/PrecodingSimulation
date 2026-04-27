"""
Receiver related processing, including:
- Receiver combining: Implement various receiver combining techniques (e.g., MMSE, ZF) to enhance the received signal quality and mitigate interference.
- Signal detection: Implement signal detection algorithms (e.g., ML, ZF) to recover the transmitted symbols from the received signals.
- Decoding: Map the detected symbols back to bits based on the modulation scheme.
- Performance evaluation: Evaluate BER.
"""

import numpy as np

def receiver_combining(Y, W):
    """
    Perform receiver combining to enhance the received signal quality.
    Input:
    - Y: Received signal matrix of shape (Nr x K), where each column represents the received signal for a specific user.
    - W: Combining matrix of shape (S x Nr x K), where S is the number of data streams per user.
    Output:
    - Z: Combined signal matrix of shape (S x K), where each column represents the combined signal for a specific user.
    """
    Nr, K = Y.shape
    S = W.shape[0]
    
    Z = np.zeros((S, K), dtype=complex)
    
    for k in range(K):
        W_k = W[:, :, k]  # Combining matrix for user k, with shape: (S x Nr)
        Y_k = Y[:, k]     # Received signal for user k, with shape: (Nr,)
        Z[:, k] = W_k @ Y_k  # Combined signal for user k, with shape: (S,)
    
    return Z

def signal_detection(Z, H_eff, method='ZF', noise_power=0.01):
    """
    Get the modulated symbols from the combined signal using the specified detection method.
    Input:
    - Z: Combined signal matrix of shape (S x K), where each column represents the combined signal for a specific user.
    - H_eff: effective channel of shape (S x S x K), s.t., Z = H_eff x X_hat + n_eff.
    - method: Detection method to use ('ZF' for Zero-Forcing, 'MMSE' for Minimum Mean Square Error).
    - noise_power: Power of the additive noise.
    Output:
    - X_hat: Detected symbol matrix of shape (S x K), where each column represents the detected symbols for a specific user.
    """
    S, K = Z.shape
    X_hat = np.zeros((S, K), dtype=complex)
    
    for k in range(K):
        Z_k = Z[:, k]  # Combined signal for user k, with shape: (S,)
        H_eff_k = H_eff[:, :, k]
        if method == 'ZF':
            Combiner_k = np.linalg.pinv(H_eff_k)  # Zero-Forcing detection
            X_hat[:, k] = Combiner_k @ Z_k
        elif method == 'MMSE':
            Combiner_k = np.linalg.inv(H_eff_k.conj().T @ H_eff_k + noise_power * np.eye(S)) @ H_eff_k.conj().T  # MMSE detection
            X_hat[:, k] = Combiner_k @ Z_k
        elif method is None:
            return Z
        else:
            raise ValueError("Unsupported detection method. Choose 'ZF' or 'MMSE'.")
    
    return X_hat

def test_receiver_processing():
    # Example parameters
    Nr = 4 # Number of receiver antennas
    S = 2  # Number of data streams per user
    K = 4  # Number of users
    
    # Simulate received signal Y and combining matrix W
    Y = np.random.randn(Nr, K) + 1j * np.random.randn(Nr, K)  # Received signal (Nr x K)
    W = np.random.randn(S, Nr, K) + 1j * np.random.randn(S, Nr, K)  # Combining matrix (S x Nr x K)
    Hp = np.random.randn(S, S, K) + 1j * np.random.randn(S, S, K)  # effective channel matrix before precoding module (S, K) 
    # Perform receiver combining
    Z = receiver_combining(Y, W)
    
    # Perform signal detection
    print("Implement ZF receiver:")
    X_hat = signal_detection(Z, Hp, method='ZF', noise_power=0.01)
    print("x_hat shpae:", X_hat.shape)
 

    # Perform signal detection
    print("Implement MMSE receiver:")
    X_hat = signal_detection(Z, Hp, method='MMSE', noise_power=0.01)
    print("x_hat shpae:", X_hat.shape)

if __name__ == "__main__":
    test_receiver_processing()