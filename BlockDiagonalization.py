import numpy as np

epsilon = 1e-6  # Small constant to avoid division by zero


# --------------------------------------------------------------
# Block Diagonalization (BD) Precoding for Multi-User MIMO Systems
# --------------------------------------------------------------
def BD(H, S):
    """
    Block Diagonalization (BD) precoding method for multi-user MIMO systems.
    Input: 
    - H: channel matrix of shape (Nr x Nt x K), where Nr is the number of receive antennas per user, Nt is the number of transmit antennas, and K is the number of users.
    Output:
    - F: precoding matrix of shape (Nt x S x K), where S is the number of data streams per user.
    - W: combining matrix of shape (S x Nr x K), where S is the number of data streams per user.
    - P: power allocation matrix of shape (S x K), where S is the number of data streams per user.
    """
    # Get the dimensions of the input matrices
    if H.ndim != 3:
        raise ValueError("Input matrices must have the correct dimensions: H (Nr x Nt x K)")
    Nr, Nt, K = H.shape
    if Nt < Nr * K:
        raise ValueError("Number of transmit antennas must be greater than or equal to Nr * K for BD precoding.")

    # Initialize the precoding and combining matrices
    F = np.zeros((Nt, S, K), dtype=complex)
    W = np.zeros((S, Nr, K), dtype=complex)
    P = np.zeros((S, K), dtype=complex)  # Power loading matrix for each user

    # Perform block diagonalization for each user
    for k in range(K):
        # Get the complementary channel matrix for user k
        H_k = np.delete(H, k, axis=2)  # Remove the channel of user k
        H_k = H_k.reshape(Nr * (K - 1), Nt)  # Reshape to ((Nr*(K-1)) x Nt)

        # Perform the first singular value decomposition (SVD) on the complementary channel matrix
        U, S_k, Vh = np.linalg.svd(H_k)
        # shape of each matrix: U (Nr*(K - 1) x Nr*(K - 1)), S_k (min(Nr*(K - 1), Nt)), Vh (Nt x Nt)
        d = min(Nt - Nr * (K - 1), Nt)  # Dimension of the null space
        if sum(S_k < epsilon) > 0:
            d = d + sum(S_k < epsilon)  # Adjust dimension based on non-zero singular values

        # Get the null space of the complementary channel matrix, whith dimension (Nt x d).
        null_space = Vh[-d:, :].conj().T  # Null space corresponds to small singular values

        # Calculate the effective channel for user k after precoding, with shape: (Nr x S)
        H_eff = H[:, :, k] @ null_space  # (Nr x Nt) x (Nt x d) = (Nr x d)
        # Perform the second singular value decomposition (SVD) on the effective channel to get the combining matrix
        U_eff, S_eff, Vh_eff = np.linalg.svd(H_eff)
        W_k = U_eff[:, :S].conj().T  # Combining matrix for user k, with shape: (S x Nr)
        F_k0 = Vh_eff[:S, :].conj().T  # Precoding matrix for user k, with shape: (d x S)
        F_k = null_space @ F_k0  # Final precoding matrix for user k, with shape: (Nt x S)
        F[:, :, k] = F_k
        W[:, :, k] = W_k

        # power loading for user k, with shape: (S x S)
        # Assuming each user has the same symbol power, water-filling algorithm can be applied to allocate power across the data streams based on the singular values of the effective channel.
        power_loading = WaterFilling(S_eff[:S], symbol_power=1.0)  # Get power allocation for the S data streams
        P[:, k] = power_loading

    return F, W, P

def WaterFilling(S_eff, symbol_power: float = 1.0):
    """
    Water-filling algorithm for power allocation across data streams based on the singular values of the effective channel.
    Optimize problem: max sum(log(1 + power_allocation[i] * S_eff[i])) subject to sum(power_allocation) <= symbol_power and power_allocation[i] >= 0.
    Input: S_eff: singular values of the effective channel with dimension (N), symbol_power: total power available for transmission, predefined as 1.0.
    Output: power_allocation: power allocated to each data stream, with dimension (N).
    Refer to: https://zhuanlan.zhihu.com/p/502453127
    A paper thinks S_eff should be squared, but I find it is wrong. Actually, S_eff represents the channel gain of each data stream.
    """

    num_streams = len(S_eff)
    power_allocation = np.zeros(num_streams)
    total_power = symbol_power

    # Calculate the water level
    water_level = (total_power + np.sum(1 / (S_eff + epsilon))) / num_streams

    # Allocate power to each stream based on the water level
    for i in range(num_streams):
        power_allocation[i] = max(0, water_level - 1 / (S_eff[i] + epsilon))

    return power_allocation


def test_BD():
    Nt = 16  # Number of transmit antennas
    Nr = 4  # Number of receive antennas per user
    K = 3  # Number of users
    S = 2  # Number of data streams per user
    print(f"Testing Block Diagonalization (BD) Precoding with Nt={Nt}, Nr={Nr}, K={K}, S={S}...")
    H = np.random.randn(Nr, Nt, K) + 1j * np.random.randn(Nr, Nt, K)  # Random channel matrix
    X = np.random.randn(S, K) + 1j * np.random.randn(S, K)  # Random transmitted signal
    F, W, P = BD(H, S)
    print("Precoding matrix F shape:", F.shape)
    print("Combining matrix W shape:", W.shape)
    print("Power allocation matrix P shape:", P.shape)


if __name__ == "__main__":
    test_BD()