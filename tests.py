#%%

import numpy as np
import matplotlib.pyplot as plt
from multipyles.multipole_eqs import spin_part
from multipyles.helper import spherical_to_cubic

# Matplotlib default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
print("Matplotlib color cycle:", colors)

# For spin-1/2
spin_values = [-0.5, 0.5]
p = 1  # set your desired p
y = 0  # set your desired y (in range -p to +p)

matrix = np.empty((2, 2), dtype=complex)
for i, s in enumerate(spin_values):
    for j, sp in enumerate(spin_values):
        matrix[i, j] = spin_part(p, y, s, sp)

print(matrix)

#%%
import numpy as np
from multipyles.multipole_eqs import orbital_part, _norm_orbital

l = 2  # for d orbitals
k = 2  # multipole component
x = 1  # pick your multipole component (-k to k)


def rotate_qe_to_vasp(mat, l=2):
    assert l == 2, "Currently only l=2 is supported"

    qe_to_vasp = np.array([
        [0, 0, 1, 0, 0],  # r^2-3z^2
        [0, 1, 0, 0, 0],  # xz
        [0, 0, 0, 1, 0],  # yz
        [1, 0, 0, 0, 0],  # xy
        [0, 0, 0, 0, 1]   # x^2-y^2
        ], dtype=complex).T
    return qe_to_vasp @ mat @ qe_to_vasp.T

def rotate_vasp_to_qe(mat, l=2):
    assert l == 2, "Currently only l=2 is supported"

    vasp_to_qe = np.array([
        [0, 0, 1, 0, 0],  # r^2-3z^2
        [0, 1, 0, 0, 0],  # xz
        [0, 0, 0, 1, 0],  # yz
        [1, 0, 0, 0, 0],  # xy
        [0, 0, 0, 0, 1]   # x^2-y^2
        ], dtype=complex)
    return vasp_to_qe @ mat @ vasp_to_qe.T

def get_matrix(l, k, x, cubic=True, qe_convention=False):
    """
    Get the orbital multipole matrix Q(k,x) for given l, k, x.
    """
    assert l == 2, "Currently only l=2 is supported"

    m_values = np.arange(-l, l+1)
    matrix = np.empty((2*l+1, 2*l+1), dtype=complex)  # or complex if needed
    for i, m in enumerate(m_values):
        for j, mp in enumerate(m_values):
            matrix[i, j] = orbital_part(l, k, x, m, mp)

    # norm_fac = _norm_orbital(l, k)**2
    norm_fac = 1/(2*k+1)*(_norm_orbital(l, k)**2)

    if cubic:
        trafo_matrix = spherical_to_cubic(l)
        # verify
        matrix = trafo_matrix.T.conj() @ matrix @ trafo_matrix

    if qe_convention:
        # print("Using Quantum Espresso convention for l=2")
        matrix = rotate_vasp_to_qe(matrix)
    return matrix, norm_fac


def build_spinorbital_density_matrix(rho_up, rho_down=None, rho_updown=None, rho_downup=None):
    """
    Build the full spin-orbital density matrix from spin-resolved density matrices.
    
    Structure: [ρ↓↓  ρ↓↑]    (following multipyles convention: down first)
               [ρ↑↓  ρ↑↑]
    
               Each block is of size (n_orb, n_orb) where n_orb is the number of orbitals.
    
    Parameters:
    -----------
    rho_up : array_like
        Spin-up density matrix (ρ↑↑ block)
    rho_down : array_like, optional
        Spin-down density matrix (ρ↓↓ block). If None, assumes zero
    rho_updown : array_like, optional
        Spin up-down density matrix (ρ↑↓ block). If None, assumes zero (collinear case)
    rho_downup : array_like, optional
        Spin down-up density matrix (ρ↓↑ block). If None, assumes zero (collinear case)
        
    Returns:
    --------
    np.ndarray : Full spin-orbital density matrix of shape (2*n_orb, 2*n_orb)
    """
    
    # Convert inputs to numpy arrays
    rho_up = np.array(rho_up, dtype=complex)
    n_orb = rho_up.shape[0]
    
    if rho_down is not None:
        rho_down = np.array(rho_down, dtype=complex)
    else:
        rho_down = np.zeros_like(rho_up)
    
    if rho_updown is not None:
        rho_updown = np.array(rho_updown, dtype=complex)
    else:
        rho_updown = np.zeros_like(rho_up)
        
    if rho_downup is not None:
        rho_downup = np.array(rho_downup, dtype=complex)
    else:
        rho_downup = np.zeros_like(rho_up)
    
    # Build full spin-orbital density matrix 
    # Following multipyles convention: first index is down (-1/2), second is up (+1/2)
    rho_full = np.zeros((2*n_orb, 2*n_orb), dtype=complex)
    rho_full[:n_orb, :n_orb] = rho_down      # ρ↓↓ block (first index)
    rho_full[:n_orb, n_orb:] = rho_downup    # ρ↓↑ block  
    rho_full[n_orb:, :n_orb] = rho_updown    # ρ↑↓ block
    rho_full[n_orb:, n_orb:] = rho_up        # ρ↑↑ block (second index)
    
    return rho_full


def generate_multipole_operators(l=2, max_k=None, cubic=True, qe_convention=False):
    """
    Generate all multipole operators for a given configuration.
    
    This creates a dictionary of all multipole operators with consistent conventions,
    eliminating the need to specify conventions multiple times.
    
    Parameters:
    -----------
    l : int, default=2
        Orbital angular momentum quantum number
    max_k : int, optional
        Maximum k value for multipole expansion. If None, uses 2*l
    cubic : bool, default=True
        Whether to use cubic harmonics basis
    qe_convention : bool, default=False
        Whether to use Quantum Espresso orbital ordering convention
        
    Returns:
    --------
    dict : Dictionary with operator information
        Keys are (k, x, p, y) tuples
        Values are dicts with 'type', 'operator', 'normalization'
    """
    
    if max_k is None:
        max_k = 2 * l
    
    # Spin matrices using multipyles convention
    spin_values = [-0.5, 0.5]
    sigma_0 = np.empty((2, 2), dtype=complex)
    sigma_z = np.empty((2, 2), dtype=complex)
    
    for i, s in enumerate(spin_values):
        for j, sp in enumerate(spin_values):
            sigma_0[i, j] = spin_part(0, 0, s, sp)  # Identity
            sigma_z[i, j] = spin_part(1, 0, s, sp)  # σz matrix
    
    # Multipole type mapping
    charge_names = {
        0: "monopole", 1: "dipole", 2: "quadrupole", 3: "octupole",
        4: "hexadecapole", 5: "triakontadipole", 6: "hexacontatetrapole"
    }
    
    magnetic_names = {
        0: "dipole", 1: "quadrupole", 2: "octupole", 
        3: "hexadecapole", 4: "triakontadipole"
    }
    
    operators = {}
    
    for k_val in range(max_k + 1):
        for x_val in range(-k_val, k_val + 1):
            # Get orbital multipole operator
            Q_orbital, norm_orbital = get_matrix(l, k_val, x_val, cubic=cubic, qe_convention=qe_convention)
            
            # Charge multipole: I_spin ⊗ Q_orbital
            Q_charge = np.kron(sigma_0, Q_orbital)
            charge_type = charge_names.get(k_val, f"k={k_val}")
            
            operators[(k_val, x_val, 0, 0)] = {
                "type": f"charge_{charge_type}",
                "operator": Q_charge,
                "normalization": norm_orbital
            }
            
            # Magnetic multipole: σz ⊗ Q_orbital
            Q_magnetic = np.kron(sigma_z, Q_orbital)
            magnetic_type = magnetic_names.get(k_val, f"k={k_val}")
            
            operators[(k_val, x_val, 1, 0)] = {
                "type": f"magnetic_{magnetic_type}_collinear",
                "operator": Q_magnetic,
                "normalization": norm_orbital
            }
    
    return operators


def get_multipole_decomposition_full(rho_up, rho_down=None, l=2, max_k=None, cubic=True, qe_convention=False, rho_updown=None, rho_downup=None):
    """
    Calculate multipole decomposition using pre-generated operators.

    This builds the full 2n_orb × 2n_orb density matrix with spin blocks:
    [ρ↓↓  ρ↓↑]    (following multipyles convention: down first)
    [ρ↑↓  ρ↑↑]
    
    Parameters:
    -----------
    rho_up : array_like
        Spin-up density matrix
    rho_down : array_like, optional  
        Spin-down density matrix. If None, assumes rho_down = 0
    l : int, default=2
        Orbital angular momentum quantum number
    max_k : int, optional
        Maximum k value for multipole expansion. If None, uses 2*l
    cubic : bool, default=True
        Whether to use cubic harmonics basis
    qe_convention : bool, default=False
        Whether to use Quantum Espresso orbital ordering convention
    rho_updown : array_like, optional
        Spin up-down density matrix. If None, assumes zero (collinear case)
    rho_downup : array_like, optional
        Spin down-up density matrix. If None, assumes zero (collinear case)
        
    Returns:
    --------
    dict : Dictionary with multipole coefficients
    """
    
    # Generate operators with consistent conventions
    operators = generate_multipole_operators(l=l, max_k=max_k, cubic=cubic, qe_convention=qe_convention)
    
    # Build the full spin-orbital density matrix
    rho_full = build_spinorbital_density_matrix(rho_up, rho_down, rho_updown, rho_downup)
    
    results = {}
    
    # Calculate multipole coefficients
    for (k, x, p, y), op_data in operators.items():
        operator = op_data["operator"]
        coefficient = np.trace(operator.T @ rho_full)
        
        results[(k, x, p, y)] = {
            "multipole_type": op_data["type"],
            "normalization": op_data["normalization"],
            "value": coefficient
        }
    
    return results


def build_rho_from_decomposition(decomposition, l=2, cubic=True, qe_convention=False):
    """
    Rebuild the full spin-orbital density matrix from multipole decomposition coefficients.
    
    This function reconstructs the 2n_orb × 2n_orb density matrix with spin blocks:
    [ρ↓↓  ρ↓↑]    (following multipyles convention: down first)
    [ρ↑↓  ρ↑↑]
    
    from the multipole coefficients obtained via get_multipole_decomposition_full().
    
    Parameters:
    -----------
    decomposition : dict
        Dictionary with multipole coefficients from get_multipole_decomposition_full()
        Keys are (k, x, p, y) tuples, values are dicts with 'value' and 'normalization'
    l : int, default=2
        Orbital angular momentum quantum number
    cubic : bool, default=True
        Whether to use cubic harmonics basis (should match decomposition)
    qe_convention : bool, default=False
        Whether to use Quantum Espresso orbital ordering (should match decomposition)
        
    Returns:
    --------
    np.ndarray : Reconstructed full spin-orbital density matrix of shape (2*n_orb, 2*n_orb)
    """
    
    # Generate the same operators used in decomposition
    operators = generate_multipole_operators(l=l, max_k=None, cubic=cubic, qe_convention=qe_convention)
    
    n_orb = 2 * l + 1  # Number of orbitals (5 for d orbitals)
    rho_full = np.zeros((2*n_orb, 2*n_orb), dtype=complex)
    
    # Reconstruct from multipole coefficients
    for (k, x, p, y), data in decomposition.items():
        if (k, x, p, y) not in operators:
            continue  # Skip if operator not available
            
        coefficient = data['value']
        normalization = data['normalization']
        operator = operators[(k, x, p, y)]['operator']
        
        # Add contribution to the reconstructed density matrix
        # Correct reconstruction formula: coeff * operator / norm / 2
        rho_full += coefficient * operator / normalization / 2
    
    return rho_full


def test_reconstruction_roundtrip():
    """
    Test that multipole decomposition -> reconstruction gives back the original matrix.
    This verifies the mathematical consistency of the implementation.
    """
    print("=" * 60)
    print("TESTING MULTIPOLE DECOMPOSITION RECONSTRUCTION")
    print("=" * 60)
    
    # Test with simple diagonal matrices
    test_rho_up = np.diag([0.8, 0.7, 0.6, 0.5, 0.4])
    test_rho_down = np.diag([0.2, 0.3, 0.4, 0.5, 0.6])
    
    print("Original spin-up matrix:")
    print(test_rho_up)
    print("\nOriginal spin-down matrix:")
    print(test_rho_down)
    
    # Build original full density matrix
    original_rho_full = build_spinorbital_density_matrix(test_rho_up, test_rho_down)
    print(f"\nOriginal full matrix shape: {original_rho_full.shape}")
    
    # Decompose into multipoles
    decomposition = get_multipole_decomposition_full(
        rho_up=test_rho_up,
        rho_down=test_rho_down,
        l=2,
        max_k=4,  # Use full expansion
        cubic=True,
        qe_convention=False
    )
    
    print(f"\nNumber of multipole components: {len(decomposition)}")
    
    # Reconstruct from multipoles
    reconstructed_rho_full = build_rho_from_decomposition(
        decomposition=decomposition,
        l=2,
        cubic=True,
        qe_convention=False
    )
    
    print(f"Reconstructed full matrix shape: {reconstructed_rho_full.shape}")
    
    # Check reconstruction accuracy
    difference = original_rho_full - reconstructed_rho_full
    max_error = np.max(np.abs(difference))
    
    print(f"\nReconstruction accuracy:")
    print(f"Maximum absolute error: {max_error:.2e}")
    print(f"Reconstruction {'SUCCESSFUL' if max_error < 1e-12 else 'FAILED'}")
    
    # Extract the spin blocks from reconstructed matrix (down first, up second in multipyles convention)
    n_orb = 5
    reconstructed_rho_down = reconstructed_rho_full[:n_orb, :n_orb]
    reconstructed_rho_up = reconstructed_rho_full[n_orb:, n_orb:]
    
    print(f"\nSpin-up reconstruction error: {np.max(np.abs(test_rho_up - reconstructed_rho_up)):.2e}")
    print(f"Spin-down reconstruction error: {np.max(np.abs(test_rho_down - reconstructed_rho_down)):.2e}")
    
    return max_error < 1e-12


# Run the test
if __name__ == "__main__":
    test_reconstruction_roundtrip()


get_matrix(2,0,0)

m1, norm_m = get_matrix(2,k,x, cubic=True)
print(m1.real)
normalization_factor= np.trace(m1.dot(m1.T.conj()))
print(normalization_factor, norm_m)



#%%
#%%
matrix_size = 5
np.random.seed(42) # for reproducibility
random_matrix_real = np.random.rand(matrix_size, matrix_size)
random_matrix_real = random_matrix_real + random_matrix_real.T 
# random_matrix_imag = np.random.rand(matrix_size, matrix_size)
# rho = random_matrix_real 

#QE MATRIX

# ns_array_3 = np.zeros((1, 5, 5, 2, 2))
# # SPIN 1 matrix
qe_spin_up_data = np.array([
    [0.476, -0.000, -0.000,  0.000,  0.000],
    [-0.000, 0.737, -0.000,  0.000,  0.000],
    [-0.000, -0.000, 0.737,  0.000,  0.000],
    [0.000,  0.000,  0.000,  0.859, -0.000],
    [0.000,  0.000,  0.000, -0.000, 0.471],
])
# # SPIN 2 matrix
qe_spin_down_data = np.array([
    [0.416, -0.000, -0.000,  0.000,  0.000],
    [-0.000, 0.285, -0.000,  0.000,  0.000],
    [-0.000, -0.000, 0.285,  0.000,  0.000],
    [0.000,  0.000,  0.000,  0.282, -0.000],
    [0.000,  0.000,  0.000, -0.000, 0.402],
])
# # Assign the matrices
# ns_array_3[0, :, :, 0, 0] = spin1
# ns_array_3[0, :, :, 1, 1] = spin2
# # ns_array = qe_to_vasp(ns_array_3)

# Spin up density matrix
vasp_spin_up_data = [
    [0.797, 0.000, -0.000, -0.000, 0.000],
    [0.000, 0.681, -0.000, -0.000, 0.000],
    [-0.000, -0.000, 0.429, -0.000, 0.000],
    [-0.000, -0.000, -0.000, 0.681, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.415]
]

# Spin down density matrix
vasp_spin_down_data = [
    [0.204, -0.000, -0.000, 0.000, -0.000],
    [-0.000, 0.207, 0.000, -0.000, -0.000],
    [-0.000, 0.000, 0.376, 0.000, 0.000],
    [0.000, -0.000, 0.000, 0.207, 0.000],
    [-0.000, -0.000, 0.000, 0.000, 0.356]]


#%%
monopole_Q = get_matrix(l=2, k=0, x=0, cubic=True)[0]

monopole = np.trace(monopole_Q.T @ vasp_spin_up_data + monopole_Q.T @ vasp_spin_down_data)


print("Monopole coefficient (k=0, x=0):", monopole)

dipole_Q = get_matrix(l=2, k=1, x=0, cubic=True)[0]
dipole = np.trace(dipole_Q.T @ vasp_spin_up_data + dipole_Q.T @ vasp_spin_down_data)
print("Dipole coefficient (k=1, x=0):", dipole)

# analyze and print all quadrupole coefficients
for k_val in [2]:
    for x_val in range(-k_val, k_val + 1):
        Q_kx, norm_kx = get_matrix(l=2, k=k_val, x=x_val, cubic=True)
        c_kx = np.trace(Q_kx.T @ vasp_spin_up_data + Q_kx.T @ vasp_spin_down_data)
        print(f"Quadrupole coefficient (k={k_val}, x={x_val}): {c_kx:.4f}")


#%% now let's do it for quantum espresso data
monopole_Q = get_matrix(l=2, k=0, x=0, cubic=True )[0]
monopole = np.trace(monopole_Q.T @ qe_spin_up_data + monopole_Q.T @ qe_spin_down_data)
print("Monopole coefficient (k=0, x=0):", monopole)

dipole_Q = get_matrix(l=2, k=1, x=0, cubic=True)[0]
dipole = np.trace(dipole_Q.T @ qe_spin_up_data + dipole_Q.T @ qe_spin_down_data)
print("Dipole coefficient (k=1, x=0):", dipole)

# analyze and print all quadrupole coefficients
# order0 = [3, 2, 0, 1, 4]
# qe_spin_up_data = qe_spin_up_data[np.ix_(order0, order0)]
# qe_spin_down_data = qe_spin_down_data[np.ix_(order0, order0)]


for k_val in [2]:
    for x_val in range(-k_val, k_val + 1):
        Q_kx, norm_kx = get_matrix(l=2, k=k_val, x=x_val, cubic=True, qe_convention=True)
        # c_kx = np.trace(Q_kx.T @ rotate_to_vasp(qe_spin_up_data) + Q_kx.T @ rotate_to_vasp(qe_spin_down_data))
        c_kx = np.trace(Q_kx.T @ qe_spin_up_data + Q_kx.T @ qe_spin_down_data)

        print(f"Quadrupole coefficient (k={k_val}, x={x_val}): {c_kx:.4f}")
                    

#%%

# Use the random matrix for testing - demonstrate orbital-space decomposition
# First hermitize the matrix to ensure perfect reconstruction
rho = random_matrix_real
# rho = qe_spin_up_data
# rho = np.array([
#     [1, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0],
#     [0, 0, 1, 0, 0],    
#     [0, 0, 0, 0.5, 0],
#     [0, 0, 0, 0, 1]])

rho = (rho + rho.T.conj()) / 2  # Make perfectly Hermitian

print("Original Random Matrix (rho) [hermitized]:\n", rho)

# Initialize a matrix to store the rebuilt result
rebuilt_rho = np.zeros_like(rho, dtype=complex)

# Dictionary to store the calculated multipole coefficients
multipole_coefficients = {}

for k_val in range(2 * l + 1): # k from 0 to 4
    for x_val in range(-k_val, k_val + 1): # x from -k to k
        # Get the basis matrix Q(k,x) - orbital space only
        Q_kx, norm_kx = get_matrix(l, k_val, x_val, cubic=False, qe_convention=True)

        c_kx = np.trace(Q_kx.T.conj() @ rho)

        multipole_coefficients[(k_val, x_val)] = c_kx

        # Add the contribution to the rebuilt matrix
        rebuilt_rho += c_kx * Q_kx / norm_kx

# print("\nCalculated Multipole Coefficients (multipole(k,x)):\n")
# for (k, x), coeff in multipole_coefficients.items():
#     print(f"multipole({k},{x}): {coeff:.4f}")

# print("\nRebuilt Matrix:\n", rebuilt_rho)

# Compare the original and rebuilt matrix
difference = rho - rebuilt_rho
print("\nDifference (Original - Rebuilt):\n", difference)
print(f"\nMaximum absolute difference: {np.max(np.abs(difference)):.4e}")


#%% decomposition of a spin density matrix
# rho_up = rho/2
# rho_down = rho/2


from multipyles import multipyles, read_from_dft
# Parse the .out file
with open(f'sco.scf.out', 'r') as file:
    occupation_per_site = read_from_dft.read_densmat_from_qe(file)

# Calculate (not-normalized) multipole moments
results, l = multipyles.calculate(occupation_per_site, verbose=False)
rho_down  = occupation_per_site[0][:, :, 1, 1]
rho_up    = occupation_per_site[0][:, :, 0, 0]


# block matrix with spin blocks
rho_total = np.block([
    [rho_down, np.zeros_like(rho)],  # ρ↓↓ and ρ↓↑
    [np.zeros_like(rho), rho_up]     # ρ↑↓ and ρ↑↑
])

rebuilt_rho_total = np.zeros_like(rho_total, dtype=complex)
multipole_coefficients = {}

for p_val in [0, 1]:  # p=0 (charge), p=1 (magnetic)
    for y_val in [0]:  # y is always 0 for collinear case
        for k_val in range(2 * l + 1):  # k from 0 to 4
            for x_val in range(-k_val, k_val + 1):  # x from -k to k
                # Get the full spin-orbital basis matrix Q(k,x,p,y)
                Q_kxpy, norm_kxpy = get_matrix(l, k_val, x_val, cubic=True, qe_convention=True)
                
                if p_val == 0:
                    # Charge multipole: I_spin ⊗ Q_orbital
                    sigma_0 = np.array([[1, 0], [0, 1]], dtype=complex)
                    Q_full = np.kron(sigma_0, Q_kxpy)
                else:
                    # Magnetic multipole: σz ⊗ Q_orbital
                    sigma_z = np.array([[ -0.5, 0], [0, 0.5]], dtype=complex)  # multipyles convention
                    Q_full = np.kron(sigma_z, Q_kxpy)
                
                c_kxpy = np.trace(Q_full.T.conj() @ rho_total)
                
                multipole_coefficients[(k_val, x_val, p_val, y_val)] = c_kxpy
                
                # Add the contribution to the rebuilt matrix
                rebuilt_rho_total += c_kxpy * Q_full / norm_kxpy /2

difference = rho_total - rebuilt_rho_total
print("\nDifference (Original - Rebuilt) for spin density matrix:\n", difference)
print(f"\nMaximum absolute difference: {np.max(np.abs(difference)):.4e}")


#%% let's do the same but with the function

decomposition = get_multipole_decomposition_full(rho_up, rho_down, qe_convention=True)


rebuilt = build_rho_from_decomposition(decomposition=decomposition, qe_convention=True)

with np.printoptions(precision=3):
    print(rebuilt[0:5, 0:5].real)
    print(rho_up[0:5, 0:5].real)

#%%
# Test the new multipole decomposition with the same random matrix (as charge-only)
print("\n" + "=" * 60)
print("COMPARISON WITH NEW MULTIPOLE DECOMPOSITION")
print("=" * 60)

# Use the random matrix as a single-spin density (no magnetic part)
random_results = get_multipole_decomposition_full(
    rho_up=rho, 
    rho_down=None,  # This will create rho_down = 0, so only charge multipoles
    l=2, 
    max_k=4, 
    cubic=True, 
    qe_convention=False
)

print("\nCharge Multipoles from new function:")
print("-" * 40)
for (k, x, p, y), data in random_results.items():
    if p == 0:  # Charge multipoles only
        old_coeff = multipole_coefficients.get((k, x), 0)
        new_coeff = data['value']
        print(f"({k:1d},{x:2d}): old={old_coeff:8.4f}, new={new_coeff:8.4f}, diff={abs(old_coeff-new_coeff):8.4e}")

len(multipole_coefficients)

#%% Test the new comprehensive multipole decomposition function

# # Test with VASP data (charge and magnetic multipoles)
# print("=" * 60)
# print("COMPREHENSIVE MULTIPOLE DECOMPOSITION - VASP DATA")
# print("=" * 60)

# vasp_results = get_multipole_decomposition_full(
#     rho_up=vasp_spin_up_data, 
#     rho_down=vasp_spin_down_data, 
#     l=2, 
#     max_k=4, 
#     cubic=True, 
#     qe_convention=False
# )

# print("\nCharge Multipoles (Time-reversal even, p=0):")
# print("-" * 50)
# for (k, x, p, y), data in vasp_results.items():
#     if p == 0:  # Charge multipoles
#         print(f"({k:1d},{x:2d},{p:1d},{y:2d}): {data['multipole_type']:20} = {data['value']:8.4f}, norm = {data['normalization']:8.4f}")

# print("\nMagnetic Multipoles (Time-reversal odd, p=1):")
# print("-" * 50)
# for (k, x, p, y), data in vasp_results.items():
#     if p == 1:  # Magnetic multipoles
#         print(f"({k:1d},{x:2d},{p:1d},{y:2d}): {data['multipole_type']:20} = {data['value']:8.4f}, norm = {data['normalization']:8.4f}")
#%%
# Test with QE data
print("\n" + "=" * 60)
print("COMPREHENSIVE MULTIPOLE DECOMPOSITION - QE DATA")
print("=" * 60)

qe_results = get_multipole_decomposition_full(
    rho_up=qe_spin_up_data, 
    rho_down=qe_spin_down_data, 
    l=2, 
    max_k=3, 
    cubic=True, 
    qe_convention=True
)
print("\nCharge Multipoles (Time-reversal even, p=0):")
print("-" * 50)
for (k, x, p, y), data in qe_results.items():
    if p == 0:  # Charge multipoles
        print(f"({k:1d},{x:2d},{p:1d},{y:2d}): {data['multipole_type']:20} = {data['value']:8.4f}, norm = {data['normalization']:8.4f}")

#%%
print("\nMagnetic Multipoles (Time-reversal odd, p=1):")
print("-" * 50)
for (k, x, p, y), data in qe_results.items():
    if p == 1:  # Magnetic multipoles
        print(f"({k:1d},{x:2d},{p:1d},{y:2d}): {data['multipole_type']:20} = {data['value']:8.4f}, norm = {data['normalization']:8.4f}")

#%%
# Test with single spin component (charge only)
print("\n" + "=" * 60)
print("SINGLE SPIN COMPONENT - CHARGE MULTIPOLES ONLY")
print("=" * 60)

single_spin_results = get_multipole_decomposition_full(
    rho_up=vasp_spin_up_data, 
    rho_down=None, 
    l=2, 
    max_k=2, 
    cubic=True
)

print("\nCharge Multipoles only:")
print("-" * 50)
for (k, x, p, y), data in single_spin_results.items():
    print(f"({k:1d},{x:2d},{p:1d},{y:2d}): {data['multipole_type']:20} = {data['value']:8.4f}, norm = {data['normalization']:8.4f}")

# Test collinear flag - only pz component for magnetic multipoles
print("\n" + "=" * 60)
print("COLLINEAR MAGNETISM - ONLY pz COMPONENT")
print("=" * 60)

collinear_results = get_multipole_decomposition_full(
    rho_up=vasp_spin_up_data, 
    rho_down=vasp_spin_down_data, 
    l=2, 
    max_k=2, 
    cubic=True
)

print("\nCharge Multipoles (unchanged):")
print("-" * 50)
for (k, x, p, y), data in collinear_results.items():
    if p == 0:  # Charge multipoles
        print(f"({k:1d},{x:2d},{p:1d},{y:2d}): {data['multipole_type']:20} = {data['value']:8.4f}, norm = {data['normalization']:8.4f}")

print("\nMagnetic Multipoles (collinear - only y=0/pz):")
print("-" * 50)
for (k, x, p, y), data in collinear_results.items():
    if p == 1:  # Magnetic multipoles
        print(f"({k:1d},{x:2d},{p:1d},{y:2d}): {data['multipole_type']:20} = {data['value']:8.4f}, norm = {data['normalization']:8.4f}")

print(f"\nNote: With collinear=True, only {sum(1 for (k,x,p,y) in collinear_results.keys() if p == 1)} magnetic multipole components calculated")
# print(f"With collinear=False, {sum(1 for (k,x,p,y) in vasp_results.keys() if p == 1)} magnetic multipole components would be calculated")

# #%%
# transformation_matrix = np.array([
#     [0,      0,              1,      0,     0    ],  # r^2-3z^2
#     [0,      1/np.sqrt(2),   0,      -1/np.sqrt(2),    0    ],  # xz
#     [0,      1j/np.sqrt(2),  0,     -1j/np.sqrt(2),   0    ],  # yz
#     [1j/np.sqrt(2),  0,     0,      0,     -1j/np.sqrt(2) ],  # xy
#     [1/np.sqrt(2),    0,     0,      0,      -1/np.sqrt(2)  ]   # x^2-y^2
# ], dtype=complex)

# with np.printoptions(precision=3):
#     print(transformation_matrix)
#     print(transformation_matrix@transformation_matrix.T.conj())

# %%
from multipyles import multipyles, read_from_dft
# Parse the .out file
with open(f'sco.scf.out', 'r') as file:
    occupation_per_site = read_from_dft.read_densmat_from_qe(file)

# Calculate (not-normalized) multipole moments
results, l = multipyles.calculate(occupation_per_site, verbose=False)
sco_rho_down  = occupation_per_site[0][:, :, 1, 1]
sco_rho_up    = occupation_per_site[0][:, :, 0, 0]

# use this file's analysis function 
decomposition = get_multipole_decomposition_full(
    rho_up=sco_rho_up, 
    rho_down=sco_rho_down, 
    l=2, 
    max_k=4, 
    cubic=True, 
    qe_convention=True
)



# Choose the multipole you prefer and filter dataframe
k, p, r = 4, 1, 5
# res_filtered = multipyles.filter_results(results, {'k': k, 'p': p, 'r': r, 'nu': (k+p)%2})
res_filtered = multipyles.filter_results(results, {'atom': 0, 'k': 2, 'p':0} )

# Transform to cubic harmonics
res = multipyles.transform_results_to_real(res_filtered)
print(res)

print("\nDecomposition from inner product:")
(k, x, p, y) = (2, 0, 1, 0)
print(f"({k:1d},{x:2d},{p:1d},{y:2d}): {decomposition[(k, x, p, y)]['multipole_type']:20} = {decomposition[(k, x, p, y)]['value']:8.4f}, norm = {decomposition[(k, x, p, y)]}")