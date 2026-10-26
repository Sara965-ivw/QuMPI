# Recorded environment (not yet a clean-install certification)

The frozen RQ1 run logs and associated run metadata record an environment built around:

- Python 3.12.3
- PyTorch 2.13.0, CUDA 13.0 build on the original GPU host
- NumPy 2.5.1
- Qiskit 2.5.0
- Qiskit Aer 0.17.2
- qiskit-ibm-runtime 0.47.0
- pymetis 2025.2.2
- scikit-learn 1.9.1 (for GBM baseline)
- matplotlib 3.11.2 (RQ2 analysis figures)
- pytest 9.1.1 (tests)
- mpi4py 4.1.2 and an MPI implementation for RQ2 execution

The root `requirements.txt` is retained from the original source snapshot. It does not include every package used in later baselines, analysis, and tests. The versions above are transcribed from recorded environment notes; a clean, network-independent installation has **not** been validated as part of this packaging step. Before a public release, generate and test an environment lockfile on a clean machine. GPU-specific PyTorch wheels and the MPI implementation may require platform-specific installation steps.
