# Limitations and deviations from the full thesis proposal

- **RQ1 scope:** confirmatory ranking was tested on 143 circuits at 16–20 qubits, rather than the originally proposed wider 22–40-qubit range.
- **Model interpretation:** the frozen primary is a ranking model, not a calibrated absolute-fidelity regressor. Its advantage over the structural baselines is supported; superiority over GBM-layout and layout-ESP is not.
- **Data scope:** three circuit families, one recorded backend calibration snapshot, limited labelled fragment widths, and some generated holdout circuits that did not require cutting.
- **Statistical reporting:** corrected circuit-level bootstrap results are provided separately from original summary files. Original outputs remain unmodified for provenance.
- **Hardware:** all RQ1 labels use simulation/noise models; no real quantum computer execution is included.
- **RQ2:** the collected pilot covers only 4 and 8 ranks and is affected by the documented shared-counter contention and cost-proxy limitations. It does not establish the proposed full load-balance objective or QUBO.
- **RQ3 / integrated framework:** the end-to-end cut-choice → MPI execution → reconstruction pipeline is not implemented in this snapshot.
- **Reproducibility:** a clean install and fresh-machine rerun have not been validated as part of this packaging pass. Training was reported as non-bit-deterministic on GPU.
