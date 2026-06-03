Canonical columns (CSV/Parquet):

- `sample_id`, `source_paper_id` (group for CV)
- `matrix_formula`, `reinforcement_formula` (pymatgen composition strings, e.g. `Al`, `SiC`, `Al2O3`)
- `fabrication_route` (free text; normalized in `features/processing.py`)
- `hardness_value`, `hardness_unit` (`hv`, `gpa`, or `kgf/mm2`)
- `vol_frac_reinf` **or** `wt_frac_reinf` + `rho_matrix_g_cm3` + `rho_reinf_g_cm3`
- Optional processing / microstructure: `temp_c`, `time_min`, `pressure_mpa`, `grain_size_nm`, `particle_size_nm`, `heat_treatment_c`, `rolling_reduction_pct`, `delta_alpha_1e6_k` (×10⁻⁶ K⁻¹ difference placeholder)
