# Stage 7: Downstream Question Answering Evaluation

## Stage 7: Downstream End-to-End QA Answer Evaluation

| Divergence Bucket | Count | Baseline RAG F1 (Cond 3) | Entity-Aware RAG F1 (Cond 5) | F1 Δ | 95% Confidence Interval | Significant (p < 0.05) |
|---|---|---|---|---|---|---|
| **No-mismatch** | 201 | 0.3099 | 0.3155 | +0.0056 | [-0.0019, +0.0165] | **NO** |
| **Partial-mismatch** | 234 | 0.4209 | 0.4219 | +0.0010 | [-0.0022, +0.0038] | **NO** |
| **Full alias-mismatch** | 65 | 0.6206 | 0.6193 | -0.0013 | [-0.0087, +0.0063] | **NO** |

### Qualitative Answer Generation Samples

**Query [Partial-mismatch]**: `Characterization and functional analysis of YBCO under operational testing.`
- **Ground Truth**: YBa2Cu3O7 exhibits critical transition temperature Tc of 92 K, orthorhombic perovskite-like structure, Pmmm space group, strong flux pinning and high critical current density Jc.
- **Condition 3 (Baseline)** (F1=0.8980): Detailed characterization revealed critical transition temperature Tc of 92 K, orthorhombic perovskite-like structure, Pmmm space group, strong flux pinning and high critical current density Jc.
- **Condition 5 (Entity-Aware)** (F1=0.8980): Detailed characterization revealed critical transition temperature Tc of 92 K, orthorhombic perovskite-like structure, Pmmm space group, strong flux pinning and high critical current density Jc.

**Query [Partial-mismatch]**: `Novel experimental methodology regarding 2212 cuprate.`
- **Ground Truth**: Bi2Sr2CaCu2O8 exhibits superconducting transition at 85 K, highly anisotropic layered structure, pancake vortices in c-axis transport.
- **Condition 3 (Baseline)** (F1=0.9375): The material exhibits superconducting transition at 85 K, highly anisotropic layered structure, pancake vortices in c-axis transport.
- **Condition 5 (Entity-Aware)** (F1=0.9375): The material exhibits superconducting transition at 85 K, highly anisotropic layered structure, pancake vortices in c-axis transport.

**Query [No-mismatch]**: `Electrochemical and Optoelectronic Performance of Bi2Sr2Ca2Cu3O10 Bi2Sr2Ca2Cu3O10 Bi2Sr2Ca2Cu3O10`
- **Ground Truth**: Bi2Sr2Ca2Cu3O10 exhibits Tc of 110 K, multi-filamentary tape fabrication, powder-in-tube processing for high-field superconducting magnets.
- **Condition 3 (Baseline)** (F1=0.0000): Comparative analysis confirms that Bi2Sr2Ca2Cu3O10+d maintains high cycling efficiency over extended operational tests.
- **Condition 5 (Entity-Aware)** (F1=0.0000): Comparative analysis confirms that Bi2Sr2Ca2Cu3O10+d maintains high cycling efficiency over extended operational tests.

**Query [Partial-mismatch]**: `Characterization and functional analysis of MgB2 superconductor under operational testing.`
- **Ground Truth**: MgB2 exhibits BCS-type phonon-mediated superconductivity with Tc of 39 K, two-band superconducting gap structure in sigma and pi bands.
- **Condition 3 (Baseline)** (F1=0.2339): X-ray absorption near-edge structure (XANES) and EXAFS at the cation edge in MgB2 superconductor confirm the oxidation state in magnesium diboride.
- **Condition 5 (Entity-Aware)** (F1=0.2339): X-ray absorption near-edge structure (XANES) and EXAFS at the cation edge in MgB2 superconductor confirm the oxidation state in magnesium diboride.

**Query [Partial-mismatch]**: `Novel experimental methodology regarding 122 pnictide.`
- **Ground Truth**: BaFe2As2 exhibits spin density wave transition at 140 K, potassium or cobalt doping suppresses magnetic order inducing superconductivity up to 38 K.
- **Condition 3 (Baseline)** (F1=0.1026): Transport measurements on 122 pnictide validated that stoichiometric BaFe2As2 exhibits reproducibility under ambient and extreme thermal conditions.
- **Condition 5 (Entity-Aware)** (F1=0.1026): Transport measurements on 122 pnictide validated that stoichiometric BaFe2As2 exhibits reproducibility under ambient and extreme thermal conditions.

**Query [No-mismatch]**: `Microstructural and Functional Evolution of FeSe FeSe FeSe`
- **Ground Truth**: FeSe exhibits nematic electronic phase transition at 90 K, bulk Tc of 8 K rising above 65 K in monolayer films on SrTiO3 substrates.
- **Condition 3 (Baseline)** (F1=0.9583): The material exhibits nematic electronic phase transition at 90 K, bulk Tc of 8 K rising above 65 K in monolayer films on SrTiO3 substrates.
- **Condition 5 (Entity-Aware)** (F1=0.9583): The material exhibits nematic electronic phase transition at 90 K, bulk Tc of 8 K rising above 65 K in monolayer films on SrTiO3 substrates.

