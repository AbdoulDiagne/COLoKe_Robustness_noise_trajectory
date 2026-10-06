# Robust COLoKe: Trajectory-Based Filtering for Deep Koopman Embeddings

This repository extends the official implementation of **COLoKe** (*Conformal Online Learning of Deep Koopman Linear Embeddings*) by introducing a trajectory-level loss computation and filtering mechanism based on $\chi^2$ statistics to handle white noise.

This work was conducted as part of a research internship focused on robust machine learning for dynamical systems under the supervision of the authors of COLoKe.

---

## Key Features & Contributions

* **Base Framework (COLoKe):** Online Koopman representation learning with conformal-based adaptive updates and theoretical sublinear dynamic regret bounds.
* **Trajectory-Level Loss Computation:** Formulation of trajectory-wise loss metrics to evaluate system behavior over complete temporal windows.
* **$\chi^2$ Trajectory Filtering:** Selection and rejection of noisy trajectories based on $\chi^2$ thresholding.
* **Robust Fallback Mechanism:** Fallback logic using median loss  when no healthy trajectories satisfy the $\chi^2$ threshold, guaranteeing continuous gradient flow.
* **Rigorous Statistical Benchmarking:** Paired Student's t-tests (`ttest_rel`) evaluated across 50 independent seeds with identical noise realizations.

---

## Paper & Citation

This extension is built directly upon the **COLoKe** framework. If you use this codebase or methodology, please cite the foundational paper:

```bibtex
@misc{gao2025conformalonlinelearningdeep,
      title={Conformal Online Learning of Deep Koopman Linear Embeddings}, 
      author={Ben Gao and Jordan Patracone and Stéphane Chrétien and Olivier Alata},
      year={2025},
      eprint={2511.12760},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={[https://arxiv.org/abs/2511.12760]}
}

### Contact

For questions regarding the trajectory-level loss computation, $\chi^2$ filtering, or experimental setups:

**Abdoul Diagne** (Internship Author) – [abdouldiagne6@gmail.com](mailto:abdouldiagne6@gmail.com)

For questions regarding the core COLoKe framework :

**Ben Gao** – [ben.gao@univ-st-etienne.fr](mailto:ben.gao@univ-st-etienne.fr)