# Proposed method (Phase 4)

**Status:** design and code only. No BraTS experiment has been run. No improvement is claimed.  
**Disclaimer:** research prototype, not a clinically validated diagnostic tool.

This document states the **mathematics** of the components we will ablate. Every piece is an **established** method. NeuroVision-X does **not** introduce a new backbone, a new attention block, or a new uncertainty estimator. The research question is whether a **matched protocol** on official BraTS data supports H1 (boundary term on ET) and H2 (patient-level uncertainty as a failure score), with optional SE fusion and Attention U-Net as controlled extras.

## 0. Baseline (unchanged)

Let \(x \in \mathbb{R}^{4 \times D \times H \times W}\) be co-registered T1, T1ce, T2, FLAIR. The Phase 3 baseline is 3D U-Net \(f_\theta\) (Çiçek et al., 2016) with early fusion (channel concatenation already performed by the loader):

\[
z = f_\theta(x) \in \mathbb{R}^{C \times D \times H \times W},\qquad
p = \mathrm{softmax}(z),\quad C=4
\]

(classes: background, NCR, ED, ET). Nested regions WT/TC/ET are derived after \(\arg\max\), not from extra heads (Menze et al., 2015).

The baseline objective is Dice + weighted CE (`boundary_weight: 0`):

\[
\mathcal{L}_{\mathrm{base}} = \lambda_{\mathrm{dice}}\mathcal{L}_{\mathrm{dice}} + \lambda_{\mathrm{ce}}\mathcal{L}_{\mathrm{ce}}.
\]

Soft Dice for class \(c\) (Milletari et al., 2016), background excluded, empty GT classes omitted:

\[
\mathrm{Dice}_c = \frac{2\langle p_c, g_c\rangle + \varepsilon}{\|p_c\|_1 + \|g_c\|_1 + \varepsilon}.
\]

Inverse-frequency CE weights (optional) follow standard class-imbalance practice.

**Difference from “novel architectures.”** We do not replace this U-Net. Wrappers attach **in front of** or **instead of** it via config. `src/neurovisionx/models/unet3d.py` stays the baseline implementation.

---

## 1. Multimodal feature fusion

**Established.** BraTS pipelines almost always **concatenate** the four modalities at the input (early fusion). Squeeze-and-Excitation (Hu et al., CVPR 2018) rescales channels from a global descriptor:

\[
s = \sigma\bigl(W_2\,\delta(W_1\,\mathrm{GAP}(x))\bigr),\qquad
\tilde{x} = s \odot x.
\]

\(W_1,W_2\) are \(1\times1\) (here implemented as linear layers on the 4 modality channels). This is **not** a new fusion network (cf. DeepMedic dual pathways, Kamnitsas et al., 2017).

**What we test.** Whether an SE stem in front of the **same** 3D U-Net changes WT/TC/ET Dice or ET HD95 under a matched split. Config: `model.fusion: early | se`.

---

## 2. Attention-based feature refinement

**Established.** Additive attention gates (Oktay et al., MIDL 2018):

\[
\alpha = \sigma\bigl(\psi^\top (\delta(W_x x + W_g g))\bigr),\qquad
x' = \alpha \odot x.
\]

Phase 3 already implements 3D Attention U-Net as a **separate** class. It is an ablation arm, not an edit to 3D U-Net.

**What we test.** Same data/split/epochs as the baseline U-Net (H3-style matched comparison, limited to CNN vs gated CNN). Config: `model.name: unet3d | attention_unet3d`.

---

## 3. Boundary-aware loss

**Established.** Kervadec et al. (2021): if \(\phi_G\) is the signed distance function of a ground-truth region (negative inside, positive outside),

\[
\mathcal{L}_{\mathrm{bd}} = \frac{1}{|\Omega|}\sum_{x\in\Omega} p(x)\,\phi_G(x).
\]

Combined:

\[
\mathcal{L} = \mathcal{L}_{\mathrm{base}} + \lambda_{\mathrm{bd}}\mathcal{L}_{\mathrm{bd}}.
\]

Karimi and Salcudean (2020) motivate surface losses when Dice stays high while Hausdorff fails. Kervadec’s main empirical setting is not glioma ET.

**What we test (H1).** \(\lambda_{\mathrm{bd}}>0\) on **ET only** (`loss.boundary_classes: [3]`) vs baseline \(\lambda_{\mathrm{bd}}=0\), same U-Net. Primary metric: ET HD95; ET Dice must not drop by more than a pre-registered margin (1 Dice point). Config: `training.loss.boundary_weight`, `boundary_classes`.

SDF maps come from real labels (`data/distance.py`). They are never synthesized as a substitute for missing BraTS.

---

## 4. Monte Carlo dropout

**Established.** Gal and Ghahramani (ICML 2016): \(T\) stochastic forwards with dropout enabled at test time approximate the posterior predictive

\[
\bar{p}(y\mid x) \approx \frac{1}{T}\sum_{t=1}^T \mathrm{softmax}(f_{\theta,t}(x)).
\]

Predictive entropy (total uncertainty)

\[
H(\bar{p}) = -\sum_c \bar{p}_c \log \bar{p}_c.
\]

Mutual information (often used as epistemic) \(I = H(\bar{p}) - \mathbb{E}_t[H(p_t)]\).

Jungo et al. (2020) found **patient-aggregated** uncertainty more useful for detecting failed brain-tumor cases than voxel-wise maps for local correction.

**What we test (H2).** On a frozen checkpoint, \(T=10\) vs \(T=1\): Spearman \(\rho\) between patient mean entropy (on WT-candidate voxels) and patient Dice; AUPRC for detecting the worst Dice quartile; ECE of \(\max_c \bar{p}_c\) vs correctness (Mehrtash et al., 2020). Config: `training.uncertainty.method`, `samples`.

**TTA (implemented, optional).** `training.uncertainty.method: tta` averages softmax over up to eight spatial flips of the whole volume (depth / height / width), then unflips. Dropout stays off. This is standard test-time flipping, not a new augmenter. `scripts/analyze_reliability.py` uses TTA when that method is set; it still refuses smoke weights and missing BraTS. Default remains `mc_dropout`.

This is **not** a clinical confidence score.

---

## 5–6. Visualization, calibration, small regions, cost

- Entropy / error overlays are **research QC**, not explanations for clinicians.  
- ECE and NLL quantify calibration; they are reported only from real volumes.  
- Small/difficult ET: patients in the lowest quartile of ET voxel count; report Dice/HD95 in that stratum (pre-registered, not a search for a flattering subset).  
- Cost: parameter count, wall-clock per volume for \(T=1\) and \(T=10\). No FLOPs theatre without a measured timer.

---

## What is *not* claimed

- Novelty of SE, attention gates, boundary loss, or MC dropout.  
- That the combination is unpublished (combination studies exist; we do not survey that as a priority-claim).  
- Any Dice/HD95/ECE number until an official BraTS job finishes and writes `experiments/<run>/`.  
- Clinical validity.
