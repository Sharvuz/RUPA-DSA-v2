# RUPA-DSA v0: implementation and Kaggle protocol

This project implements the narrow research claim as:

> Video-specific counterfactual normal reconstruction with category-aware residual semantic alignment.

The implementation deliberately does **not** claim that normal prototypes or residual learning are new by themselves.

## Implemented computation

For every video, SG-NM selects low-anomaly snippets with a reliability weight and conditions the learnable prototype seeds on that video. The resulting dynamic normal patterns reconstruct a counterfactual normal component:

```text
F_video --SG-NM(video-specific DNP)--> F_rec
R = F_video - F_rec

F_rec + DNP <--> normal text
R             <--> anomaly-category text
```

The routing score used for event/background pooling and test-time anomaly scoring is:

```text
S* = w_det S_det + w_rec S_rec + w_sem S_sem
```

with defaults `w_det=0.5`, `w_rec=0.3`, and `w_sem=0.2`. The CLI normalizes these three weights to sum to one.

The minimum RUPA losses are:

- `loss4`: residual-event/category alignment, applied only to anomalous videos;
- `loss5`: reconstructed-component/normal-text alignment;
- `dnp_normal_loss`: DNP/normal-text cosine alignment;
- the original detector, snippet semantic, consistency, gather, and text disentanglement losses.

All new loss weights and routing weights are CLI arguments in `src/ucf_option.py` and `src/xd_option.py`.

## Files

- `src/model.py`: reconstruction residual, dual semantic views, score fusion, and RUPA routing.
- `src/ucf_train.py`, `src/xd_train.py`: RUPA loss composition and resumable checkpoints.
- `src/ucf_test.py`, `src/xd_test.py`: fused RUPA score and residual-category logits at inference.
- `RUPA_DSA_Kaggle.ipynb`: Kaggle notebook that clones this repository directly.

## Kaggle input

Upload the feature archives as Kaggle Datasets, then attach the required dataset(s) to the notebook:

- UCF session: `UCFClipFeatures.zip`;
- XD session: `XDTrainClipFeatures.zip` and `XDTestClipFeatures.zip`.

The notebook clones `https://github.com/Sharvuz/RUPA-DSA-v0.git`, accepts either extracted `.npy` files or the ZIP archives, rebuilds the CSV paths, runs a GPU preflight, trains one benchmark per session, evaluates the best checkpoint, and creates a downloadable ZIP under `/kaggle/working/rupa_v0_artifacts`.

Enable **GPU** and **Internet** in Kaggle. Internet is required for this repository and the CLIP ViT-B/16 checkpoint.

## Recommended experiment order

Use identical seed, batch size, and epoch budget for a defensible ablation:

1. DSANet control: `--rupa-use false`.
2. Reconstruction residual without semantic routing: set routing weights to `0.7, 0.3, 0.0`.
3. Full RUPA-DSA v0: use the default fixed-routing weights `0.5, 0.3, 0.2`.
4. Remove DNP-normal alignment: `--loss-dnp-normal-weight 0`.
5. Remove residual-event alignment: `--loss-residual-weight 0`.

Report mean and standard deviation over at least three seeds. The code makes the hypothesis trainable; it does not by itself establish an accuracy improvement or novelty claim.
