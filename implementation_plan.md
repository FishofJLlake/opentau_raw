# Fix Pad Dimensions in Normalization and Add Quantile Normalization

This plan addresses the two required changes:
1. Fix padded dimensions in mean-std/min-max normalization.
2. Implement quantile normalization (q01, q99).

## User Review Required
None for the general approach, but please confirm the formula for quantile normalization maps the range `[q01, q99]` to `[-1, 1]` linearly (same as used in min-max), and does not use `clip()` so any values outside the percentiles can just normally exceed `[-1, 1]`.

## Proposed Changes

### Configuration
#### [MODIFY] [types.py](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/configs/types.py)
- Add `QUANTILE = "quantile"` to `NormalizationMode` enum class.

### Dataset Stats Calculation
#### [MODIFY] [compute_stats.py](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/datasets/compute_stats.py)
- In [get_feature_stats](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/datasets/compute_stats.py#175-193): Calculate `q01` and `q99` using `np.percentile(array, [1, 99], axis=axis, keepdims=keepdims)`.
- In [aggregate_feature_stats](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/datasets/compute_stats.py#281-329): Aggregate `q01` and `q99` using a weighted mean across the stats list (similar to how `mean` is handled).
- Adjust [_assert_type_and_shape](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/datasets/compute_stats.py#254-279) to support the new `q01` and `q99` keys in image checks.
#### [MODIFY] [test_compute_stats.py](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/tests/datasets/test_compute_stats.py)
- Fix any tests broken by the addition of the new stats items `q01` and `q99`.

### Normalization Logic
#### [MODIFY] [normalize.py](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/policies/normalize.py)
- Create a `pad_to_dim(vector: torch.Tensor, new_dim: int, pad_value: float)` component. This will be used exclusively when [stats](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/datasets/v21/convert_stats.py#81-112) are loaded into the buffers in [create_stats_buffers](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/policies/normalize.py#52-143).
- Modifying [create_stats_buffers](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/policies/normalize.py#52-143):
    - Use `pad_to_dim` to pad statistics parameters up to the expected model dimension (like `max_action_dim`) correctly.
    - Default `mean`, `min`, `q01` pad values to `0.0`.
    - Default `std`, `max`, `q99` pad values to `1.0`.
    - Support adding `q01` and `q99` buffers for `NormalizationMode.QUANTILE`.
- In `Normalize.forward`:
    - Add logic for `NormalizationMode.QUANTILE`: Compute `batch[key] = (batch[key] - q01) / (q99 - q01 + EPS)` followed by `batch[key] = batch[key] * 2 - 1`.
- In `Unnormalize.forward`:
    - Add logic for `NormalizationMode.QUANTILE`: Revert scaling with `batch[key] = (batch[key] + 1) / 2` followed by `batch[key] = batch[key] * (q99 - q01 + EPS) + q01`.

## Verification Plan
### Automated Tests
- Run `pytest tests/datasets/test_compute_stats.py`.
- Run a custom sanity script validating padding maintains exact 0s.
