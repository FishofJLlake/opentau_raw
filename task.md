# Task Checklist

## 1. Implement Padding Fix for Normalization
- [ ] Add `pad_to_dim(vector, new_dim, pad_value)` to [src/opentau/policies/normalize.py](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/policies/normalize.py).
- [ ] Update [create_stats_buffers](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/policies/normalize.py#52-143) in [normalize.py](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/policies/normalize.py) to use `pad_to_dim` on `mean`, `std`, `min`, `max` before loading them into buffers (pad mean/min with 0.0, std/max with 1.0).

## 2. Implement Quantile Normalization
- [ ] Add `q01` and `q99` extraction using `np.percentile` to [get_feature_stats](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/datasets/compute_stats.py#175-193) in [src/opentau/datasets/compute_stats.py](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/datasets/compute_stats.py).
- [ ] Aggregate `q01` and `q99` with weighted mean in [aggregate_feature_stats](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/datasets/compute_stats.py#281-329) within [compute_stats.py](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/tests/datasets/test_compute_stats.py).
- [ ] Add `NormalizationMode.QUANTILE` enum in [src/opentau/configs/types.py](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/configs/types.py).
- [ ] Support `QUANTILE` mode in [create_stats_buffers](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/src/opentau/policies/normalize.py#52-143) (add `q01` buffer padded with 0.0 and `q99` padded with 1.0).
- [ ] Add `QUANTILE` calculation in `Normalize.forward` mapping q01->-1 and q99->1.
- [ ] Add `QUANTILE` calculation in `Unnormalize.forward`.

## 3. Verify
- [ ] Write a test script similar to [tmp_test_norm.py](file:///d:/Downloads/OpenTau-main%20%286%29/OpenTau-main/tmp_test_norm.py) to verify the pad fix behaves correctly.
