# Marketing Channel Statistical Analysis

**Date:** 25 August 2026

**Analyst:** Marija Kavaliauskaite

**Dataset:** Campaign Effectiveness (Kaggle; simulated public campaign data)

**Period analyzed:** No date field; 5,500 independent campaign executions

## Executive Summary

The 10-channel dataset shows no CPA pair surviving Benjamini–Hochberg FDR correction, while 36 of 45 aggregate conversion-rate pairs do survive. **Email Marketing** has the lowest aggregate CPA ($247.54), **Snapchat Ads** has the highest ROAS (1.360), and **Google Search Ads** has the highest conversion rate (9.19%). Even the largest FDR-significant conversion-rate gap is only 0.37%, so practical importance is modest. Because the source is simulated and does not establish causal channel effects, the proposed $500K budget is a constrained planning heuristic, not a forecast.

## Key Findings

- Lowest aggregate CPA: **Email Marketing**, $247.54; its campaign-level mean CPA is $538.51 with a bootstrap 95% CI of $471.76–$604.48.
- Highest aggregate ROAS: **Snapchat Ads**, 1.360; aggregate profit is $55,115,914.
- Highest aggregate conversion rate: **Google Search Ads**, 9.19%.
- Pairwise CPA results: 2 of 45 uncorrected, 0 Bonferroni-significant, and 0 FDR-significant.
- Fisher conversion-rate results: 36 of 45 uncorrected, 32 Bonferroni-significant, and 36 FDR-significant.
- Across 90 tests, alpha 0.05 implies 4.5 expected false positives under the complete null. Conversion-rate differences remain statistically detectable after correction, but their small absolute size and the simulated, non-randomized source do not support a causal winner claim.

| Test family | Method | Significant results |
| --- | --- | ---: |
| CPA t-tests | Uncorrected | 2 |
| CPA t-tests | Bonferroni | 0 |
| CPA t-tests | FDR (BH) | 0 |
| Conversion Fisher tests | Uncorrected | 36 |
| Conversion Fisher tests | Bonferroni | 32 |
| Conversion Fisher tests | FDR (BH) | 36 |

## Data Adequacy / Power Analysis

The required simulation assumes campaign-level CPA has a standard deviation equal to 15% of baseline, which is narrower than the actual heterogeneous campaign-level distribution; its power estimates are planning illustrations, not a fitted model of this dataset. At 90 observations per group, the simulation gives:

| True CPA difference | Power at 90 days | 90 days sufficient? | Minimum tested days for ~80% power |
| ---: | ---: | :---: | ---: |
| 5% | 61.2% | No | 180 |
| 10% | 99.8% | Yes | 60 |
| 15% | 100.0% | Yes | 30 |
| 20% | 100.0% | Yes | 30 |

The current per-channel samples (523–576 campaigns) exceed 180 observations, but rows are campaign executions rather than dated daily observations. No FDR-significant CPA pair exists for an observed-effect power follow-up.

## $500K Monthly Budget Recommendation

| Channel | Allocation | Allocation % | Rationale |
| --- | ---: | ---: | --- |
| Influencer Marketing | $75,000 | 15.0% | Heuristic 50/50 CPA–ROAS rank; CPA rank 2, ROAS rank 2; constrained to 5%–15%. |
| Snapchat Ads | $75,000 | 15.0% | Heuristic 50/50 CPA–ROAS rank; CPA rank 4, ROAS rank 1; constrained to 5%–15%. |
| Email Marketing | $70,600 | 14.1% | Heuristic 50/50 CPA–ROAS rank; CPA rank 1, ROAS rank 5; constrained to 5%–15%. |
| Instagram Ads | $70,600 | 14.1% | Heuristic 50/50 CPA–ROAS rank; CPA rank 3, ROAS rank 3; constrained to 5%–15%. |
| Facebook Ads | $48,500 | 9.7% | Heuristic 50/50 CPA–ROAS rank; CPA rank 7, ROAS rank 4; constrained to 5%–15%. |
| Twitter Ads | $39,700 | 7.9% | Heuristic 50/50 CPA–ROAS rank; CPA rank 5, ROAS rank 8; constrained to 5%–15%. |
| YouTube Ads | $39,700 | 7.9% | Heuristic 50/50 CPA–ROAS rank; CPA rank 6, ROAS rank 7; constrained to 5%–15%. |
| Affiliate Marketing | $30,900 | 6.2% | Heuristic 50/50 CPA–ROAS rank; CPA rank 9, ROAS rank 6; constrained to 5%–15%. |
| Google Search Ads | $25,000 | 5.0% | Heuristic 50/50 CPA–ROAS rank; CPA rank 8, ROAS rank 9; constrained to 5%–15%. |
| LinkedIn Ads | $25,000 | 5.0% | Heuristic 50/50 CPA–ROAS rank; CPA rank 10, ROAS rank 10; constrained to 5%–15%. |
| **Total** | **$500,000** | **100.0%** | Constrained heuristic total |

The allocation combines descriptive CPA and ROAS ranks at 50% each, enforces a 5% floor and 15% ceiling, and rounds to $100 while totaling exactly $500,000. Larger allocations indicate comparatively favorable historical point estimates, not proven incremental returns. The ceiling limits overcommitment where corrected evidence is weak.

## Statistical Caveats

- Kaggle describes this as a **simulated** public dataset; it is not audited company performance and has no date field, geography, targeting assignment, or channel-selection process.
- Campaigns are observational groupings, not randomized channel assignments. Correlation does not establish that shifting spend will cause the same outcome.
- Welch t-tests compare campaign-level CPA means; CPA is right-skewed and influenced by campaign scale. Fisher tests aggregate clicks/conversions and assume independent attempts, which cannot be verified from the source.
- Bonferroni controls family-wise error and is conservative; FDR controls the expected false-discovery proportion and is the primary screening rule here. Neither correction fixes confounding or measurement quality.
- Confidence intervals quantify sampling uncertainty under the available campaigns, not data-generation or model uncertainty. Statistical significance and practical value are different questions.
- The power simulation uses a stylized normal CPA model and fixed 15% variability. Historical results cannot validate a future $500K budget or predict saturation.

## Next Steps

1. Run randomized, geo- or audience-matched incrementality tests before materially reallocating live spend.
2. Collect daily campaign data with audience, placement, attribution window, and margin-adjusted revenue for at least the sample durations indicated above.
3. Monitor CPA, ROAS, conversion quality, and marginal return weekly; hold the 5%–15% guardrails until replicated evidence supports changes.
4. Re-run the same corrected analysis on observed company data and pre-register the primary metric and decision thresholds.
