Do Russian Real-Estate Descriptions Improve Asking-Price Prediction?

Nikita Bagramov and Alfiya IskhakovaНикита Баграмов и Альфия ИсхаковаNatural Language Processing & LLMs, autumn 202613 November 2026

Abstract

We study whether Russian apartment advertisements add predictive information beyond structured housing characteristics. A public Saint Petersburg listing collection is audited and converted into 30,775 distinct-ID observations. Shared addresses and repeated text templates are kept in the same partition. We compare tabular gradient boosting, TF-IDF, frozen Qwen3 embeddings, dictionary features, and evidence-checked Qwen3 attribute extraction. A separate, equal-training-budget experiment compares generative and supervised approaches. The model selected by development error is Extended + embedding residual, with test log MAE 0.1551 and median absolute percentage error 12.14%. We report group-bootstrap intervals, feature ablations, and limitations of text-derived explanations. The target is an asking price, not a transaction price. No global state-of-the-art claim is made.

Code and reproducibility materials: https://github.com/NikiBagramov/avito-nlp-price-prediction.

1. Introduction

Housing listings mix measurable attributes with claims such as renovated interiors, furniture remaining after sale, waterfront views, and transaction urgency. These claims are natural-language observations rather than reliable database fields. The practical NLP question is whether their representation improves prediction on previously unseen addresses, and whether interpretable extraction preserves useful information.

We formulate the main task as supervised regression of the logarithm of the advertised total price. Our primary hypothesis is that textual descriptions contain incremental information beyond a basic hedonic table. A stronger control retains the additional structured fields already present in the source. Beating a restricted baseline alone would not demonstrate that text outperforms all available tabular information.

The contribution is a reproducible Russian-language benchmark with explicit duplicate handling, a fixed address/template holdout, multiple text representations, and an evidence admission rule for generative attributes. We adapt published families of methods rather than claim a new foundation architecture. The analysis of factors concerns predictive associations; it cannot estimate the causal return from renovating a particular apartment.

1.1. Team members and responsibilities

The project team supplied by the requester consists of Nikita Bagramov and Alfiya Iskhakova. Individual contributions have not been documented by the team. Code, experiment orchestration, analysis drafts, and report preparation in this version were produced with Codex assistance. Dataset collection is credited to Veronika Vyalkova. The team must verify the results, understand the implementation, and supply an accurate contribution statement before presenting this as its completed coursework; no independent manual annotation or instructor approval is asserted.

2. Related Work

Zhang, Li and Branco [1] combine housing descriptions and tabular variables on 10,251 Canadian listings. They compare TF-IDF, Word2Vec and BERT with four regression families. Their grid-search GB + TF-IDF configuration reports R² = 0.7214 and normalized RMSE = 0.0192; these are validation results on their scale, not ruble test errors. This motivates our text-only, table-only and hybrid comparisons. Our Russian collection and group holdout differ from their random 90/10 partition. The linked original repository returned 404 when checked, so our implementation is independent.

Geerts et al. [2] evaluate prompting for real-estate appraisal. In their King County comparison with ten mixed examples, MAPE is 0.1378 for LGBM, 0.1861 for GPT-4o-mini and 0.2995 for Llama 3.2:3B. Their public MIT repository supplies prompts and baselines. Our zero-shot and three-example Qwen experiment adapts the principle of retrieving similar training properties. It is not a reproduction of their ten-example configuration, geographic features, original datasets, or commercial model results.

Qwen3 Embedding [3] provides frozen multilingual text representations; the 0.6B variant produces 1024-dimensional vectors. Qwen3 [4] also offers a small generative model suitable for a 6 GB GPU. We compare an embedding representation with explicit structured extraction from Qwen3-1.7B. Neither the foundation-model benchmarks nor multilingual retrieval scores establish performance on Russian asking-price regression.

CatBoost [5] handles mixed numerical and categorical inputs and is used as the main supervised control. TF-IDF and Ridge [7] provide an inexpensive lexical comparator. SHAP [6] supplies additive prediction attributions; feature dependence and the observational collection prevent a causal interpretation. The exact dataset and split have no independently verified published record in this project. The strongest baseline we implement is not equivalent to an established state of the art.

3. Model Description

3.1. Baselines and text representations

Let x be structured housing features, t the cleaned title plus description, and z = log(price). The constant baseline predicts the training median of z. The basic table contains 11 numerical fields (areas, rooms, floors, construction year, ceiling height, and three metro times) and six categorical fields (district, three metro stations, city, and house type). Missing numerical values receive a dedicated -1 sentinel; missing categories receive a distinct token. URLs, IDs, exact addresses and raw parameter strings are excluded.

The extended table adds 11 source fields including renovation, furniture, finishing, windows, balcony, bathroom, parking, completion date, sale method, transaction type and development name. Dictionary features are eight explicit indicators for renovation types, furniture, water views, urgency, legal phrases, and developer advertising. They are transparent but weak: negation and generic project descriptions can trigger false matches.

The lexical representation is word/unigram-bigram TF-IDF, sublinear term frequency, minimum document frequency three, and at most 40,000 features. Ridge minimizes squared log-price residuals plus alpha times the squared coefficient norm. Alpha is selected from {1, 10, 100} by development log MAE. A convex table/text blend searches 21 weights from zero to one using development data only.

3.2. Cross-fitted residual learning

A text-only estimator can relearn apartment size and location already present in the table. To measure incremental signal, we fit a tabular model on two thirds of training groups and predict the remaining third, rotating across three group folds. For training observation i, r_i = z_i - f_out-of-fold(x_i). Ridge then maps the text representation to r. The final prediction is z_hat = f_all-train(x) + g(text). This avoids fitting residual targets from the same observations used by each fold model. Fold vocabularies are not used to score those folds: the residual regressor and its vocabulary are subsequently fitted on train only; dev and test remain outside vocabulary fitting.

For dense representations we use frozen Qwen3-Embedding-0.6B, last-token pooling with left padding, a 512-token limit and L2 normalization. The downstream Ridge model uses the same alpha grid. No housing labels fine-tune the embedding model. The residual hybrid uses the same cross-fitted tabular residuals as the lexical hybrid. Combining representations through residual learning was chosen for resource feasibility and comparability, rather than a 1024-feature boosted-tree search.

3.3. Generative semantic extraction and direct prediction

Qwen3-1.7B extracts five attributes: renovation state, furniture included/excluded, water view, explicit sale urgency, and explicit legal encumbrance status. Unknown is a separate value. A claim about a development in general is not intended to describe the particular apartment. Each non-unknown value must include an exact quotation from the supplied excerpt. A syntactically valid but unsupported quotation becomes unsupported; invalid JSON is exposed as invalid. Both are retained as explicit categorical statuses, not silently interpreted as absence.

The prompt instructs the model to treat the advertisement as untrusted data. Generation is deterministic greedy decoding, non-thinking mode, up to 256 output tokens, with a maximum 1536-token input. Texts exceeding 2400 characters use the first 1800 and last 600 characters with an explicit truncation marker. Direct price prediction uses a separate JSON prompt and 32 output tokens. Three-shot examples are retrieved only from the fixed training subset using scaled area/room/floor/metro-time distances plus a district mismatch penalty. An invalid direct answer receives the training price median; failure counts are reported.

After the initial extraction prompt showed near-total abstention, we tested a shorter English instruction with two assistant-reviewed positive excerpts from training IDs 7929104599 and 7866165689. These few-shot examples demonstrate renovation, furniture, and explicit no-encumbrance evidence. This is an exploratory prompt revision rather than a clean preregistered comparison. The initial prompt is retained as a separate baseline, and no independent human annotation is claimed. The same schema, quotation rule, subset and decoding budget are kept.

4. Dataset

The source is Veronika Vyalkova’s public Kaggle collection of Saint Petersburg Avito listings [8], described as collected in February 2026. The downloaded workbook contains 32,967 rows and 51 columns. The archive and extracted files are identified by SHA-256 checksums in data_provenance.json. The author lists CC BY-NC-SA 4.0. Raw source data are excluded from the source repository and downloaded from the original page. Any released derivative data must preserve attribution and applicable noncommercial/share-alike terms. The code license does not replace dataset or model licenses.

| Audit item | Observed value |
| --- | --- |
| Workbook rows / columns | 32,967 / 51 |
| Repeated listing IDs before filtering | 1915 |
| Missing/invalid/short-text exclusions | 308 |
| Prepared distinct-ID observations | 30,775 |
| Address/template components | 8,610 |
| Train / dev / test | 21,718 / 4,582 / 4,475 |
| Texts with masked monetary amounts | 1468 |
| Median / 99th-percentile text length | 1244 / 3254 characters |

A valid observation requires a listing ID, positive price and total area, and at least 80 description characters. After this filter, the first valid record per ID is retained. No target-quantile trimming removes expensive apartments. Original price values range from 750,000 to about 2.86 billion rubles; unusually expensive properties therefore influence RMSE. Sampling and missingness from platform scraping remain sources of bias. The city field sometimes contains street or water-area fragments and is treated as noisy text rather than an authoritative geography.

Preprocessing normalizes Unicode, replaces Latin lookalikes within mixed Cyrillic tokens, masks currency-denominated amounts, removes an exact spelling of the target price with optional separators, and masks telephone patterns. It preserves area, floors and construction dates. This does not prove that every indirect or verbal price clue has been removed. Two advertisements are grouped if they share a normalized nonempty address or an identical text fingerprint after removing digits and nonletters. Transitive links form 8,610 components; the largest contains 478 listings.

GroupShuffleSplit with seeds 42 and 43 creates approximately 70/15/15 partitions by groups. The actual row counts differ slightly. Tests assert that IDs are unique and group, address and template keys do not cross partitions. This does not eliminate all near-duplicates, developer-wide stylistic patterns, or differently written versions of one address. It is a spatial/template holdout within one time snapshot, not a temporal or cross-city evaluation.

Expensive generation uses a fixed uniform row sample within each split: 1400 train, 300 dev and 400 test, sampled with seed 2026 without conditioning on prices. Every supervised comparator in the small benchmark is retrained using the same 1400 training rows. There is no independent human gold annotation for the extracted attributes. The existing renovation field is an incomplete silver reference; disagreement can reflect either model mistakes or inconsistent source fields.

5. Experiments

5.1. Metrics

The primary metric is mean absolute log-price error: L_log = (1/n) sum_i |log(p_i) - z_hat_i|. It weights multiplicative deviations more evenly than absolute ruble errors. We also report MAE = (1/n) sum_i |p_i - exp(z_hat_i)|, RMSE in rubles, R² on the original price scale, and MdAPE = median_i |p_i - p_hat_i| / p_i. MdAPE describes the typical relative error, not the average relative error and not an accuracy score. Exponentiating a squared-log-loss prediction does not make it an unbiased mean price.

5.2. Experiment setup and selection

CatBoost uses CPU, four threads, learning rate 0.04, squared-error loss on log price, depths {6, 8}, at most 1500 iterations and development early stopping after 100 non-improving iterations. Depth is selected by development log MAE. Early stopping itself monitors the configured RMSE objective. Tabular controls and the LLM-feature regressor are checked with seeds 42, 43 and 44 at the selected number of trees. The headline prediction uses seed 42 rather than an unannounced ensemble. Development data are not merged back into training.

The local machine has an NVIDIA RTX 3060 Laptop GPU with 6144 MiB VRAM. Embedding and generative jobs run sequentially to respect that memory limit. Generation uses float16, SDPA attention and batch size eight. Both Qwen repositories are pinned by immutable revisions in model_provenance.json. Library versions are recorded in environment.json. Commercial LLMs and paid endpoints are not included because no approved API access/budget was supplied for this execution. We do not imply that a 1.7B model is representative of commercial-model quality.

Uncertainty uses a paired bootstrap over test components, resampling 1000 times with seed 2026. Each sampled component retains all its observed rows; differences are row-weighted within a draw. Intervals condition on this split and these trained models. Multiple exploratory method comparisons and model-development exposure are not corrected as a confirmatory study. The original protocol was written before training, but implementation continued after some earlier test results became visible; later methods still use dev for parameter selection. This weakens confirmatory claims and is documented in PROTOCOL.md.

Two further diagnostic controls were added during development: an arithmetic mean of the identical three retrieved training prices used by the direct LLM, and text residuals on the extended table. The latter repeats three-fold group cross-fitting for the extended CatBoost and selects a residual weight from 0 to 1 in steps of 0.05 on dev. They are explicitly exploratory; their inclusion does not retrospectively turn earlier exploratory results into a confirmatory state-of-the-art test.

6. Results

6.1. Full-data comparison

| Method | Dev log MAE | Test log MAE | Test MdAPE | MAE, m RUB |
| --- | --- | --- | --- | --- |
| Train median | 0.5176 | 0.5368 | 42.85% | 10.635 |
| Basic table | 0.1561 | 0.1648 | 12.33% | 3.949 |
| Extended table | 0.1311 | 0.1606 | 12.57% | 3.875 |
| Basic + dictionary | 0.1516 | 0.1602 | 12.10% | 3.890 |
| TF-IDF + Ridge | 0.2735 | 0.2592 | 21.82% | 5.886 |
| Basic / TF-IDF blend | 0.1545 | 0.1562 | 11.82% | 3.804 |
| Basic + TF-IDF residual | 0.1477 | 0.1530 | 11.74% | 3.789 |
| Qwen embedding + Ridge | 0.3173 | 0.2975 | 23.58% | 6.572 |
| Basic + embedding residual | 0.1411 | 0.1590 | 12.02% | 3.984 |
| Extended + TF-IDF residual | 0.1299 | 0.1557 | 12.23% | 3.795 |
| Extended + embedding residual | 0.1266 | 0.1551 | 12.14% | 3.803 |

Development-only selection chooses Extended + embedding residual. Its held-out MAE is 3.803 million RUB and R² is 0.782. The table includes all evaluated methods, including those with worse test performance. The lowest observed test error is descriptive; it is not used to replace the dev-selected model after the fact.

The lexical residual hybrid changes log MAE relative to the basic table by -0.0119, with paired group-bootstrap 95% interval [-0.0211, -0.0010]. This supports incremental textual signal relative to that restricted control on this holdout. It does not establish superiority over the extended structured control, future data, commercial systems, or published benchmarks.

Relative to the extended control, Extended + TF-IDF residual has delta log MAE -0.00489 and 95% group-bootstrap interval [-0.008737, -0.001040]. The interval lies below zero on this split, supporting an incremental reduction, subject to the exploratory design.

Relative to the extended control, Extended + embedding residual has delta log MAE -0.00557 and 95% group-bootstrap interval [-0.011196, +0.000048]. The interval includes zero, so this comparison does not establish a reliable improvement.

![Figure 1. Full-data comparison on the fixed address/template holdout.](../../artifacts/figures/full_metrics.png)

6.2. Equal-budget generative comparison

| Method | Dev log MAE | Test log MAE | Test MdAPE | MAE, m RUB |
| --- | --- | --- | --- | --- |
| Train median | 0.5172 | 0.5086 | 41.91% | 9.976 |
| Basic table | 0.1683 | 0.1725 | 12.55% | 3.906 |
| Extended table | 0.1594 | 0.1623 | 12.00% | 3.944 |
| Basic + dictionary | 0.1739 | 0.1729 | 11.41% | 4.113 |
| TF-IDF + Ridge | 0.3340 | 0.3557 | 28.97% | 7.568 |
| Basic / TF-IDF blend | 0.1660 | 0.1698 | 11.83% | 3.785 |
| Basic + TF-IDF residual | 0.1668 | 0.1691 | 12.03% | 3.841 |
| Qwen embedding + Ridge | 0.3615 | 0.3303 | 27.44% | 7.071 |
| Basic + embedding residual | 0.1593 | 0.1710 | 12.38% | 3.856 |
| Basic + LLM attributes | 0.1742 | 0.1805 | 13.06% | 4.079 |
| Basic + few-shot LLM attributes | 0.1711 | 0.1730 | 12.48% | 3.839 |
| Qwen direct: zero-shot | 1.9981 | 2.0128 | 73.71% | 13.282 |
| Qwen direct: 3-shot | 0.4610 | 0.4207 | 30.12% | 9.167 |
| Three retrieved neighbors | 0.2915 | 0.2951 | 24.24% | 7.178 |

All methods above use the same 1400 supervised training rows or the same permitted retrieval pool. The small-sample scores should not be compared as if they had been trained on the full 21,718-row training partition. The direct LLM controls receive examples in the prompt rather than gradient training. Attribute extraction is label-free at inference; the downstream CatBoost still requires price labels.

Across 2100 generated attribute records, JSON parsing succeeds for 100.00% and all five evidence checks pass for 86.86%. These are formatting/evidence-string checks, not semantic accuracy. On the test subset, the source renovation field is present for 257 rows; 1 have a comparable non-unknown admitted prediction. Agreement on that selected overlap is 100.00%. Coverage and this conditional agreement must be read together. Missing or unknown cases are not counted as correct negatives.

| Attribute | Admitted test coverage |
| --- | --- |
| renovation | 0.25% |
| furniture | 0.25% |
| water_view | 0.00% |
| urgency | 0.25% |
| legal | 0.00% |

For the revised few-shot extractor, JSON validity is 92.43% and all-evidence validity is 43.33%. The test renovation comparison contains 106 admitted, non-unknown cases out of 257 available source labels; conditional agreement is 65.09%. This conditional silver agreement is not a population accuracy estimate.

| Few-shot attribute | Admitted test coverage |
| --- | --- |
| renovation | 39.50% |
| furniture | 15.25% |
| water_view | 10.50% |
| urgency | 11.00% |
| legal | 16.75% |

Qwen direct: zero-shot produces 147 invalid test answers out of 400 and 104 invalid dev answers out of 300. The reported price errors include the declared training-median fallback. Poorly scaled but positive numeric answers remain errors; they are not silently repaired using test labels.

Qwen direct: 3-shot produces 36 invalid test answers out of 400 and 28 invalid dev answers out of 300. The reported price errors include the declared training-median fallback. Poorly scaled but positive numeric answers remain errors; they are not silently repaired using test labels.

![Figure 2. Equal-training-budget comparison. Direct price outputs include the declared fallback for invalid JSON or invalid prices.](../../artifacts/figures/subset_metrics.png)

6.3. Which factors matter for prediction?

We report mean absolute SHAP values on a fixed 512-row test sample for the extended table. They rank contributions to this fitted predictor on its log-price scale. Correlated features can share or redistribute importance: total and living area, district and metro, or renovation and finishing should not be read as independent economic effects.

![Figure 3. Feature associations of the extended table. The ranking is predictive, not a causal estimate of price premiums.](../../artifacts/figures/factors.png)

| Retrained ablation | Test log MAE | Change from extended table |
| --- | --- | --- |
| without_area | 0.1752 | +0.0145 |
| without_location | 0.1832 | +0.0226 |
| without_condition | 0.1570 | -0.0036 |
| without_building | 0.1667 | +0.0060 |

Ablations retrain at the extended control’s selected depth and iteration budget; their hyperparameters are not retuned individually. They test how readily other available fields compensate for a removed feature group. A small or negative change can arise from redundancy, noise, or regularization and does not prove that the physical attribute is economically irrelevant.

6.4. Error analysis and validity limits

| Observed price band | Test n | Log MAE | MdAPE |
| --- | --- | --- | --- |
| <10m | 1519 | 0.1289 | 10.02% |
| 10-20m | 1733 | 0.1376 | 11.41% |
| 20-50m | 1003 | 0.1835 | 14.11% |
| >=50m | 220 | 0.3439 | 29.05% |

Ruble errors disproportionately reflect the expensive tail. The saved largest-error table identifies IDs, observed prices, predictions, and districts without republishing addresses or full descriptions. It supports case review but does not establish which advertisements are erroneous, fraudulent, or mispriced. Ambiguous legal statements, boilerplate about the entire development, conditional furniture offers, and generic proximity to water are expected failure modes for extraction; exact-quote checking cannot resolve those semantic distinctions.

Reviewing large errors of the extended structured control identifies target-granularity issues. Listing 7909276909 is titled as a share in a two-room apartment; its description refers to a 12.5 m² room, while the source total-area field is 565 m² and the asking price is 2.35 million RUB. Listing 7878369739 is a one-room apartment with source total area 310 m², living area 31 m² and asking price 9.8 million RUB. These combinations motivate source review, but do not establish the true area or authorize automatic repair. Neither record is removed after observing its error. There are 128 titles containing the share marker in the prepared collection, including 94 in train. Selling an interest and selling an entire apartment are different price targets even when some housing fields coincide.

![Figure 4. Asking-price predictions of the model selected on dev; both axes use log10 RUB.](../../artifacts/figures/full_scatter.png)

Remaining limitations include only one snapshot and city region, incomplete sale/renovation metadata, noisy geographic parsing, undetected near-duplicates, tokenizer truncation, and no independent human attribute labels. No sale-price comparison, causal experiment, commercial LLM evaluation, or externally validated SOTA claim is available. The observed associations should be used to formulate further research rather than advise a purchase or infer a seller’s motives.

7. Conclusion

We built an auditable Russian-language asking-price benchmark, separated addresses and repeated templates across partitions, and compared lexical, dense and generative text methods. A lexical residual correction improves the extended tabular control on this split with a negative group-bootstrap interval. Development selection instead chooses the embedding residual model, whose improvement interval includes zero. Small-model direct pricing and semantic extraction do not outperform the strong supervised controls. These exploratory results support studying incremental textual signal without assuming LLM superiority. Next steps are an independent manual attribute set, temporal/cross-city validation, and a budgeted comparison with a stronger open or commercial LLM.

References

[1] H. Zhang, Y. Li, P. Branco. Describe the house and I will tell you the price. Natural Language Engineering 30, 661-695, 2024 (online 2023). https://doi.org/10.1017/S1351324923000360.

[2] M. Geerts et al. On the Performance of LLMs for Real Estate Appraisal. ECML-PKDD 2025, proceedings 2026, 195-211. https://arxiv.org/abs/2506.11812.

[3] Y. Zhang et al. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. 2025. https://arxiv.org/abs/2506.05176.

[4] A. Yang et al. Qwen3 Technical Report. 2025. https://arxiv.org/abs/2505.09388.

[5] L. Prokhorenkova et al. CatBoost: unbiased boosting with categorical features. NeurIPS 2018. https://arxiv.org/abs/1706.09516.

[6] S. Lundberg, S.-I. Lee. A Unified Approach to Interpreting Model Predictions. NeurIPS 2017. https://arxiv.org/abs/1705.07874.

[7] scikit-learn. Official documentation: text feature extraction and Ridge regression. Accessed 13 November 2026. https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction.

[8] V. Vyalkova. Saint Petersburg Avito Listings. Kaggle, source snapshot described as February 2026. https://www.kaggle.com/datasets/veronikavyalkova/saint-petersburg-avito-listings.

[9] NLP course project template. Structure followed in this report. https://www.overleaf.com/read/vyqdnhjfjnwh.

Appendix A. Reproducibility details

Exact model revisions: embedding 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3; generation 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e. Both model families are Apache-2.0. Python/library versions used in this execution: pandas 2.2.3, numpy 1.26.4, scipy 1.14.1, scikit-learn 1.7.2, catboost 1.2.8, torch 2.6.0.dev20241112+cu121, transformers 4.57.6, accelerate 1.12.0, huggingface-hub 0.36.0, safetensors 0.4.5, matplotlib 3.9.3. Extraction used the existing local copy of the pinned generation weights; a fresh run can download the same repository revision.

Commands, full prompts, invariant tests, source hashes, split membership, prediction tables, bootstrap intervals, seed-level predictions, runtime manifests, and data/model cards accompany the source distribution. Local raw generation logs contain excerpts and are excluded from the public source archive. Aggregate diagnostics and prompts suffice to regenerate them from the original licensed source. A new run should preserve source checksums and split IDs; dependency and hardware changes may produce small numeric differences.
