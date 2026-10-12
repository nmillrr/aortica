# PTB-XL regression fixtures

Four unmodified records from the PTB-XL 1.0.3 **test fold (10)**, each in
both of the corpus's native forms:

| ecg_id | Finding (scp_codes)        | `records500/` (500 Hz) | `records100/` (100 Hz) |
|--------|----------------------------|------------------------|------------------------|
| 9      | NORM, SR                   | `00009_hr`             | `00009_lr`             |
| 318    | AFIB, DIG                  | `00318_hr`             | `00318_lr`             |
| 180    | CLBBB, SR                  | `00180_hr`             | `00180_lr`             |
| 172    | CRBBB, LAFB, SR            | `00172_hr`             | `00172_lr`             |

`tests/api/test_preprocessing_regression.py` feeds the 500 Hz file through the
serving pipeline and checks the result against the model's output on the
100 Hz file read by the training data path. That is the comparison that
would have caught inference running a 100 Hz model on 500 Hz input.

Source: Wagner, P., Strodthoff, N., Bousseljot, R.-D., Samek, W., Schaeffter,
T. (2022). *PTB-XL, a large publicly available electrocardiography dataset*
(version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45 — licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
