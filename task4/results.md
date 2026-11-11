
Rows repeated over seeds (mean +- std):
| variant | train | change | seeds | valA | cameraA | testB | testH |
|---|---|---|---|---|---|---|---|
| B1 | A | baseline | 3 | 0.548 +- 0.084 | 0.490 +- 0.022 | 0.437 +- 0.091 | 0.458 +- 0.024 |
| L_best | A | best epoch weights | 3 | 0.548 +- 0.084 | 0.483 +- 0.029 | 0.460 +- 0.059 | 0.489 +- 0.059 |
| L_gap | A | global average pooling | 3 | 0.524 +- 0.045 | 0.483 +- 0.055 | 0.276 +- 0.000 | 0.442 +- 0.034 |
| L_cos | A | cosine schedule, 30 epochs | 3 | 0.655 +- 0.034 | 0.541 +- 0.020 | 0.356 +- 0.016 | 0.447 +- 0.034 |
| B2 | A | baseline | 3 | 0.726 +- 0.061 | 0.624 +- 0.037 | 0.598 +- 0.033 | 0.519 +- 0.051 |
| L_color | A | colour jitter | 3 | 0.607 +- 0.058 | 0.567 +- 0.067 | 0.448 +- 0.102 | 0.453 +- 0.045 |
| L_drop | A | dropout 0.3 | 3 | 0.631 +- 0.034 | 0.529 +- 0.029 | 0.379 +- 0.056 | 0.414 +- 0.017 |
| L_adamw | A | AdamW, weight decay 1e-2 | 3 | 0.595 +- 0.017 | 0.498 +- 0.019 | 0.460 +- 0.114 | 0.464 +- 0.021 |
| L_ls | A | label smoothing 0.1 | 3 | 0.595 +- 0.045 | 0.493 +- 0.032 | 0.448 +- 0.113 | 0.500 +- 0.045 |
| M_aug | A | geometric augmentation | 3 | 0.702 +- 0.073 | 0.698 +- 0.053 | 0.667 +- 0.033 | 0.544 +- 0.057 |
| L_dataB | AB | training images +B | 3 | 0.595 +- 0.017 | 0.462 +- 0.020 | - | 0.500 +- 0.034 |
| L_dataH | AH | training images +H | 3 | 0.607 +- 0.029 | 0.595 +- 0.026 | 0.678 +- 0.043 | - |
| L_dataBH | ABH | training images +B+H | 3 | 0.500 +- 0.050 | 0.533 +- 0.015 | - | - |
| L_combo1 | A | colour jitter + best epoch | 3 | 0.631 +- 0.034 | 0.581 +- 0.029 | 0.529 +- 0.059 | 0.475 +- 0.030 |
| L_combo2 | A | colour jitter + label smoothing + best epoch | 3 | 0.631 +- 0.034 | 0.576 +- 0.028 | 0.517 +- 0.056 | 0.461 +- 0.032 |
| M_combo | A | augmentation + cosine 30 + best epoch | 3 | 0.750 +- 0.029 | 0.733 +- 0.047 | 0.724 +- 0.056 | 0.625 +- 0.027 |
| M_frozen | A | frozen backbone | 3 | 0.559 +- 0.017 | 0.581 +- 0.032 | 0.632 +- 0.043 | 0.450 +- 0.025 |
| M_cos | A | cosine schedule, 30 epochs | 3 | 0.691 +- 0.045 | 0.626 +- 0.038 | 0.586 +- 0.028 | 0.531 +- 0.016 |
| M_best | A | best epoch weights | 3 | 0.631 +- 0.034 | 0.614 +- 0.021 | 0.552 +- 0.049 | 0.528 +- 0.051 |
| M_dataB | AB | training images +B | 3 | 0.548 +- 0.034 | 0.581 +- 0.065 | - | 0.608 +- 0.012 |
| M_dataH | AH | training images +H | 3 | 0.809 +- 0.067 | 0.669 +- 0.036 | 0.678 +- 0.016 | - |
| M_dataBH | ABH | training images +B+H | 3 | 0.619 +- 0.017 | 0.669 +- 0.009 | - | - |
| M_aug_cos | A | augmentation + cosine 30 | 3 | 0.762 +- 0.017 | 0.767 +- 0.032 | 0.724 +- 0.000 | 0.686 +- 0.020 |

Single runs and their change against the parent mean:
| variant | train | change | seed | valA | cameraA | testB | testH |
|---|---|---|---|---|---|---|---|
| B1 | A | baseline | 0 | 0.6071 | 0.5214 | 0.4828 | 0.4250 |
| B1 | A | baseline | 1 | 0.6071 | 0.4714 | 0.3103 | 0.4750 |
| B1 | A | baseline | 2 | 0.4286 | 0.4786 | 0.5172 | 0.4750 |
| L_best | A | best epoch weights (+0.059 valA) | 0 | 0.6071 | 0.5214 | 0.4828 | 0.4250 |
| L_best | A | best epoch weights (+0.059 valA) | 1 | 0.6071 | 0.4500 | 0.3793 | 0.5667 |
| L_best | A | best epoch weights (-0.119 valA) | 2 | 0.4286 | 0.4786 | 0.5172 | 0.4750 |
| L_gap | A | global average pooling (-0.083 valA) | 0 | 0.4643 | 0.5071 | 0.2759 | 0.4417 |
| L_gap | A | global average pooling (+0.024 valA) | 1 | 0.5714 | 0.4071 | 0.2759 | 0.4000 |
| L_gap | A | global average pooling (-0.012 valA) | 2 | 0.5357 | 0.5357 | 0.2759 | 0.4833 |
| L_cos | A | cosine schedule, 30 epochs (+0.131 valA) | 0 | 0.6786 | 0.5643 | 0.3448 | 0.4417 |
| L_cos | A | cosine schedule, 30 epochs (+0.131 valA) | 1 | 0.6786 | 0.5143 | 0.3448 | 0.4917 |
| L_cos | A | cosine schedule, 30 epochs (+0.059 valA) | 2 | 0.6071 | 0.5429 | 0.3793 | 0.4083 |
| B2 | A | baseline | 0 | 0.6429 | 0.5714 | 0.6207 | 0.4833 |
| B2 | A | baseline | 1 | 0.7857 | 0.6500 | 0.6207 | 0.5917 |
| B2 | A | baseline | 2 | 0.7500 | 0.6500 | 0.5517 | 0.4833 |
| L_color | A | colour jitter (-0.012 valA) | 0 | 0.5357 | 0.4714 | 0.4138 | 0.4250 |
| L_color | A | colour jitter (+0.059 valA) | 1 | 0.6071 | 0.6143 | 0.3448 | 0.4167 |
| L_color | A | colour jitter (+0.131 valA) | 2 | 0.6786 | 0.6143 | 0.5862 | 0.5167 |
| L_drop | A | dropout 0.3 (+0.059 valA) | 0 | 0.6071 | 0.5286 | 0.3793 | 0.3917 |
| L_drop | A | dropout 0.3 (+0.059 valA) | 1 | 0.6071 | 0.5643 | 0.3103 | 0.4167 |
| L_drop | A | dropout 0.3 (+0.131 valA) | 2 | 0.6786 | 0.4929 | 0.4483 | 0.4333 |
| L_adamw | A | AdamW, weight decay 1e-2 (+0.059 valA) | 0 | 0.6071 | 0.5143 | 0.4828 | 0.4417 |
| L_adamw | A | AdamW, weight decay 1e-2 (+0.059 valA) | 1 | 0.6071 | 0.4714 | 0.3103 | 0.4917 |
| L_adamw | A | AdamW, weight decay 1e-2 (+0.024 valA) | 2 | 0.5714 | 0.5071 | 0.5862 | 0.4583 |
| L_ls | A | label smoothing 0.1 (+0.095 valA) | 0 | 0.6429 | 0.5357 | 0.4483 | 0.4417 |
| L_ls | A | label smoothing 0.1 (+0.059 valA) | 1 | 0.6071 | 0.4571 | 0.3103 | 0.5500 |
| L_ls | A | label smoothing 0.1 (-0.012 valA) | 2 | 0.5357 | 0.4857 | 0.5862 | 0.5083 |
| M_aug | A | geometric augmentation (-0.119 valA) | 0 | 0.6071 | 0.6286 | 0.6897 | 0.5000 |
| M_aug | A | geometric augmentation (-0.012 valA) | 1 | 0.7143 | 0.7571 | 0.6897 | 0.6250 |
| M_aug | A | geometric augmentation (+0.059 valA) | 2 | 0.7857 | 0.7071 | 0.6207 | 0.5083 |
| L_dataB | AB | training images +B | 0 | 0.6071 | 0.4643 | in-train | 0.5000 |
| L_dataB | AB | training images +B | 1 | 0.6071 | 0.4857 | in-train | 0.5417 |
| L_dataB | AB | training images +B | 2 | 0.5714 | 0.4357 | in-train | 0.4583 |
| L_dataH | AH | training images +H | 0 | 0.6071 | 0.5643 | 0.7241 | in-train |
| L_dataH | AH | training images +H | 1 | 0.6429 | 0.6286 | 0.6897 | in-train |
| L_dataH | AH | training images +H | 2 | 0.5714 | 0.5929 | 0.6207 | in-train |
| L_dataBH | ABH | training images +B+H | 0 | 0.4286 | 0.5143 | in-train | in-train |
| L_dataBH | ABH | training images +B+H | 1 | 0.5357 | 0.5500 | in-train | in-train |
| L_dataBH | ABH | training images +B+H | 2 | 0.5357 | 0.5357 | in-train | in-train |
| L_combo1 | A | colour jitter + best epoch (+0.059 valA) | 0 | 0.6071 | 0.5429 | 0.5517 | 0.4583 |
| L_combo1 | A | colour jitter + best epoch (+0.059 valA) | 1 | 0.6071 | 0.5857 | 0.4483 | 0.4500 |
| L_combo1 | A | colour jitter + best epoch (+0.131 valA) | 2 | 0.6786 | 0.6143 | 0.5862 | 0.5167 |
| L_combo2 | A | colour jitter + label smoothing + best epoch (+0.059 valA) | 0 | 0.6071 | 0.5500 | 0.4483 | 0.4750 |
| L_combo2 | A | colour jitter + label smoothing + best epoch (+0.059 valA) | 1 | 0.6071 | 0.6143 | 0.5172 | 0.4167 |
| L_combo2 | A | colour jitter + label smoothing + best epoch (+0.131 valA) | 2 | 0.6786 | 0.5643 | 0.5862 | 0.4917 |
| M_combo | A | augmentation + cosine 30 + best epoch (+0.024 valA) | 0 | 0.7500 | 0.7286 | 0.7241 | 0.6583 |
| M_combo | A | augmentation + cosine 30 + best epoch (-0.012 valA) | 1 | 0.7143 | 0.6786 | 0.6552 | 0.6250 |
| M_combo | A | augmentation + cosine 30 + best epoch (+0.059 valA) | 2 | 0.7857 | 0.7929 | 0.7931 | 0.5917 |
| M_frozen | A | frozen backbone (-0.155 valA) | 0 | 0.5714 | 0.6071 | 0.6897 | 0.4833 |
| M_frozen | A | frozen backbone (-0.191 valA) | 1 | 0.5357 | 0.6000 | 0.6207 | 0.4250 |
| M_frozen | A | frozen backbone (-0.155 valA) | 2 | 0.5714 | 0.5357 | 0.5862 | 0.4417 |
| M_cos | A | cosine schedule, 30 epochs (-0.048 valA) | 0 | 0.6786 | 0.6071 | 0.5862 | 0.5417 |
| M_cos | A | cosine schedule, 30 epochs (+0.024 valA) | 1 | 0.7500 | 0.6786 | 0.6207 | 0.5417 |
| M_cos | A | cosine schedule, 30 epochs (-0.083 valA) | 2 | 0.6429 | 0.5929 | 0.5517 | 0.5083 |
| M_best | A | best epoch weights (-0.119 valA) | 0 | 0.6071 | 0.5857 | 0.5862 | 0.4667 |
| M_best | A | best epoch weights (-0.048 valA) | 1 | 0.6786 | 0.6214 | 0.5862 | 0.5250 |
| M_best | A | best epoch weights (-0.119 valA) | 2 | 0.6071 | 0.6357 | 0.4828 | 0.5917 |
| M_dataB | AB | training images +B | 0 | 0.5714 | 0.6643 | in-train | 0.5917 |
| M_dataB | AB | training images +B | 1 | 0.5714 | 0.5714 | in-train | 0.6167 |
| M_dataB | AB | training images +B | 2 | 0.5000 | 0.5071 | in-train | 0.6167 |
| M_dataH | AH | training images +H | 0 | 0.8571 | 0.7071 | 0.6897 | in-train |
| M_dataH | AH | training images +H | 1 | 0.7143 | 0.6214 | 0.6897 | in-train |
| M_dataH | AH | training images +H | 2 | 0.8571 | 0.6786 | 0.6552 | in-train |
| M_dataBH | ABH | training images +B+H | 0 | 0.6071 | 0.6714 | in-train | in-train |
| M_dataBH | ABH | training images +B+H | 1 | 0.6429 | 0.6571 | in-train | in-train |
| M_dataBH | ABH | training images +B+H | 2 | 0.6071 | 0.6786 | in-train | in-train |
| M_aug_cos | A | augmentation + cosine 30 (+0.024 valA) | 0 | 0.7500 | 0.7286 | 0.7241 | 0.6583 |
| M_aug_cos | A | augmentation + cosine 30 (+0.024 valA) | 1 | 0.7500 | 0.8071 | 0.7241 | 0.7000 |
| M_aug_cos | A | augmentation + cosine 30 (+0.059 valA) | 2 | 0.7857 | 0.7643 | 0.7241 | 0.7000 |
