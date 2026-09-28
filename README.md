# PPDP Project — Milestone 2 Setup

This repository snapshot contains the setup artifacts for Milestone 2 of the KC-SLICE project.

## Milestone 2 scope

- Install and verify the Mondrian implementation.
- Install a JVM and verify the ARX Java library.
- Create the initial KC-Slice reimplementation structure.
- Download and verify the Adult Census Income dataset.
- Download and verify the Diabetes 130-US Hospitals dataset.
- Verify the required `?` missing-value sentinels.
- Create small smoke-test datasets.
- Store reproducible setup and verification scripts.

## Verified environment

- Python: Colab Python 3.13 environment
- Java: OpenJDK 21.0.12.1
- ARX: 3.9.2
- Mondrian: `qiyuangong/Mondrian`

## Dataset verification

Adult:
- 32,561 rows
- 15 columns

Diabetes:
- 101,766 rows
- 50 columns in the raw CSV
- `?` sentinel verified in `race`, `weight`, `payer_code`, and `medical_specialty`

## Repository contents

`scripts/setup_milestone2.sh` contains the environment/data setup commands.

`scripts/verify_setup.py` verifies the expected local setup and dataset properties.

`notebooks/milestone2_setup.ipynb` records the Colab setup workflow.

`results/milestone2_setup_verified.txt` records the successful verification output.

Large datasets and binary dependencies are intentionally excluded from Git and are downloaded by the setup script.

## Source links

- Mondrian: https://github.com/qiyuangong/Mondrian
- ARX: https://github.com/arx-deidentifier/arx/releases/tag/v3.9.2
- Adult Census Income: https://huggingface.co/datasets/scikit-learn/adult-census-income
- Diabetes 130-US Hospitals: https://archive.ics.uci.edu/dataset/296
