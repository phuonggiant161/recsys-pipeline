# Recommender Data Pipeline

This project performs:
1. k-core densification
2. sparsity metrics calculation before and after densification

## Prepare data


Due to large file size, dataset is hosted on Google Drive.

Download here:
https://drive.google.com/drive/folders/15D99fn2hSeKwfRGJWadF9hDNB4DaRqrb

After downloading, place files into:

data/raw/

## Required columns

The dataset must contain at least:
- `customer_id`
- `article_id`

## Run on Windows

Double click:

`run.bat`

## Run in terminal

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py --verbose