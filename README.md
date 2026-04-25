# Recommender Data Pipeline
1. File code
src.metrics: đo độ thưa của dataset
src.thinning: làm thưa hóa
src.kcore: làm dày hóa

2. Report
Các số liệu in ra được save tại folder reports

## Prepare data


Due to large file size, dataset is hosted on Google Drive.

Download here:
https://drive.google.com/drive/folders/15D99fn2hSeKwfRGJWadF9hDNB4DaRqrb

After downloading, place files into:

data/raw/


## Run in terminal

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py --verbose