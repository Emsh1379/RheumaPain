# Data

This study is a secondary analysis of the public **RheumaPain** dataset, version 3.

- **Dataset:** Yıldırım E, Patlar Akbulut F. RheumaPain. Mendeley Data, v3. doi:[10.17632/y6pgjdj22f.3](https://doi.org/10.17632/y6pgjdj22f.3) (CC BY 4.0)
- **Data article:** *Data in Brief* 2026;68:113131. doi:[10.1016/j.dib.2026.113131](https://doi.org/10.1016/j.dib.2026.113131)
- **Content:** 42 children aged 5–18 years with rheumatic disease in a physiotherapy unit in Istanbul; Empatica E4 wristband
  (BVP 64 Hz, ACC 32 Hz, EDA and TEMP 4 Hz); rest and exercise phases; Wong–Baker FACES pain (0–5) per phase;
  demographics in `Demographic Data/patients.xlsx`.

The raw data (~277 MB) are **not redistributed** in this repository.

## Download and verify

`manifest.json` lists all 547 files of v3 with path, size, Mendeley download URL and SHA-256 checksum.

```bash
python data/download_data.py /path/to/parent      # downloads all files (~277 MB) and checks the SHA-256
```

or download the dataset from Mendeley by hand. Then point the pipeline to it:

```bash
export RP_DATA=/path/to/parent      # must contain RheumaPain/Biosignal Data/... and RheumaPain/Demographic Data/...
```

Expected layout:

```
RheumaPain/
├── Biosignal Data/
│   ├── Raw Data/s001 … s042/        ACC.csv, BVP.csv, EDA.csv, TEMP.csv, HR.csv, IBI.csv, info.txt
│   └── Processed Data/4Hz|8Hz|…|64Hz/s001.csv …
└── Demographic Data/patients.xlsx
```

## Derived data in this repository

`results/features_w10.csv`, `features_w30.csv` and `features_w60.csv` hold the 26 window features per child,
phase segment and window. They contain only de-identified subject codes from the public dataset and allow every
experiment from `s03_experiments.py` onward to be rerun without downloading the raw data.
