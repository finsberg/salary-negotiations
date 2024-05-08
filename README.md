# Compute useful stats for Salary negotiations

We compute different useful stats based on the calculation schemas from Tekna. 

Here we assume that you have the following files available
```
├── 2020.xlsx
├── 2021.xlsx
├── 2022.xlsx
├── 2023
│   ├── SC.xlsx
│   ├── SMET.xlsx
│   ├── SRL.xlsx
│   ├── SUIB.xlsx
```
where `2020.xlsx`, `2021.xlsx` and `2022.xlsx` contains the spreadsheets from the year 2020, 2021 and 2022 respectively. The folder `2023` contains the spreadsheets for the four different companies in Simula for the year 2023. 

## Install 
Create a virtual environment
```
python3 -m venv venv
```
activate it
```
. venv/bin/activate
```
and install the dependencies
```
python3 -m pip install -r requirements.txt
```

## Run the analysis
```
python3 main.py
```