# ADSC21 – Analyseprojekt: Vorhersage von Immobilienverkaufspreisen (Milwaukee)

## Zielsetzung
Vorhersage des Verkaufspreises von Immobilien in Milwaukee auf Basis öffentlicher Verkaufsdaten.

## Datensatz
- Quelle: Milwaukee Open Data Portal (https://data.milwaukee.gov/dataset/property-sales-data)
- Datei: property_sales_2024.csv (ca. 10.000–50.000 Einträge je nach Jahrgang)
- Zielvariable: `sale_price`

## Reproduktion
1. Repository klonen: `git clone <url>`
2. Abhängigkeiten installieren: `pip install -r requirements.txt`
3. Notebooks in der Reihenfolge 01 → 04 ausführen
4. Für Cloud-Ausführung: Azure ML Studio Workspace einrichten (vgl. Sektion 3 des Kurses)

## Wichtige Abhängigkeiten
- Python 3.10+
- scikit-learn >= 1.3
- pandas, numpy, matplotlib, seaborn
- mlflow
- pytorch (optional, für Sektion 6 Exkurs)