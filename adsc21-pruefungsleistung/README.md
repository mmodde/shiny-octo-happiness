# ADSC21 Prüfungsleistung – Milwaukee Property Sales

**Kurs:** Applied Data Science II: Operationalisierung von Machine Learning (ADSC21)  
**Thema:** End-to-End Machine Learning Projekt – Vorhersage von Immobilienverkaufspreisen  
**Datensatz:** [Milwaukee Open Data Portal – Property Sales Data](https://data.milwaukee.gov/dataset/property-sales-data)

---

## Projektbeschreibung

Dieses Projekt demonstriert einen vollständigen End-to-End Machine Learning Workflow zur Vorhersage von Immobilienverkaufspreisen in Milwaukee (Wisconsin, USA). Die zentrale Komponente ist ein Regressionsmodell, das auf Basis von Merkmalen wie Wohnfläche, Baujahr, Gebäudestil und Lage den Verkaufspreis vorhersagt.

### Forschungsfrage
Welche Merkmale einer Immobilie erklären den Verkaufspreis am stärksten, und mit welcher Präzision kann dieser vorhergesagt werden?

---

## Projektstruktur

```
adsc21-pruefungsleistung/
│
├── README.md                          ← Diese Datei
├── requirements.txt                   ← Python-Abhängigkeiten
│
├── data/
│   ├── property_sales_clean.csv       ← Bereinigter Datensatz (generiert durch NB 01)
│   ├── baseline_pipeline.pkl          ← Baseline-Modell (generiert durch NB 02)
│   ├── best_model_rf_tuned.pkl        ← Bestes Modell nach Tuning (NB 03)
│   ├── production_model.pkl           ← Finales Produktionsmodell (NB 04)
│   ├── model_results.csv              ← Vergleichstabelle aller Modelle
│   ├── final_metrics.json             ← Finale Metriken des besten Modells
│   └── meta.json                      ← Feature-Metadaten
│
├── notebooks/
│   ├── 01_exploration.ipynb           ← EDA & Datenbereinigung & Visualisierungen
│   ├── 02_preprocessing_pipeline.ipynb ← sklearn ColumnTransformer Pipeline + Baseline
│   ├── 03_modellvergleich.ipynb       ← ⭐ Hauptnotebook: Modellvergleich, Hyperparameteroptimierung, KPI
│   └── 04_model_serving.ipynb        ← MLflow Tracking & Model Serving
│
└── report/
    ├── abb1_preisverteilung.png       ← Generiert durch NB 01
    ├── abb2_korrelation.png
    ├── abb3_flaeche_preis.png
    ├── abb4_preis_stil.png
    ├── abb5_zeitreihe.png
    ├── abb6_predicted_vs_actual_baseline.png  ← Generiert durch NB 02
    ├── abb7_modellvergleich_rmse.png  ← Generiert durch NB 03
    ├── abb8_predicted_vs_actual_rf.png
    ├── abb9_feature_importance.png
    └── abb10_kpi_residuen.png
```

---

## Reproduktion

### Voraussetzungen
- Python 3.10+
- - pip
 
  - ### Installation
  - ```bash
    git clone https://github.com/mmodde/shiny-octo-happiness.git
    cd shiny-octo-happiness/adsc21-pruefungsleistung
    pip install -r requirements.txt
    ```

    ### Ausführung
    Notebooks in dieser Reihenfolge ausführen:

    ```bash
    jupyter notebook notebooks/01_exploration.ipynb        # Daten laden & bereinigen
    jupyter notebook notebooks/02_preprocessing_pipeline.ipynb  # Pipeline + Baseline
    jupyter notebook notebooks/03_modellvergleich.ipynb    # Hauptanalyse ⭐
    jupyter notebook notebooks/04_model_serving.ipynb      # MLflow Tracking
    ```

    **Hinweis:** Notebook 01 lädt den Datensatz direkt von Milwaukee Open Data. Eine Internetverbindung ist erforderlich. Der bereinigte Datensatz wird als `data/property_sales_clean.csv` gespeichert und von den nachfolgenden Notebooks genutzt.

    ### Cloud-Ausführung (Azure ML Studio)
    Alternativ können alle Notebooks in Azure ML Studio ausgeführt werden (vgl. Kurs Sektion 3):
    1. Azure ML Workspace erstellen (Notebook 3A im Kurs)
    2. 2. Compute Instance starten
       3. 3. Notebooks hochladen und ausführen
         
          4. ---
         
          5. ## Analysemethodik
         
          6. ### Verwendete Modelle
          7. | Modell | Beschreibung |
          8. |--------|-------------|
          9. | Lineare Regression | Baseline-Modell |
          10. | Decision Tree | Interpretierbar, nicht-lineares Modell |
          11. | Random Forest | Ensemble aus 100–500 Trees, bestes Modell |
          12. | Gradient Boosting | Boosting-Ensemble, starke Performance |
         
          13. ### Feature Engineering
          14. - **Numerisch:** Wohnfläche, Baujahr, Schlafzimmer, Bäder, Grundstücksgröße, Stockwerke
              - - **Kategoriell:** Gebäudestil, Außenwandmaterial, Postleitzahl (One-Hot-Encoding)
                - - **Pipeline:** `SimpleImputer` → `RobustScaler` / `OneHotEncoder` via `ColumnTransformer`
                 
                  - ### Hyperparameteroptimierung
                  - `RandomizedSearchCV` mit 30 Iterationen und 3-Fold Cross-Validation (effizienter als GridSearchCV bei großem Parameterraum, vgl. Géron 2022).
                 
                  - ### KPI
                  - - **RMSE** (Root Mean Squared Error) als Hauptmetrik
                    - - **MdAPE** (Median Absolute Percentage Error) als robuste KPI
                      - - **Trefferquote ±15%**: Anteil Vorhersagen innerhalb von 15% des tatsächlichen Preises
                       
                        - ---

                        ## Produktivsetzung (MLflow)

                        ```bash
                        # Modell als REST-API starten
                        mlflow models serve --model-uri 'models:/property_price_random_forest/Production' --port 5001

                        # Vorhersage anfragen
                        curl -X POST http://localhost:5001/invocations \
                          -H 'Content-Type: application/json' \
                          -d '{"dataframe_records": [{"fin_sqft": 1200, "year_built": 1985, "bedrooms": 3}]}'
                        ```

                        ---

                        ## Literatur

                        - Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3. Aufl.). O'Reilly.
                        - - Zheng, A. & Casari, A. (2018). *Feature Engineering for Machine Learning*. O'Reilly.
                          - - City of Milwaukee (2024). *Property Sales Data*. https://data.milwaukee.gov/dataset/property-sales-data
                            - - Provost, F. & Fawcett, T. (2013). *Data Science for Business*. O'Reilly.
                              - - scikit-learn Dokumentation. https://scikit-learn.org/stable/
                               
                                - ---

                                ## Einsatz von (gen)KI

                                Bei der Erstellung dieses Projekts wurde generative KI (Claude, Anthropic) zur Unterstützung bei der Code-Strukturierung und als Hilfestellung genutzt. Alle Analysen, Interpretationen und Schlussfolgerungen wurden eigenständig überprüft und verantwortet. (Gemäß DBU-Leitfaden zum Einsatz von (gen)KI.)
