# Movie Revenue Prediction - Ergebnisse Zusammenfassung

**Projekt:** Ablation Study zur Feature-Wichtigkeit bei Movie Revenue Prediction
**Modell:** Random Forest Regression (optimiert)
**Dataset:** TMDB Movies (7,380 Filme nach Bereinigung)
**Evaluation:** Test-Set (n=1,107, 15% der Daten)

---

## 🎯 Hauptergebnisse auf einen Blick

### Das Wichtigste in 3 Sätzen:
1. **Budget ist King**: Budget allein erklärt 25.7% der Modell-Performance
2. **3 Features reichen fast**: Budget + Vote Count + Release Year erreichen 95% der Baseline-Genauigkeit
3. **Genres sind unwichtig**: 10 Genre-Features zusammen bringen nur 0.5% Performance-Gewinn

---

## 📊 Baseline-Modell Performance

**Alle 26 Features, optimierte Hyperparameter**

| Metrik | Train | Validation | Test |
|--------|-------|------------|------|
| **R² Score** | 0.876 | 0.737 | **0.729** |
| **RMSE (log)** | 1.056 | 1.489 | **1.594** |
| **MAE (log)** | — | 0.943 | **1.027** |

**Interpretation Test-Set:**
- Modell erklärt 73% der Revenue-Varianz
- Typischer Fehler: Faktor 4.9× (z.B. ±$200M bei $100M Film)
- Moderate Überanpassung (Train-Test Gap: 14.7%)

---

## 🔝 Top-K Features: Wie viele braucht man wirklich?

| K Features | R² (Test) | % der Baseline | Feature-Reduktion |
|------------|-----------|----------------|-------------------|
| **3** | 0.692 | **95.0%** | -88% (26→3) |
| **5** | 0.717 | **98.3%** | -81% (26→5) |
| 7 | 0.716 | 98.2% | -73% |
| 10 | 0.718 | 98.5% | -62% |
| 26 | 0.729 | 100.0% | Baseline |

### Top-5 Features sind optimal:
1. **budget_log** - Log-Budget
2. **vote_count_log** - Anzahl Bewertungen (log)
3. **release_year** - Erscheinungsjahr
4. **runtime** - Laufzeit in Minuten
5. **popularity_log** - TMDB Popularity Score (log)

**💡 Praktische Empfehlung:**
- **Schnelle Schätzung:** Nutze Top-3 (95% Genauigkeit)
- **Production Decision:** Nutze Top-5 (98% Genauigkeit)
- **Maximale Genauigkeit:** Nutze Top-10 (statistisch = Baseline)

---

## 🔍 Individual Feature Ablation - Die 10 wichtigsten Features

**Was passiert wenn man ein Feature entfernt?**

| Rang | Feature | R² (Test) | R² Drop | Beitrag |
|------|---------|-----------|---------|---------|
| 1 | **budget_log** | 0.542 | **0.188** | **25.7%** 🔥 |
| 2 | vote_count_log | 0.711 | 0.018 | 2.5% |
| 3 | release_year | 0.719 | 0.011 | 1.5% |
| 4 | runtime | 0.725 | 0.005 | 0.6% |
| 5 | lang_es | 0.725 | 0.004 | 0.6% |
| 6 | cast_size | 0.727 | 0.002 | 0.3% |
| 7 | popularity_log | 0.727 | 0.002 | 0.3% |
| 8 | lang_en | 0.728 | 0.002 | 0.2% |
| 9 | production_country_count | 0.728 | 0.001 | 0.2% |
| 10 | lang_fr | 0.729 | 0.001 | 0.1% |

**Erkenntnisse:**
- Budget ist 10× wichtiger als das zweitwichtigste Feature
- Top-3 Features machen 30% der Performance aus
- Features 4-26 zusammen nur 3%

### Die 5 unwichtigsten Features:
- vote_average, director_count, production_company_count → Negative Drops (verschlechtern Modell!)
- Genres einzeln: alle < 0.001 Drop
- Sprachen (außer Spanisch): alle < 0.001 Drop

---

## 📦 Group Ablation - Welche Kategorien sind wichtig?

**Was passiert wenn man eine ganze Feature-Gruppe entfernt?**

| Feature-Gruppe | Anzahl | R² ohne Gruppe | R² Drop | Beitrag |
|----------------|--------|----------------|---------|---------|
| **Financial** | 1 | 0.542 | **0.188** | **25.7%** 💰 |
| **Popularity** | 3 | 0.652 | 0.077 | 10.6% 📈 |
| **Temporal** | 2 | 0.717 | 0.012 | 1.7% 📅 |
| **Languages** | 5 | 0.722 | 0.007 | 1.0% 🌐 |
| **Content** | 3 | 0.724 | 0.005 | 0.7% 🎬 |
| **Genres** | 10 | 0.726 | 0.003 | 0.5% 🎭 |
| **Production** | 2 | 0.729 | 0.000 | 0.0% 🏭 |

**Kernbotschaften:**
- **Financial + Popularity** (4 Features) = 36.3% der Performance
- **Alle anderen** (22 Features) = nur 3.4% der Performance
- **Genres**: 10 Features bringen nur 0.5% → Überflüssig!
- **Production**: Vollkommen nutzlos (0.0%)

---

## 📈 Statistische Validierung

### Bootstrap Confidence Intervals (95%, n=1000, Test-Set)

| Modell | R² Mean | 95% CI | Breite |
|--------|---------|--------|--------|
| Baseline (26) | 0.729 | [0.696, 0.760] | 0.063 |
| Top-3 | 0.692 | [0.658, 0.724] | 0.067 |
| Top-5 | 0.717 | [0.683, 0.748] | 0.064 |
| Top-10 | 0.718 | [0.685, 0.749] | 0.064 |

→ Alle CIs relativ schmal (~0.06) = stabile Schätzungen

### Paired t-Tests (Test-Set)

| Vergleich | R² Diff | p-value | Signifikant? | Interpretation |
|-----------|---------|---------|--------------|----------------|
| Baseline vs. Top-3 | -0.037 | < 0.001 | ✅ Ja*** | Baseline deutlich besser |
| Baseline vs. Top-5 | -0.012 | 0.035 | ✅ Ja* | Baseline leicht besser |
| Baseline vs. Top-10 | -0.011 | 0.062 | ❌ Nein | Kein signifikanter Unterschied |
| Top-3 vs. Top-5 | +0.025 | < 0.001 | ✅ Ja*** | Top-5 deutlich besser als Top-3 |
| Top-5 vs. Top-10 | +0.002 | 0.808 | ❌ Nein | Kein Unterschied |

**Fazit:**
- **Top-10 = Baseline** (statistisch äquivalent, p=0.062)
- **Top-5 optimal** (praktischer Trade-off: 98% Performance, 81% weniger Features)
- **Top-3 reicht für Schnellschätzungen** (95% Performance)

---

## 🎓 Hyperparameter-Tuning Details

**GridSearchCV Ergebnis (5-Fold CV, 24 Kombinationen):**

```
Beste Konfiguration:
  max_depth: 12
  n_estimators: 200
  min_samples_split: 15
  min_samples_leaf: 8
  max_features: 'sqrt'

Cross-Validation R²: 0.701
Train-CV Gap: 7.7% ✓ Sehr gut
```

**Regularisierungs-Strategie:**
- Flachere Bäume (max_depth ≤ 12)
- Größere Blätter (min_samples_leaf ≥ 8)
- Feature-Subsampling (nur √26 ≈ 5 Features pro Split)
- → Reduziert Overfitting von 17.2% auf 13.9%

---

## 💡 Praktische Implikationen

### Was bedeutet das für die Praxis?

#### 1. **Datensammlung vereinfachen**
- ✅ **Fokus auf:** Budget, Vote Count, Release Year (Minimum)
- ✅ **Optional:** Runtime, Popularity (für 98% Genauigkeit)
- ❌ **Nicht nötig:** Genres, Sprachen, Production Details
- **Ersparnis:** 88% weniger Features bei 95% Performance

#### 2. **Budget ist unverzichtbar**
- Budget allein = 26% der Performance
- Alle anderen 25 Features zusammen = 74%
- → Jedes Revenue-Prediction-System MUSS Budget enthalten

#### 3. **Popularity > Genre**
- Vote Count (2.5%) > alle 10 Genres zusammen (0.5%)
- → Pre-Release Engagement wichtiger als Content-Typ

#### 4. **Simplicity wins**
- 3-Feature-Modell: Einfach zu erklären, zu warten, zu verstehen
- 5-Feature-Modell: Sweet Spot zwischen Genauigkeit und Komplexität
- 26-Feature-Modell: Minimal besser, aber 5× komplexer

---

## 🔬 Wissenschaftliche Erkenntnisse

### Warum ist Budget so dominant?

Budget ist ein **Proxy für viele unmessbare Faktoren:**
- Produktionsqualität (CGI, Set Design, Technik)
- Star Power (teure Schauspieler)
- Marketing-Budget (korreliert mit Production Budget)
- Distribution (Wide Release vs. Limited Release)
- Studio-Confidence (High Budget = High Expectations)

→ Budget kodiert implizit das "Potential" eines Films

### Warum sind Genres unwichtig?

**Hypothesen:**
1. Budget dominiert Genre-Effekte
   - High-Budget Action = High Revenue
   - High-Budget Drama = High Revenue
   - → Budget matters, nicht Genre

2. Popularity erfasst Genre-Präferenzen
   - Audiences wählen Filme basierend auf Genre
   - → Vote Count spiegelt Genre-Appeal wider

3. Multi-Genre-Filme
   - Viele Filme haben 2-4 Genres
   - Binary Encoding zu simpel

### Log-Transformation ist kritisch

**Warum Log?**
- Budgets: $1M bis $300M (6 Größenordnungen)
- Revenue: $1K bis $2.8B (6 Größenordnungen)
- Log fängt diminishing returns ein:
  - $1M → $10M: Riesiger Impact
  - $100M → $200M: Kleiner Impact

**Interpretation:**
- RMSE = 1.59 (log) ≠ "$1.59 Fehler"
- RMSE = 1.59 (log) = Faktor 10^1.59 ≈ 39× Fehler
- Oder: e^1.59 ≈ 4.9× auf natürlicher Skala

---

## ⚠️ Limitationen

### 1. **Selection Bias**
- Nur 6.2% der Original-Daten verwendet (7,380 von 119,938)
- Bias zu Major Studio Releases
- Unabhängige/Ausländische Filme unterrepräsentiert

### 2. **Overfitting bleibt**
- Train-Test Gap: 14.7% (moderat-stark)
- Trotz Regularisierung nicht eliminiert
- Modell könnte auf neuen Daten schlechter sein

### 3. **Fehlende Features**
- Marketing Spend (oft = Production Budget)
- Star Power (nur indirekt via Cast Size)
- Critical Reviews (Rotten Tomatoes, Metacritic)
- Competition (andere Releases im gleichen Zeitfenster)
- Social Media Buzz (Twitter, YouTube)

### 4. **Temporale Validität**
- Daten bis 2017
- Industrie hat sich verändert: Streaming, COVID-19, Social Media
- Features die 2010 wichtig waren ≠ Features die 2025 wichtig sind

### 5. **Kausalität vs. Korrelation**
- Modell sagt Revenue **gegeben** Budget vorher
- Nicht: Kausaler Effekt von Budget auf Revenue
- Reverse Causation: Studios vergeben Budgets basierend auf erwarteter Revenue

---

## 📋 Empfehlungen nach Use Case

### Schnell-Schätzung (Greenlighting Early Stage)
**Nutze: Top-3 Model**
- Features: Budget, Vote Count, Release Year
- Genauigkeit: 95% der Baseline
- Vorteil: Minimal Data, Maximum Speed

### Production Decision (Greenlighting Final Stage)
**Nutze: Top-5 Model**
- Features: +Runtime, Popularity
- Genauigkeit: 98% der Baseline
- Vorteil: Optimal Balance

### Research / Maximale Genauigkeit
**Nutze: Top-10 oder Baseline**
- Genauigkeit: Statistisch = Baseline
- Vorteil: Höchste Genauigkeit
- Nachteil: Komplexität, Overfitting-Risiko

---

## 📊 Daten-Übersicht

**Dataset nach Cleaning:**
- 7,380 Filme total
- 5,166 Training (70%)
- 1,107 Validation (15%)
- 1,107 Test (15%)

**Features:**
- 26 Total
- 1 Financial
- 3 Popularity
- 2 Temporal
- 3 Content
- 2 Production
- 10 Genres
- 5 Languages

**Revenue Distribution:**
- Mean: $88.5M
- Median: $35.2M
- Std: $149.3M
- Range: $1K - $2.8B

---

## 🎯 Wichtigste Takeaways

1. **Budget > All**: 25.7% Performance, 10× wichtiger als nächstes Feature
2. **3 Features = 95%**: Budget + Votes + Year reichen fast
3. **5 Features = Optimal**: Sweet Spot bei 98% Performance
4. **Genres nutzlos**: 10 Features = 0.5% Beitrag
5. **Simplicity wins**: Einfache Modelle fast so gut wie komplexe
6. **Log-Scale essentiell**: Normalisiert Größenordnungen
7. **Test-Set kritisch**: Validation allein überschätzt Performance
8. **Ablation > Importance**: Kausale statt korrelative Evidence

---

**Generiert:** 2025-11-27
**Code:** Week5_Ablation_Study_26_Features.ipynb
**Paper:** Movie_Revenue_Prediction_Paper.md
