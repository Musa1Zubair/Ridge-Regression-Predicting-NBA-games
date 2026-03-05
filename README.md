# NBA Game Outcome Predictor

A machine learning project that predicts NBA game outcomes using historical team performance data from the last two seasons.  
The project combines **web scraping, data engineering, feature engineering, and predictive modeling** to build a practical sports analytics pipeline.

---

# Project Overview

This project collects NBA game statistics from the web, processes the raw data into structured datasets, and trains a machine learning model to predict game outcomes.

The pipeline automates the process of:

- Collecting raw game data
- Cleaning and structuring statistics
- Engineering predictive features
- Training a machine learning model

The goal was to explore how historical performance metrics and contextual features can be used to forecast NBA game results.

---

# Project Screenshot

<!-- Replace the file path below with your screenshot -->

![Project Screenshot](images/project_screenshot.png)

*Example: Model predictions, data pipeline visualization, or output results.*

---

# Features

- Automated scraping of NBA game statistics using **Playwright**
- Parsing over **2,000 HTML game pages** using **Beautiful Soup**
- Data cleaning and transformation using **Pandas**
- Feature engineering including **rolling averages and opponent context**
- Machine learning model built using **Ridge Regression**
- Game outcome prediction using historical statistics

---

# Tech Stack

- **Python**
- **Playwright** – Web scraping
- **Beautiful Soup** – HTML parsing
- **Pandas** – Data processing
- **Scikit-learn** – Machine learning
- **NumPy**
- **CSV data pipelines**

---

# Data Pipeline

The project follows a full data engineering workflow:

### 1. Data Collection
Used **Playwright** to scrape NBA game pages from the past two seasons.

### 2. Data Extraction
Parsed over **2,000 HTML files** using **Beautiful Soup** to extract:
- Game links
- Box scores
- Team statistics

### 3. Data Processing
Cleaned and structured raw data using **Pandas**, converting box scores into structured DataFrames.

### 4. Data Aggregation
Merged all processed box score data into a single dataset for modeling.

### 5. Feature Engineering
Added features such as:
- Rolling averages for recent team performance
- Opponent context for matchup analysis
- Removal of low-impact features through feature selection

---

# Model

A **Ridge Regression model** was used to predict NBA game outcomes based on historical team statistics.

### Model Improvements

- Feature selection to remove irrelevant columns
- Rolling statistical averages
- Opponent-based contextual features

---

# Model Performance

Initial model accuracy: **52.6%**

After feature engineering improvements: **64.8%**

The improvements came from incorporating **time-aware statistics** and **contextual matchup features**.

---

# Live Game Predictions

The system includes logic that allows predictions for upcoming games by:

1. Using the most recent team statistics
2. Simulating matchups between teams
3. Feeding the data into the trained model

---

# Key Learnings

Through this project I gained experience in:

- Building an **end-to-end data pipeline**
- Handling large volumes of **scraped HTML data**
- Feature engineering for **sports analytics**
- Improving machine learning models using **context-aware features**
