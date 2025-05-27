import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("nba_games.csv", index_col=0)
df = df.sort_values("date").reset_index(drop=True)
df = df.drop(columns=["mp.1", "mp_opp.1", "index_opp"])
df["target"] = df.groupby("team")["won"].shift(-1)
df = df.dropna(subset=["target"])
df["target"] = df["target"].astype(int)
nulls = df.isnull().sum()
for col in df.columns:
    if col != "target" and nulls.get(col, 0) > 0:
        df = df.drop(columns=[col])

rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)
removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = df.columns[~df.columns.isin(removed_columns)]
scaler = MinMaxScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])
sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction="forward", cv=split)
sfs.fit(df[selected_columns], df["target"])
predictors = list(selected_columns[sfs.get_support()])

def backtest(data, model, predictors, start=1, step=1):
    out = []
    seasons = sorted(data["season"].unique())
    for i in range(start, len(seasons), step):
        train = data[data["season"] < seasons[i]]
        test = data[data["season"] == seasons[i]]
        model.fit(train[predictors], train["target"])
        prediction = model.predict(test[predictors])
        temp = pd.DataFrame({"actual": test["target"], "prediction": prediction}, index=test.index)
        out.append(temp)
    return pd.concat(out)

df_rolling = df[list(selected_columns) + ["won", "team", "season"]]
def team_averages(team):
    num = team.select_dtypes(include=["number"]).columns
    roll = team[num].rolling(10).mean()
    for col in ["team", "season"]:
        roll[col] = team[col]
    return roll
df_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(team_averages)
rolling_cols = [f"{col}_10" for col in df_rolling.columns]
df_rolling.columns = rolling_cols
df = pd.concat([df, df_rolling], axis=1)
df = df.dropna()

def shift_col(team, col):
    return team[col].shift(-1)

def add_col(df, col):
    return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col))

df["home_next"] = add_col(df, "home")
df["team_opp_next"] = add_col(df, "team_opp")
df["date_next"] = add_col(df, "date")

full = df.merge(
    df[rolling_cols + ["team_opp_next", "date_next", "team"]],
    left_on=["team", "date_next"],
    right_on=["team_opp_next", "date_next"]
)

removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns
selected_columns = full.columns[~full.columns.isin(removed_columns)]
sfs.fit(full[selected_columns], full["target"])
predictors = list(selected_columns[sfs.get_support()])

new_game = full.iloc[-1:].copy()
new_game["team"] = "IND"
new_game["team_opp"] = "NYK"
new_game["home"] = 0
new_game["won"] = 0
new_game["date"] = "2025-05-27"
new_game["season"] = 2025
new_game["target"] = 0
full = pd.concat([full, new_game], ignore_index=True)

rr.fit(full[predictors][:-1], full["target"][:-1])
tonight = rr.predict(full[predictors].tail(1))[0]

print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")

print("Winner tonight:", "Pacers!!!" if tonight == 1 else "Knicks!!!")
predictions = backtest(full, rr, predictors)
accuracy = accuracy_score(predictions["actual"], predictions["prediction"])
print(f"With an Accuracy Score of: {accuracy}")
