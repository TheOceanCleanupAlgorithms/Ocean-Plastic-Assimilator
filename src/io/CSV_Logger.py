import pandas as pd


class CSV_Logger:
    def __init__(self, path: str):
        self.df = pd.DataFrame()
        self.currentLog = dict()
        self.path = path

    def log(self, key: str, value: float):
        self.currentLog[key] = value
        return

    def flush(self):
        self.df = self.df.append(self.currentLog, ignore_index=True)
        self.currentLog.clear()
        return

    def export_csv(self):
        self.df.to_csv(self.path + ".csv")
