import numpy as np
import volengine as ve
import yfinance as yf
from matplotlib import pyplot as plt

aapl = yf.Ticker("AAPL")
hist = aapl.history(period="1y")
close_log = np.log(hist['Close'].values)
close_log_diff = np.diff(close_log)
date = hist.index[1:]

specs: list[ve.Spec] = []
for model in ("sgarch", "egarch", "gjrgarch", "aparch", "igarch", "csgarch"):
    for dist in ("norm", "std", "ged", "sstd", "sged", "nig", "ghyp", "jsu"):
        for p in (1, 2):
            for q in (1, 2):
                for arfima in (True, False):
                    for ar in (0, 1):
                        for ma in (0, 1):
                            specs.append(ve.Spec(model=model, dist=dist, p=p, q=q, arfima=arfima, arma=(ar, ma)))

ctx = ve.Context()
fits = ctx.fit(specs, close_log_diff)

best_fit: ve.Fit = max(fits, key=lambda f: f.loglik if f.status == "converged" else -np.inf)

print(best_fit.coef("robust"))

plt.plot(date, close_log_diff, label="Close Log Diff")
plt.plot(date, best_fit.fitted, label="Fitted")
plt.legend()
plt.show()