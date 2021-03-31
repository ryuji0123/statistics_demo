import numpy as np
import plotly.graph_objects as go
from sympy import Symbol, pi, E, sqrt, integrate, oo

mu = 0
sigma = 1

# gaussian probability density function (gaussian pdf)


def f(t):
    return np.exp(-(t-mu) ** 2 / 2 * sigma ** 2) / np.sqrt(2 * np.pi * (sigma ** 2))


t = np.linspace(-5, 5, 100)
y = []
for i in range(len(t)):
    y.append(float(f(t[i])))
y = np.array(y)
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=y, name="gaussian pdf"))
fig.update_xaxes(title="x")
fig.update_yaxes(title="y")
fig.update_layout(title="gaussian pdf")
fig.show()

x = Symbol('x')
t = Symbol('t')

# gaussian probability density function (gaussian pdf)
y = E**(-(t-mu) ** 2 / 2 * sigma ** 2) / sqrt(2 * pi * (sigma ** 2))

# integrate gaussian pdf
F = integrate(y, (t, -oo, x))
cdf = []
t = np.linspace(-5, 5, 100)
for i in range(len(t)):
    cdf.append(float(F.subs({x: t[i]})))
cdf = np.array(cdf)
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=cdf, name="gaussian cdf"))
fig.update_xaxes(title="x")
fig.update_yaxes(title="y")
fig.update_layout(title="gaussian cdf")
fig.show()
