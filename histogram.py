import plotly.graph_objects as go
from llmeter.plotting import histogram_by_dimension
from llmeter.results import Result

result = Result.load("./results/20260506-2319")

fig = go.Figure()
trace = histogram_by_dimension(result=result, dimension="time_to_last_token", xbins={"size":0.02})
fig.add_trace(trace)
fig.show()