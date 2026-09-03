"""Directional forecasting - and the machinery to judge whether it works.

The package answers one question: what is the probability that the forward
return over N days is positive. It deliberately does not forecast a price
level; at 4% daily volatility the confidence interval on a 30-day price would
be wider than the forecast itself.

Read `baselines.py` before anything else. Bitcoin rose in about 55% of
historical 30-day windows, so the reference point is 55%, not 50%, and a model
that does not beat momentum is measuring trend rather than any cycle.
"""
