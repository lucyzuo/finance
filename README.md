# finance
Codes for Financial Mathematics

Monte Carol simulation is an important algorithm in finance. It is very useful for option pricing and risk managment. We can apply Monmte Carlo to price options using the model of stock prices from Black Scholes Merton (1973). The stock price in BSM is a stochastic differential equation with geometric Brownian motion under the risk neutral probability. A discretization scheme for the SDE is given by  St=St−Δtexp((r−12σ2)Δt+σΔt⎯⎯⎯⎯√Wt)St=St−Δtexp((r−12σ2)Δt+σΔtWt) .

To implement a Monte Carlo valuation for an option the following algorithm can be applied.

Divide the time interval [0,T][0,T] in to equal subintervals of length ΔtΔt.
Start iterating i=1,2,…,Ii=1,2,…,I.

a. For every time step t∈Δt,2Δt,…,Tt∈Δt,2Δt,…,T draw a random number from the normal distribution.

b. Determine the time T value of the index level ST(i)ST(i) for each time step in the discretization scheme: St=St−Δtexp((r−12σ2)Δt+σΔt⎯⎯⎯⎯√Wt)St=St−Δtexp((r−12σ2)Δt+σΔtWt).

c. At T, determine the value of the option vT(ST(i))vT(ST(i)) according to the payoff.

d. Iterate until i = I.

Average all the values of vT(ST(i))vT(ST(i)) and discount them back

montecarlo.py shows the code for Monte Carlo, and adds code for the discretization scheme
St=St−Δtexp((r−12σ2)Δt+σΔt⎯⎯⎯⎯√Wt)St=St−Δtexp((r−12σ2)Δt+σΔtWt)

option_pricing.py prices 3 different call/put options with varying maturity.
