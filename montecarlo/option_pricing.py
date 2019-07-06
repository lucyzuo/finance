# Call with Strike = 105, Maturity T = 1
seed(2000)
S=[]
K = 105
sum = 0
for i in range(I):
    path = []
    for t in range(N+1):
        if t == 0:
            path.append(S0)
        else:
            wt = gauss(0.0,1.0)
            K = 105
            # add code for St 
            St =path[t-1] * exp((r-0.5*sigma**2) * dt + sigma *sqrt(dt) *wt)
            
            path.append(St)
    S.append(path)
    sum = sum + exp(-r)*max(St-K, 0)
    S.append(path)
value = sum/I
print(value)

# Put with Strike = 105, Maturity T = 1
seed(2000)
S=[]
K = 105
sum = 0
for i in range(I):
    path = []
    for t in range(N+1):
        if t == 0:
            path.append(S0)
        else:
            wt = gauss(0.0,1.0)
            K = 105
            # add code for St 
            St =path[t-1] * exp((r-0.5*sigma**2) * dt + sigma *sqrt(dt) *wt)
            
            path.append(St)
    S.append(path)
    sum = sum + exp(-r)*max(K-St, 0)
    S.append(path)
value = sum/I
print(value)

# Call with Strike $(S-K)^2$ , Maturity T = 1
seed(2000)
S=[]
K = 105
sum = 0
for i in range(I):
    path = []
    for t in range(N+1):
        if t == 0:
            path.append(S0)
        else:
            wt = gauss(0.0,1.0)
            K = 105
            # add code for St 
            St =path[t-1] * exp((r-0.5*sigma**2) * dt + sigma *sqrt(dt) *wt)
            
            path.append(St)
    S.append(path)
    s_squared = (St-K) * (St-K)
    sum = sum + exp(-r)*max(s_squared, 0)
    S.append(path)
value = sum/I
print(value)