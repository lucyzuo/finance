# Lucy Zuo
# Programming Assignment #1

import math

import sys
sys.version
sys.version_info

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# we are reading in the csv file
# MSCI EUROPE, MSCI USA, MSCI PACIFIC, Treasury.Bon.10Y

inpath = "/Users/lucyzuo/Desktop/FinanceCode"
outpath = "/Users/lucyzuo/Desktop/FinanceCode"

infile = "/ReturnsPortfolios.csv"

print("We are reading from the following file:")
print(inpath + infile)
indata = pd.read_csv( inpath + infile) #reading csv file

indata.head(5) # takes in the first n rows

print("")

RiskyAsset = ["MSCI EUROPE","MSCI USA","MSCI PACIFIC","Treasury.Bond.10Y"]
RiskFreeAsset = "Treasury.Bill.90D"

indata.loc[1:3, RiskyAsset] #.loc is selection based on label

#QUESTION 1
#find mean
print("THIS IS THE MEAN")
print(indata[RiskyAsset].mean())
print("")

#find median
print("THIS IS THE MEDIAN")
print(indata[RiskyAsset].median())
print("")

#find standard deviation (risk)
print("THIS IS STANDARD DEVIATION")
print(indata[RiskyAsset].std())
print("")

#find skew
print("SKEW")
print(indata[RiskyAsset].skew())
print("")

#find kurtosis
print("KURTOSIS")
print(indata[RiskyAsset].kurtosis())
print("")

#find risk return ratio
print("RISK RETURN RATIO")
print((indata[RiskyAsset].mean())/(indata[RiskyAsset].std()))


# PLOT ASSET CLASSES
fig, ax = plt.subplots()
dp = pd.DataFrame({'Month Standard Deviation': indata[RiskyAsset].std(),
     'Monthly Returns': indata[RiskyAsset].mean()})

ax = dp.plot.scatter(y='Monthly Returns', x='Month Standard Deviation', ax=ax)
for k, v in dp.iterrows():
    ax.annotate(k, v)

plt.title("Asset Class - Return vs Risk")
plt.ylabel("Monthly Return")
plt.xlabel("Monthly Standard Deviation")
plt.show()

# Write paragraph comparing the statistics of risky assets

#QUESTION 2

#plot distribution for each asset class
# MSCI Europe
indata["MSCI EUROPE"].plot.hist(stacked=True, bins=20)
plt.title("MSCI EUROPE DISTRIBUTION")
plt.xlabel("Monthly Returns")
plt.ylabel("Frequency")
plt.show()

# MSCI USA
indata["MSCI USA"].plot.hist(stacked=True, bins=20)
plt.title("MSCI USA DISTRIBUTION")
plt.xlabel("Monthly Returns")
plt.ylabel("Frequency")
plt.show()

# MSCI Pacific
indata["MSCI PACIFIC"].plot.hist(stacked=True, bins=20)
plt.title("MSCI PACIFIC DISTRIBUTION")
plt.xlabel("Monthly Returns")
plt.ylabel("Frequency")
plt.show()

# Treasury.Bond.10Y
indata["Treasury.Bond.10Y"].plot.hist(stacked=True, bins=20)
plt.title("Treasury.Bond.10Y")
plt.xlabel("Monthly Returns")
plt.ylabel("Frequency")
plt.show()
print("")

# QUESTION 3
# Covariance Matrix
data = indata[RiskyAsset]
covar = np.cov(data, rowvar=False)
print("Covariance Matrix: ")
print(covar)
print("")

#determinant of covariance
print("Calculate determinant: ")
print(np.linalg.det(covar))
print("Determinant != 0, so matrix is non-singular")
print("")

# check symmetry
print("Is covariance matrix symmetric?")
print(np.allclose(covar, covar))
print("")

# check positive definite
print("Is covariance matrix positive definite?")
print(np.all(np.linalg.eigvals(covar) > 0))
print("")

# CORRELATION MATRIX
coeff = np.corrcoef(data, rowvar=False)
print("Correlation matrix")
print(coeff)
print("")

# QUESTION 4
# Minimum variance portfolio
u = np.array([1, 1, 1, 1])
uT = u.transpose()
covar_inverse = np.linalg.inv(covar)

#MVP Weights
print("Weights of Minimum Variance Portfolio: ")
weightsMVP = (u@covar_inverse) / (u@covar_inverse@uT)
print(weightsMVP)

#MVP Returns
returns = indata[RiskyAsset].mean()
m = np.array(returns)
mT = m.transpose()

print("Expected returns on minimum variance portfolio")
mew = weightsMVP@m.transpose()
print(mew)

#MVP Variance and Risk (Std Dev)
print("Variance on minimum variance portfolio")
sigma_squared = weightsMVP @ covar @ weightsMVP.transpose()
print(sigma_squared)
print("Risk")
risk = math.sqrt(sigma_squared)
print(risk)

# get the M matrix, which is a matrix of coefficients
M_11 = m@covar_inverse@mT
M_12 = u@covar_inverse@mT
M_21 = m@covar_inverse@uT
M_22 = u@covar_inverse@uT

M = np.matrix([[M_11, M_12],
               [M_21, M_22]])
M_inverse = np.linalg.inv(M)

# Inverse M
print("Inverse Matrix")
print(M_inverse)
print("")

print(m@covar_inverse)
print(u@covar_inverse)

# Constant vectors a and b
a = (M_inverse.item(0, 0) * (m @ covar_inverse)) + (M_inverse.item(1, 0) * (u @ covar_inverse))
print ("a = ", a)
b = (M_inverse.item(0, 1) * (m @ covar_inverse)) + (M_inverse.item(1, 1) * (u @ covar_inverse))
print("b = ", b)

print("")

# expected return of 9%
nine = (0.09/12 * a) + b
print("portfolio with 9% expected return and minimum variance")
print("weights: ", nine)
print("expected return: 0.09/12")
print("risk: ", math.sqrt(nine @ covar @ nine.transpose()))

# plot asset classes
mus = [0.0061439749807556026, 0.0064, 0.0067, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.016]
sigmas = []

for val in mus:
    allweights = (val * a) + b
    print("Return: ", val)
    print("WEIGHT: ", allweights)
    newrisk = np.sqrt(allweights @ covar @ allweights[np.newaxis].transpose())[0]
    sigmas.append(newrisk)
    print("RISK: ", newrisk)
    print("")

fig, ax = plt.subplots()
dp = pd.DataFrame({'Month Standard Deviation': indata[RiskyAsset].std(),
     'Monthly Returns': indata[RiskyAsset].mean()})

ax = dp.plot.scatter(y='Monthly Returns', x='Month Standard Deviation', ax=ax)
for k, v in dp.iterrows():
    ax.annotate(k, v)

plt.plot(sigmas, mus)

plt.title("Efficient Frontier and Assets")
plt.ylabel("Monthly Return")
plt.xlabel("Monthly Standard Deviation (Risk)")


# EQUAL WEIGHTED PORTFOLIO #
equalweights = u/4 # your equal weights

e_mew = equalweights@m.transpose()
equal_sigma_squared = equalweights @ covar @ equalweights.transpose()
e_risk = math.sqrt(equal_sigma_squared)

print("EQUAL WEIGHTED PORTFOLIO")
print("RETURNS: ", e_mew)
print("RISK: ", e_risk)
plt.plot(e_risk, e_mew, marker='o', markersize=5, color="red")
ax.annotate('Equal Weight',(e_risk,e_mew))

plt.show()

print("")
# QUESTION 5
#Beta of MSCI AC WORLD
print("Market Portfolio 1: MSCI AC WORLD")
beta = {'Risky': indata['MSCI PACIFIC'], 'Other': indata['MSCI AC WORLD']}
ok = pd.DataFrame(beta)
AC_beta1 = ok.cov().as_matrix()[0][1]/indata['MSCI AC WORLD'].var()
print("MSCI PACIFIC:", AC_beta1)

beta = {'Risky': indata['MSCI EUROPE'], 'Other': indata['MSCI AC WORLD']}
ok = pd.DataFrame(beta)
EU_beta1 = ok.cov().as_matrix()[0][1]/indata['MSCI AC WORLD'].var()
print("MSCI EUROPE:", EU_beta1)

beta = {'Risky': indata['MSCI USA'], 'Other': indata['MSCI AC WORLD']}
ok = pd.DataFrame(beta)
USA_beta1 = ok.cov().as_matrix()[0][1]/indata['MSCI AC WORLD'].var()
print("MSCI USA:", USA_beta1)

beta = {'Risky': indata['Treasury.Bond.10Y'], 'Other': indata['MSCI AC WORLD']}
ok = pd.DataFrame(beta)
TB_beta1 = ok.cov().as_matrix()[0][1]/indata['MSCI AC WORLD'].var()
print("Treasury.Bond.10Y:", TB_beta1)

print("")

print("Market Portfolio 2: MSCI USA")
beta = {'Risky': indata['MSCI PACIFIC'], 'Other': indata['MSCI USA']}
ok = pd.DataFrame(beta)
AC_beta2 = ok.cov().as_matrix()[0][1]/indata['MSCI USA'].var()
print("MSCI PACIFIC:", AC_beta2)

beta = {'Risky': indata['MSCI EUROPE'], 'Other': indata['MSCI USA']}
ok = pd.DataFrame(beta)
EU_beta2 = ok.cov().as_matrix()[0][1]/indata['MSCI USA'].var()
print("MSCI EUROPE:", EU_beta2)

beta = {'Risky': indata['MSCI USA'], 'Other': indata['MSCI USA']}
ok = pd.DataFrame(beta)
USA_beta2 = ok.cov().as_matrix()[0][1]/indata['MSCI USA'].var()
print("MSCI USA:", USA_beta2)

beta = {'Risky': indata['Treasury.Bond.10Y'], 'Other': indata['MSCI USA']}
ok = pd.DataFrame(beta)
TB_beta2 = ok.cov().as_matrix()[0][1]/indata['MSCI USA'].var()
print("Treasury.Bond.10Y:", TB_beta2)


# EXTRA CREDIT
# Lambda 0.2
print("")
print("Lambda = 0.2")
lambda1 = (indata[RiskyAsset].ewm(alpha=.2).mean().cov()).as_matrix() #covariance matrix with lambda 0.2
print("Covariance Matrix")
print(lambda1)
# use this covariance to find MVP weights
lambda_i1 = np.linalg.inv(lambda1)
lambdaweights1 = (u@lambda_i1) / (u@lambda_i1@uT)
print("Weights: ", lambdaweights1)

# Lambda 0.4
print("")
print("Lambda = 0.4")
lambda2 = (indata[RiskyAsset].ewm(alpha=.4).mean().cov()).as_matrix() #covariance matrix with lambda 0.4
print("Covariance Matrix")
print(lambda2)
lambda_i2 = np.linalg.inv(lambda2)
lambdaweights2 = (u@lambda_i2) / (u@lambda_i2@uT)
print("Weights: ", lambdaweights2)

# Lambda 0.6
print("")
print("Lambda = 0.6")
lambda3 = (indata[RiskyAsset].ewm(alpha=.6).mean().cov()).as_matrix() #covariance matrix with lambda 0.6
print("Covariance Matrix")
print(lambda3)
lambda_i3 = np.linalg.inv(lambda3)
lambdaweights3 = (u@lambda_i3) / (u@lambda_i3@uT)
print("Weights: ", lambdaweights3)

# Lambda 0.8
print("")
print("Lambda = 0.8")
lambda4 = (indata[RiskyAsset].ewm(alpha=.8).mean().cov()).as_matrix() #covariance matrix with lambda 0.8
print("Covariance Matrix")
print(lambda4)
lambda_i4 = np.linalg.inv(lambda4)
lambdaweights4 = (u@lambda_i4) / (u@lambda_i4@uT)
print("Weights: ", lambdaweights4)

