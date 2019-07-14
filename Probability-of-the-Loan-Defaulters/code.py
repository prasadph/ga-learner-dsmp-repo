# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)
p_a = (df.fico > 700).value_counts(normalize=True)[True]
p_b = (df.purpose == "debt_consolidation").value_counts(normalize=True)[True]
df1 = df[df.purpose == "debt_consolidation"]
p_a_b = len(df1[df1.fico > 700])/len(df)/p_b
print("p_a", p_a)
print("p_a_b", p_a_b)
result = p_a_b == p_a
print("result", result)


# code ends here


# --------------
# code starts here

prob_lp = (df["paid.back.loan"] == 'Yes').value_counts(normalize=True)[True]
prob_cs = (df["credit.policy"] == 'Yes').value_counts(normalize=True)[True]


new_df = df[df["paid.back.loan"] == 'Yes']
prob_pd_cs = len(new_df[new_df['credit.policy'] == 'Yes'])/len(df)/prob_lp
bayes = prob_pd_cs * prob_lp /prob_cs
print(prob_lp)
print(prob_cs)
print(prob_pd_cs)
print(bayes)
# code ends here


# --------------
# code starts here

df1 = df[df["paid.back.loan"] == "No"]
df1.purpose.value_counts().plot(kind="bar")


# code ends here


# --------------
# code starts here
inst_median = df.installment.median()
inst_mean = df.installment.mean()

fig, (ax_1, ax_2) = plt.subplots(nrows = 2 , ncols = 1, figsize=(20,10))

ax_1.hist(df.installment)
ax_1.set_xlabel('Installment')
ax_2.hist(df['log.annual.inc'])
ax_2.set_xlabel('log.annual.inc')
# code ends here


