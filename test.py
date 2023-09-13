import  numpy as np

x = 3
if x >= 0:  # A new section begin before leaving next state
    r_parking = -2 * (np.exp(5 * x) - 1)
else:  # still in current section
    r_parking = -2 * (np.exp(5 * x) - 1)
print(r_parking)