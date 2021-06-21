import math
# dict = []

for i in range(1000):
    for j in range(math.floor(0.9*i),i):
        if round(j/i, ndigits=3)==0.945:
            print(j,i, round(j/i, ndigits=3))
            break
