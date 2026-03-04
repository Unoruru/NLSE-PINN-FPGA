import numpy as np
import matplotlib.pyplot as plt

output = np.load("output_0.npy")

I_0 = []
descale = 10/((2**24/2)-1)
        
# for i in output:
#     I_0.append((i[0])**2 + (i[1])**2)

for i in output:
    I_0.append((i[0]*descale)**2 + (i[1]*descale)**2)
    
# I_0 = I_0/max(I_0)

t = np.linspace(-127, 127, 255)
t = (t/127)*10

plt.figure()
plt.plot(t, I_0)
plt.xlabel("t")
plt.ylabel("Intensity")
plt.tight_layout()
plt.savefig("results_test.png")
print("saved figure")