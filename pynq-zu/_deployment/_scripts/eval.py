import numpy as np
import matplotlib.pyplot as plt

output = np.load("output_0.npy")
# print(output)

I_0 = []
# descale = 2**16/2
        
for i in output:
    I_0.append((i[0])**2 + (i[1])**2)
    
# print(I_0)

I_0 = I_0/max(I_0)

t = np.linspace(-127, 127, 255)
t = (t/127)*10

plt.figure()
plt.plot(t, I_0)
plt.xlabel("t")
plt.ylabel("Intensity")
plt.tight_layout()
plt.savefig("results.png")
print("saved figure")