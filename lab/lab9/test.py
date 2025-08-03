import matplotlib.pyplot as plt
import numpy as np

N = 31
R = 2000  # Ohms
i2 = 1e-3  # A (1 mA)
U_x = 0.3 * (1 + N % 5)  # Volts
t_R = 1 * (1 + N % 5)    # ms
V = 6  # V — assumed HIGH level of the digital output signal

t_R_sec = t_R * 1e-3  # Convert ms to seconds for formulas
t_d = (U_x * t_R_sec) / (R * i2) * 1e3  # ms
T = t_R + t_d  # Total period in ms

num_cycles = 5
dt = 0.01  # ms
t = np.arange(0, num_cycles * T, dt)
u_x = np.zeros_like(t)

for i in range(num_cycles):
    t_start = i * T
    t_high_end = t_start + t_R
    t_end = t_start + T
    u_x[(t >= t_start) & (t < t_high_end)] = 1  # HIGH
    u_x[(t >= t_high_end) & (t < t_end)] = 0    # LOW

plt.figure(figsize=(10, 3))
plt.plot(t, u_x * V, drawstyle='steps-post', label='Uₓ(t)')

plt.title(f"Dual-Slope A/D Output Signal Uₓ(t) for N = {N}")
plt.xlabel("Time [ms]")
plt.ylabel("Uₓ(t)")

plt.yticks([0, V], ["0", "V"])
plt.ylim(-1, V + 1)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


print(f"N = {N}")
print(f"Uₓ = {U_x:.2f} V")
print(f"t_R = {t_R} ms")
print(f"t_d = {t_d:.2f} ms")
print(f"Period T = {T:.2f} ms")
print(f"Frequency fₓ = {1000 / T:.2f} Hz")
print(f"Duty cycle = {t_R / T * 100:.1f}%")
