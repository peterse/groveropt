# Dummy cost fucntion; replace with something nice later
cost_func = lambda x, y: np.cos(3*x) * np.sin(1.3 * y)
algorithm_steps = np.asarray([(1, 1), (1.2, 0.8), (1.1, 0.6), (0.98, 0.33), (1.2, 0.12), (1.15, 0.00), (1.17, 0.01)])
steps_x, steps_y = zip(*algorithm_steps)
steps_x = np.asarray(steps_x)
steps_y = np.asarray(steps_y)

npts = 100
x = np.linspace(-np.pi, np.pi, npts)
y = np.linspace(-np.pi, np.pi, npts)
z = np.asarray([cost_func(xi, yj) for xi in x for yj in y])

X, Y = np.meshgrid(x, y)
Z = z.reshape(npts, npts)

cmap = 'PRGn'
stepscolor = 'r'
fig, ax = plt.subplots(figsize=(10, 10))
ax.pcolor(X, Y, Z, cmap=cmap)
ax.plot(steps_x, steps_y, c=stepscolor, lw=2.5)
ax.scatter(steps_x[-1], steps_y[-1], marker="*", s=130, c=stepscolor)