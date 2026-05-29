import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

layers = []
current_layer = None

path = "dyn_out.cli"

for raw in Path(path).read_text().splitlines():
	line = raw.strip()

	if line.startswith("$$LAYER/"):
		z = float(line.split("/")[1])

		current_layer = {
			"z": z,
			"hatches": [],
			"params": {}
		}

		layers.append(current_layer)

	elif line.startswith("$$PARAM/"):
		_, payload = line.split("/", 1)

		name, typ, value = payload.split(",")

		current_layer["params"][name] = float(value)

	elif line.startswith("$$HATCHES/"):
		_, payload = line.split("/", 1)

		vals = payload.split(",")

		hatch_type = int(vals[0])

		coords = list(map(float, vals[2:]))

		pts = list(zip(coords[::2], coords[1::2]))

		current_layer["hatches"].append({
			"type": hatch_type,
			"points": pts,
			"params": current_layer["params"].copy()
		})

# ---------------------------------
# plot first layer
# ---------------------------------

print(len(layers))
layer = layers[0]

fig, ax = plt.subplots(figsize=(8, 8))

powers = []
speeds = []
for hatch in layer["hatches"]:

	pts = hatch["points"]

	if len(pts) < 2:
		continue

	xs, ys = zip(*pts)

	power = hatch["params"].get("laser_power", 0)
	speed = hatch["params"].get("scanning_speed", 0)
	powers.append(power)
	speeds.append(speed/1000)

	color = "red" if power >= 400 else "blue"

	ax.plot(xs, ys, color=color)

ax.set_aspect("equal")
ax.set_title(f"Layer Z = {layer['z']}")

plt.savefig("cli.png")
print(min(speeds),max(speeds),np.mean(speeds))
print(min(powers),max(powers),np.mean(powers))