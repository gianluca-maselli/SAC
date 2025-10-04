import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def load_file(fname):
	return np.loadtxt(fname)

def get_data(env_name, envs_dict, files, res_type, lower, upper):
	all_data = []
	for f in files:
		data = load_file(f)
		all_data.append(data[:, 1])
	all_data = np.array(all_data)
	mean_res = np.mean(all_data, axis=0)
	# Compute mean and 95% confidence interval (percentiles)
	y_lower = np.percentile(all_data, 2.5, axis=0)
	y_upper = np.percentile(all_data, 97.5, axis=0)
	
	envs_dict[env_name][res_type] = mean_res # take only y values
	envs_dict[env_name][lower] = y_lower # take only y values
	envs_dict[env_name][upper] = y_upper # take only y values
	return envs_dict


folder = "curves/"

envs = os.listdir(folder)
print(envs)

disc_folder = 'out_disc_1_update_step/'
gauss_folder = 'out_gaus_1_update_step/'

envs_dict = {}

for env_name in envs:
	envs_dict[env_name] = {'disc_res' : [], 'disc_lower': [], 'disc_upper': [], 
							'gaus_res': [], 'gaus_lower': [], 'gaus_upper': []}
	disc_files = sorted(glob.glob(folder + env_name + "/" + disc_folder + "*.txt"))
	gaus_files = sorted(glob.glob(folder + env_name + "/" + gauss_folder + "*.txt"))
	
	envs_dict = get_data(env_name, envs_dict, disc_files, 'disc_res', 'disc_lower', 'disc_upper')
	envs_dict = get_data(env_name, envs_dict, gaus_files, 'gaus_res', 'gaus_lower', 'gaus_upper')
	
steps  = load_file(disc_files[0])[:, 0]
#print(envs_dict)

# === Plotting ===
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=False)
axes = axes.flatten()

colors = {"disc": "orange", "gaus": "blue"}
labels = {"disc": "SAC Discrete Policy", "gaus": "SAC Gaussian Policy"}

for idx, env_name in enumerate(envs):
	ax = axes[idx]
	# Plot Gaussian policy
	ax.plot(steps, envs_dict[env_name]['gaus_res'], color=colors["gaus"], label=labels["gaus"])
	ax.fill_between(steps, envs_dict[env_name]['gaus_lower'], envs_dict[env_name]['gaus_upper'], color=colors["gaus"], alpha=0.3)
	# Plot Discrete policy
	ax.plot(steps, envs_dict[env_name]['disc_res'], color=colors["disc"], label=labels["disc"])
	ax.fill_between(steps, envs_dict[env_name]['disc_lower'], envs_dict[env_name]['disc_upper'],
                    color=colors["disc"], alpha=0.3)
	ax.set_title(env_name, fontsize=12)
	ax.set_xlabel("Environment Steps")
	ax.set_ylabel("Episode Return")
	ax.grid(True, alpha=0.3)

# Put legend outside, only once
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
plt.show()
