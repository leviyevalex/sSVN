import imageio
import deepdish as dd
import os
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
def create_kde_plots(composite_map, output_dir, begin, end):
    save_path = output_dir + '/figures'
    iter = sorted(list(composite_map.keys()))[-1]
    kde_file_path = save_path + '/kde0.jpeg'
    gif_file_path = save_path + '/kde_flow.gif'
    with imageio.get_writer(gif_file_path, mode='I') as writer:
        for i in range(iter):
            data_i = pd.DataFrame(composite_map[i]['X'])
            data_i.rename(columns={0: 'x', 1: 'y'}, inplace=True)
            g = sns.jointplot(x='x', y='y', data=data_i, kind='kde',joint_kws={"cut": 15}, xlim=(begin, end), ylim=(begin, end), space=0, cmap='viridis')
            plt.subplots_adjust(top=0.9)
            g.ax_marg_x.set_title('stein Ensemble KDE Iteration: %i' %i)
            if i != 0:
                kde_file_path = save_path + '/kdef.jpeg'
            g.fig.savefig(kde_file_path)
            image = imageio.imread(kde_file_path)
            writer.append_data(image)
