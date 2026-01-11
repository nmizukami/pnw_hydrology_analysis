from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y)) 

def custom_div_cmap(numcolors=11, name='custom_div_cmap',
                    mincol='blue', midcol='white', maxcol='red'):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    https://xkcd.com/color/rgb/
    """
    cmap = LinearSegmentedColormap.from_list(name=name,
                                             colors =[mincol, midcol, maxcol],
                                             N=numcolors)
    return cmap


# kge
#cmap = mpl.colors.ListedColormap(mpl.cm.Spectral_r(np.arange(9)))
#norm0 = mpl.colors.BoundaryNorm(vals0, cmap0.N)
cmap0 = plt.get_cmap('plasma_r', 8)
cmap0.set_under('cyan')
vals0 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
norm0 = mpl.colors.BoundaryNorm(vals0, cmap0.N, extend='min')

# %bias
vals_bias1=[-60, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 60]
cmap_bias1 = LinearSegmentedColormap.from_list('custom1', 
                                             [(0.0, 'xkcd:red'),
                                              (0.5, 'xkcd:white'),
                                              (1.0, 'xkcd:blue')], N=13)
cmap_bias1.set_over('xkcd:dark blue')
cmap_bias1.set_under('xkcd:dark red')
norm_bias1 = mpl.colors.BoundaryNorm(vals_bias1, cmap_bias1.N)

# %bias
vals_bias2=[-80, -70, -60, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 60, 70, 80]
cmap_bias2 = LinearSegmentedColormap.from_list('custom1', 
                                             [(0.0, 'xkcd:red'),
                                              (0.5, 'xkcd:white'),
                                              (1.0, 'xkcd:blue')], N=17)
cmap_bias2.set_over('xkcd:dark blue')
cmap_bias2.set_under('xkcd:dark red')
norm_bias2 = mpl.colors.BoundaryNorm(vals_bias2, cmap_bias2.N)

cmap_bias_continuous = LinearSegmentedColormap.from_list('custom1', 
                                             [(0.0, 'xkcd:red'),
                                              (0.5, 'xkcd:white'),
                                              (1.0, 'xkcd:blue')], N=255)
cmap_bias_continuous.set_over('xkcd:dark blue')
cmap_bias_continuous.set_under('xkcd:dark red')
norm_bias_continuous = mpl.colors.Normalize(vmin=-0.6, vmax=0.6)


# ratio
vals2=[0.75, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.25] 
#cmap = cm.get_cmap('RdYlBu', (7))
cmap2 = LinearSegmentedColormap.from_list('custom1', 
                                             [(0.0, 'xkcd:red'),
                                              (0.5, 'xkcd:white'),
                                              (1.0, 'xkcd:blue')], N=11)
cmap2.set_over('xkcd:dark blue')
cmap2.set_under('xkcd:dark red')
norm2 = mpl.colors.BoundaryNorm(vals2, cmap2.N)

# corr
vals3=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
cmap3 = LinearSegmentedColormap.from_list('custom1', 
                                             [(0.00, 'xkcd:yellow'),
                                              (0.50, 'xkcd:green'),
                                              (1.00, 'xkcd:blue')], N=8)
cmap3.set_over('xkcd:dark blue')
cmap3.set_under('xkcd:saffron')
norm3 = mpl.colors.BoundaryNorm(vals3, cmap3.N)

# kge/corr experiment
# https://coolors.co/palette/54478c-2c699a-048ba8-0db39e-16db93-83e377-b9e769-efea5a-f1c453-f29e4c
vals_kge=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
cmap_kge = LinearSegmentedColormap.from_list('custom1', 
                                             [(0.00, '#EFEA5A'),
                                              (1.0/len(vals_kge), '#B9E769'),
                                              (2.0/len(vals_kge), '#83E377'),
                                              (3.0/len(vals_kge), '#16DB93'),
                                              (4.0/len(vals_kge), '#0DB39E'),
                                              (5.0/len(vals_kge), '#048BA8'),
                                              (1.00, '#2C699A')], N=8)
cmap_kge.set_over('#54478C')
cmap_kge.set_under('#F1C453')
norm_kge = mpl.colors.BoundaryNorm(vals_kge, cmap_kge.N)


# KGE difference
vals4=[-0.1, -0.08, -0.06, -0.04, -0.02, 0.02, 0.04, 0.06, 0.08, 0.1]
cmap4 = LinearSegmentedColormap.from_list('custom1', 
                                             [(0.0, 'xkcd:red'),
                                              (0.5, 'xkcd:light grey'),
                                              (1.0, 'xkcd:blue')], N=11)
cmap4.set_over('xkcd:royal blue')
cmap4.set_under('xkcd:magenta')
norm4 = mpl.colors.BoundaryNorm(vals4, cmap4.N)

cmap_summa_diff = LinearSegmentedColormap.from_list('custom 1', 
                                             [(0,    'xkcd:red'),
                                              (0.50, 'xkcd:light grey'),
                                              (1,    'xkcd:blue')], N=250)

cmap_summa_swe_diff = LinearSegmentedColormap.from_list('custom 2', 
                                             [(0,    'xkcd:red'),
                                              (0.50, 'xkcd:pale salmon'),
                                              (1,    'xkcd:light grey')], N=250)

# ---------------------
# climate change signal
# ---------------------
# --------
# annual mean flow change from control period
cmap_mean_flow_diff = LinearSegmentedColormap.from_list('custom1', 
                                             [(0.0,      'xkcd:red'),
                                              (100/1600, 'xkcd:white'),
                                              (1.0,      'xkcd:blue')], N=255)
cmap_mean_flow_diff.set_over('xkcd:dark blue')
cmap_mean_flow_diff.set_under('xkcd:dark red')
norm_mean_flow_diff=mpl.colors.Normalize(vmin=-100, vmax=1500)

# percent diff
cmap_mean_flow_pdiff = LinearSegmentedColormap.from_list('custom1', 
                                             [(0.0,      'xkcd:red'),
                                              (40/100, 'xkcd:white'),
                                              (1.0,      'xkcd:blue')], N=255)
cmap_mean_flow_pdiff.set_over('xkcd:dark blue')
cmap_mean_flow_pdiff.set_under('xkcd:dark red')
norm_mean_flow_pdiff=mpl.colors.Normalize(vmin=-40, vmax=60)

# annual mean flow
norm_mean_flow = mpl.colors.LogNorm(vmin=1, vmax=10000)

# --------
# annual centroid day change from control period
cmap_centroid_diff = LinearSegmentedColormap.from_list('custom1', 
                                             [(0.0, 'xkcd:red'),
                                              (50/60, 'xkcd:white'),
                                              (1.0, 'xkcd:blue')], N=255)
cmap_centroid_diff.set_over('xkcd:dark blue')
cmap_centroid_diff.set_under('xkcd:dark red')
norm_centroid_diff = mpl.colors.Normalize(vmin=-50, vmax=10)

# --------
# annual maximum date change from control period
cmap_max_day_diff = LinearSegmentedColormap.from_list('custom1', 
                                             [(0.0, 'xkcd:red'),
                                              (60/80, 'xkcd:white'),
                                              (1.0, 'xkcd:blue')], N=255)
cmap_max_day_diff.set_over('xkcd:dark blue')
cmap_max_day_diff.set_under('xkcd:dark red')
norm_max_day_diff = mpl.colors.Normalize(vmin=-60, vmax=20)

# --------
# annual minimum date change from control period
cmap_min_day_diff = LinearSegmentedColormap.from_list('custom1', 
                                             [(0.0,    'xkcd:red'),
                                              (50/150, 'xkcd:white'),
                                              (1.0,    'xkcd:blue')], N=255)
cmap_min_day_diff.set_over('xkcd:dark blue')
cmap_min_day_diff.set_under('xkcd:dark red')
norm_min_day_diff = mpl.colors.Normalize(vmin=-50, vmax=100)

# --------
# annual maximum flow change from control period
# cms
cmap_max_flow_diff = LinearSegmentedColormap.from_list('custom1', 
                                             [(0.0,      'xkcd:red'),
                                              (100/1600, 'xkcd:white'),
                                              (1.0,      'xkcd:blue')], N=255)
cmap_max_flow_diff.set_over('xkcd:dark blue')
cmap_max_flow_diff.set_under('xkcd:dark red')
norm_max_flow_diff=mpl.colors.Normalize(vmin=-100, vmax=1500)

# percent diff
cmap_max_flow_pdiff = LinearSegmentedColormap.from_list('custom1', 
                                             [(0.0,      'xkcd:red'),
                                              (20/70, 'xkcd:white'),
                                              (1.0,      'xkcd:blue')], N=255)
cmap_max_flow_pdiff.set_over('xkcd:dark blue')
cmap_max_flow_pdiff.set_under('xkcd:dark red')
norm_max_flow_pdiff=mpl.colors.Normalize(vmin=-20, vmax=50)

# annual maximum flow
norm_max_flow = mpl.colors.LogNorm(vmin=20, vmax=15000)

# --------
# annual minmum flow change from control period
cmap_min_flow_diff=LinearSegmentedColormap.from_list('custom1', 
                                             [(0.0, 'xkcd:red'),
                                              (20/520, 'xkcd:white'),
                                              (1.0, 'xkcd:blue')], N=255)
cmap_min_flow_diff.set_over('xkcd:dark blue')
cmap_min_flow_diff.set_under('xkcd:dark red')
norm_min_flow_diff=mpl.colors.Normalize(vmin=-20, vmax=500)
# annual minimum flow
norm_min_flow = mpl.colors.LogNorm(vmin=1, vmax=2000)

# --------
# freq_high_q per yr
vals1=[-4, -3, -2, -1, 0, 1, 2, 3, 4]
cmap_freq_high_q_diff = LinearSegmentedColormap.from_list('custom1', 
                                             [(0.0, 'xkcd:red'),
                                              (0.5, 'xkcd:white'),
                                              (1.0, 'xkcd:blue')], N=10)
cmap_freq_high_q_diff.set_over('xkcd:dark blue')
cmap_freq_high_q_diff.set_under('xkcd:dark red')
norm_freq_high_q_diff = mpl.colors.BoundaryNorm(vals1, cmap_freq_high_q_diff.N)
norm_freq_high_q = mpl.colors.Normalize(vmin=0, vmax=10)

# --------
# mean_high_q_duration per yr
cmap_freq_high_dur_diff=LinearSegmentedColormap.from_list('custom1', 
                                             [(0.0, 'xkcd:red'),
                                              (10/20, 'xkcd:white'),
                                              (1.0, 'xkcd:blue')], N=255)
cmap_freq_high_dur_diff.set_over('xkcd:dark blue')
cmap_freq_high_dur_diff.set_under('xkcd:dark red')
norm_freq_high_dur_diff=mpl.colors.Normalize(vmin=-10, vmax=10)
norm_freq_high_dur = mpl.colors.Normalize(vmin=0, vmax=20)