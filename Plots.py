import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

new_coordinate_list = []
data_set = []
origin = []
speed = []
heading = []
def get_coords(r, t, h):
    return 0
    print()


coordinate = origin
for i in range(len(data_set)):
    new_coordinate = get_coords(coordinate, speed[i], heading[i])
    new_coordinate_list.append(new_coordinate)

    coordinate = new_coordinate


index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'all']
t1 = [0.926162027, 0.915016866, 0.944345234, 0.947436924, 0.942142802, 0.950546742, 0.958182533, 0.952890048, 0.94444993, 0.959789982, 0.945630693]
t2 = [0.898433036	,0.907145139	,0.942741483	,0.953654032	,0.960969	,0.959754117	,0.960472415	,0.965472712	,0.964503303	,0.959431094	,0.972188614]
t3 = [0.875754696	,0.894673121	,0.940988019	,0.943938691	,0.944370228	,0.946042603	,0.946966037	,0.947727185	,0.948278931	,0.948353336	,0.949391766]
t4 = [0.932151526	,0.946591581	,0.958397295	,0.956047833	,0.964449702	,0.965309038	,0.968977789	,0.968320487	,0.967256945	,0.9668379	,0.950815453]
t5 = [0.922746457	,0.931898815	,0.940993412	,0.93949047	,0.950196612	,0.946521402	,0.944888144	,0.939330881	,0.950739864	,0.939551783	,0.94524833]

d1 = [0.19047111	,0.479082669	,0.256694119	,0.116098662	,0.679673146	,0.661945358	,0.643692901	,0.697537369	,0.745464066	,0.753731059	,0.893759463]
d2 = [0.668711004	,0.583788987	,0.369280413	,0.344905229	,0.745845668	,0.740786837	,0.771414197	,0.827550007	,0.80135701	,0.767123316	,0.94087949]
d3 = [0.21172775	,0.400324406	,0.399927591	,0.387354107	,0.360748974	,0.361208345	,0.374809771	,0.379034555	,0.435936265	,0.439307826	,0.987193645]
d4 = [-0.207153417,	0.192423601, 0.404089562	,0.405052796	,0.727544278	,0.730104879	,0.767545806	,0.764988823	,0.741844496	,0.740672919	,0.992205626]
d5 = [0.04081415	,0.008119508	,-0.083583091	,0.223194524	,0.414780714	,0.434842619	,0.382771566	,0.407409547	,0.280427587	,0.227275016	,0.856885125]

t1_patch = mpatches.Patch(color='blue', label='Decision Tree')
t2_patch = mpatches.Patch(color='red', label='Random Forrest')
t3_patch = mpatches.Patch(color='green', label='Linear Regression')
t4_patch = mpatches.Patch(color='yellow', label='Support Vector Regression')
t5_patch = mpatches.Patch(color='purple', label='Multi Layer Regression')

plt.subplot(211)
plt.xlabel('# of features')
plt.ylabel('R2 Score')
plt.title('Feature Selection - Speed over ground')
plt.legend(handles=[t1_patch, t2_patch, t3_patch, t4_patch, t5_patch])

plt.plot(index, t1, 'b', index, t2, 'r', index, t3, 'g', index, t4, 'y', index, t5, 'purple')

plt.subplot(212)
plt.subplots_adjust(hspace=0.5)
plt.xlabel('# of features')
plt.ylabel('R2 Score')
plt.title('Feature Selection - Heading')
plt.legend(handles=[t1_patch, t2_patch, t3_patch, t4_patch, t5_patch])

plt.plot(index, d1, 'b', index, d2, 'r', index, d3, 'g', index, d4, 'y', index, d5, 'purple')


# cut_s = 1000
# cut_e = 5000
# plt.plot(lat_true[cut_s:cut_e], lon_true[cut_s:cut_e], 'b', lat_new[cut_s:cut_e], lon_new[cut_s:cut_e], 'r')

plt.show()