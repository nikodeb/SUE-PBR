import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset



# N=1
# S=2
# D=2
# U=3
#
# bert_out = torch.rand((N,S,D))
# users = torch.rand((N,U))
# mask = torch.rand((N,S))
# mask = mask>0.4
#
# proj = torch.rand(U,D)
# users_D = torch.matmul(users, proj)
# attn_scores = torch.matmul(bert_out, users_D.unsqueeze(-1)).squeeze(-1)
# attn_scores = torch.where(mask, torch.tensor(float('-inf')), attn_scores)
# attn_weights = F.softmax(attn_scores, dim=1)
# out = torch.matmul(attn_weights.unsqueeze(1), bert_out).squeeze(1)
#
# print('')

sse_loc = 'C:/Users/ndebo/Downloads/run-sse.csv'
core_loc = 'C:/Users/ndebo/Downloads/run-core.csv'
noreg_log = 'C:/Users/ndebo/Downloads/run-noreg.csv'

noreg_data = np.genfromtxt(noreg_log, dtype=float, delimiter=',', skip_header=1)
core_data = np.genfromtxt(core_loc, dtype=float, delimiter=',', skip_header=1)
sse_data = np.genfromtxt(sse_loc, dtype=float, delimiter=',', skip_header=1)

y_noreg = noreg_data[:,1]
x_noreg = noreg_data[:,0]
y_sse = sse_data[:,1]
x_sse = sse_data[:,0]
y_core = core_data[:,1]
x_core = core_data[:,0]

f, ax = plt.subplots()
plt.plot(x_noreg, y_noreg, '-', label='Pers w/o SSE')
plt.plot(x_sse, y_sse, '-', label='Pers w/ SSE')
plt.plot(x_core, y_core, '-', label='Unpers')
plt.ylabel('MLM Train Loss')
plt.xlabel('Epoch')
plt.title('Training Curves')
plt.legend(loc='upper right', frameon=True)


axins = zoomed_inset_axes(ax, 2.5, loc=10)
axins.plot(x_noreg, y_noreg, '-', label='Pers w/o SSE')
axins.plot(x_sse, y_sse, '-', label='Pers w/ SSE')
axins.plot(x_core, y_core, '-', label='Unpers')

x1, x2, y1, y2 = 170, 201, 4.25, 4.9 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits

mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")


f.savefig('C:\\Users\\ndebo\\PycharmProjects\\Personalising_Bert4Rec_git\\Images\\training_curve.pdf', bbox_inches='tight')

# plt.show()