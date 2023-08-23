import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt

DATASETS  = ['UCIHAR', 'MotionSense', 'ExtraSensory']
SOLUTIONS = ['FedAvg-None', 'POC-POC-0.5', 'DEEV-DEEV-0.01', 'DEEV-PER-DEEV-0.01', 'DEEV-PER-SHARED-3-DEEV-0.01']
NAMES     = {'FedAvg-None' : 'FedAvg', 'POC-POC-0.5' : 'POC', 'DEEV-DEEV-0.01' : 'DEEV', 'DEEV-PER-DEEV-0.01' : 'ACSP-FL', 'DEEV-PER-SHARED-3-DEEV-0.01' : 'ACSP-FL LR'}
COLORS    = {'FedAvg-None' : 'r', 'POC-POC-0.5' : 'k', 'DEEV-DEEV-0.01' : 'gray', 'DEEV-PER-DEEV-0.01' : 'b', 'DEEV-PER-SHARED-3-DEEV-0.01' : 'lime'}
LINE      = {'FedAvg-None' : '-', 'POC-POC-0.5' : '-.', 'DEEV-DEEV-0.01' : ':', 'DEEV-PER-DEEV-0.01' : '--', 'DEEV-PER-SHARED-3-DEEV-0.01' : '-'}
MARKER    = {'FedAvg-None' : 'o', 'POC-POC-0.5' : '+', 'DEEV-DEEV-0.01' : 's', 'DEEV-PER-DEEV-0.01' : 'o', 'DEEV-PER-SHARED-3-DEEV-0.01' : '*'}

def plot_acc_savefig():
    
    
    for idx, ds in enumerate(DATASETS):
        fig, ax = plt.subplots(figsize=(5, 5))
        for sol in SOLUTIONS:
            
            df = pd.read_csv(f'../logs/{ds}/{sol}/DNN/server.csv',
                             names=['timestamp', 'round', 'acc', 'acc2', 'acc3'])
            
            sns.lineplot(x='round', y=df['acc'].rolling(3).mean(), data=df, ax=ax, label=NAMES[sol], 
                         color=COLORS[sol], linestyle=LINE[sol],
                         linewidth=2.5)
            if idx == 0:
                ax.set_ylim(0.75, 0.96)

            elif idx == 1:
                ax.set_ylim(0.5, 0.8)
            else:    
                ax.set_ylim(0.75, 0.95)

            ax.grid(True, linestyle=':')
            # ax.set_title(f'{ds}')
            ax.set_xlabel(f'Communication Round (#)', size=14)
            ax.set_ylabel(f'Test Accuracy (%)', size=14)
        
        fig.savefig(f'plots/accuracy_{ds}.pdf', dpi=200, format="pdf", bbox_inches="tight")

def plot_acc():
    fig, ax    = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
    
    for sol in SOLUTIONS:
        for idx, ds in enumerate(DATASETS):
            df = pd.read_csv(f'../logs/{ds}/{sol}/DNN/server.csv',
                             names=['timestamp', 'round', 'acc', 'acc2', 'acc3'])
            
            sns.lineplot(x='round', y=df['acc'].rolling(3).mean(), data=df, ax=ax[idx], label=NAMES[sol], 
                         color=COLORS[sol], linestyle=LINE[sol],
                         linewidth=2.5)
            # ax[0].set_ylim(0.75, 0.92)
            ax[0].set_ylim(0.75, 0.95)
            ax[1].set_ylim(0.5, 0.8)
            ax[2].set_ylim(0.75, 0.92)
            ax[idx].grid(True, linestyle=':')
            ax[idx].set_title(f'{ds}')
            ax[idx].set_xlabel(f'Communication Round (#)', size=14)
            ax[idx].set_ylabel(f'Test Accuracy (%)', size=14)

def plot_net():
    fig, ax  = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
    for sol in SOLUTIONS:
        if 'FedAvg' in sol: continue
        for idx, ds in enumerate(DATASETS):
            df_fedavg = pd.read_csv(f'../logs/{ds}/FedAvg-None/DNN/train_client.csv',
                             names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
            df_fedavg = df_fedavg.groupby('round').sum()

            df_poc = pd.read_csv(f'../logs/{ds}/POC-POC-0.5/DNN/train_client.csv',
                             names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
            df_poc = df_poc.groupby('round').sum()
            
            df = pd.read_csv(f'../logs/{ds}/{sol}/DNN/train_client.csv',
                             names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
            
            
            poc_value = np.mean(df_poc['param'].values/df_fedavg['param'].values)

            df_grouped = df.groupby('round').sum()
            sol_param  = df_grouped['param'].rolling(5).mean()
            ax[idx].plot(range(100), 1 - sol_param/df_fedavg['param'].values, label=f'{NAMES[sol]} vs FedAvg', 
                         color=COLORS[sol], linestyle=LINE[sol],
                         linewidth=2.5, )
            # ax[idx].axhline(poc_value, linestyle='--', color='k', label='POC vs FedAvg')
            # ax[idx].plot(range(100), 1 - df_grouped['param'].values/df_poc['param'].values, label=f'{NAMES[sol]} vs POC', 
            #              color=COLORS[sol], linestyle='--',
            #              linewidth=2.5, )
            
            # ax[0].set_ylim(0.75, 0.92)
            # ax[0].set_ylim(0.5, 0.92)
            # ax[1].set_ylim(0.5, 1.01)
            # ax[2].set_ylim(0.7, 0.92)
            ax[idx].legend()
            # ax[idx].set_yscale('log')
            ax[idx].grid(True, linestyle=':')
            ax[idx].set_title(f'{ds}')
            ax[idx].set_xlabel(f'Communication Round (#)', size=14)
            ax[idx].set_ylabel(f'Reduction in Communication (%)', size=14)

def plot_time_savefig():
    
    for idx, ds in enumerate(DATASETS):
        fig, ax  = plt.subplots(figsize=(5, 5))
        for sol in SOLUTIONS:
            if 'FedAvg' in sol: continue
            df_fedavg = pd.read_csv(f'../logs/{ds}/FedAvg-None/DNN/train_client.csv',
                             names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
            df_fedavg = df_fedavg.groupby('round').sum()
            avg_time  = df_fedavg['time'].mean()
            
            df = pd.read_csv(f'../logs/{ds}/{sol}/DNN/train_client.csv',
                             names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
            
            
            df_grouped = df.groupby('round').sum()
            sol_time   = df_grouped['time'].rolling(5).mean()
            ax.plot(range(100), 1 - sol_time/df_fedavg['time'].values, label=NAMES[sol], 
                         color=COLORS[sol], linestyle=LINE[sol],
                         linewidth=2.5, )

            ax.legend()
            ax.set_ylim(0.4, 1)
            ax.grid(True, linestyle=':')
            # ax.set_title(f'{ds}')
            ax.set_xlabel(f'Communication Round (#)', size=14)
            ax.set_ylabel(f'Lantecy Reduction (%)', size=14)

        fig.savefig(f'plots/latency_{ds}.pdf', dpi=200, format="pdf", bbox_inches="tight")


def plot_time():
    fig, ax  = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
    for sol in SOLUTIONS:
        if 'FedAvg' in sol: continue
        for idx, ds in enumerate(DATASETS):
            df_fedavg = pd.read_csv(f'../logs/{ds}/FedAvg-None/DNN/train_client.csv',
                             names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
            df_fedavg = df_fedavg.groupby('round').sum()
            avg_time  = df_fedavg['time'].mean()
            
            df = pd.read_csv(f'../logs/{ds}/{sol}/DNN/train_client.csv',
                             names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
            
            
            df_grouped = df.groupby('round').sum()
            sol_time   = df_grouped['time'].rolling(5).mean()
            ax[idx].plot(range(100), 1 - sol_time/df_fedavg['time'].values, label=NAMES[sol], 
                         color=COLORS[sol], linestyle=LINE[sol],
                         linewidth=2.5, )
            
            # ax[0].set_ylim(0.75, 0.92)
            # ax[0].set_ylim(0.5, 0.92)
            # ax[1].set_ylim(0.5, 1.01)
            # ax[2].set_ylim(0.7, 0.92)
            ax[idx].legend()
            ax[idx].set_ylim(0.4, 1)
            ax[idx].grid(True, linestyle=':')
            ax[idx].set_title(f'{ds}')
            ax[idx].set_xlabel(f'Communication Round (#)', size=14)
            ax[idx].set_ylabel(f'Lantecy Reduction (%)', size=14)

def plot_hist_acc():
    fig, ax  = plt.subplots(nrows=5, ncols=3, figsize=(12, 8))
    
    for idx_sol, sol in enumerate(SOLUTIONS):
        for idx, ds in enumerate(DATASETS):

            df = pd.read_csv(f'../logs/{ds}/{sol}/DNN/evaluate_client.csv',
                             names=['round', 'cid', 'param', 'loss', 'acc'])
                                        
            df_filtered = df[df['round']== 100]
            sns.histplot(df_filtered['acc'].values, label=NAMES[sol], ax=ax[idx_sol, idx],
                         color=COLORS[sol],  kde=True, bins=15)
            
            if idx > 0:
                ax[idx_sol, idx].set_ylabel('')    
            # ax[idx_sol, idx].legend()
            ax[idx_sol, idx].set_axisbelow(True)
            ax[idx_sol, idx].grid(True, linestyle=':')

def plot_cdf_acc_savefig(): 
    for idx, ds in enumerate(DATASETS):
        fig, ax  = plt.subplots(figsize=(5, 5))

        for idx_sol, sol in enumerate(SOLUTIONS):

            df = pd.read_csv(f'../logs/{ds}/{sol}/DNN/evaluate_client.csv',
                             names=['round', 'cid', 'param', 'loss', 'acc'])
            
            df_filtered = df[df['round']== 100]
            sns.ecdfplot(y=df_filtered['acc'].values, label=NAMES[sol], ax=ax,
                         color=COLORS[sol], linewidth=2, linestyle=LINE[sol])
            
            ax.legend()
            ax.set_axisbelow(True)
            ax.set_xlabel('Proportion', size=14)
            ax.set_ylabel('Test Accuracy', size=14)
            ax.grid(True, linestyle=':')
        
        fig.savefig(f'plots/acc_cdf_{ds}.pdf', dpi=200, format="pdf", bbox_inches="tight")


def plot_cdf_acc():
    fig, ax  = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
    fig.add_gridspec(3, hspace=0)
    for idx_sol, sol in enumerate(SOLUTIONS):
        for idx, ds in enumerate(DATASETS):

            df = pd.read_csv(f'../logs/{ds}/{sol}/DNN/evaluate_client.csv',
                             names=['round', 'cid', 'param', 'loss', 'acc'])
            
            df_filtered = df[df['round']== 100]
            sns.ecdfplot(y=df_filtered['acc'].values, label=NAMES[sol], ax=ax[idx],
                         color=COLORS[sol], linewidth=2, linestyle=LINE[sol])
            
            ax[idx].legend()
            ax[idx].set_axisbelow(True)
            ax[idx].grid(True, linestyle=':')

def client_selected():
    fig, ax  = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    for sol in SOLUTIONS:
        if 'FedAvg' in sol or 'POC' in sol: continue
        for idx, ds in enumerate(DATASETS):
            
            df = pd.read_csv(f'../logs/{ds}/{sol}/DNN/train_client.csv',
                             names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
            
            
            df_grouped = df.groupby('round').sum()
            selected   = df_grouped['selected'].rolling(5).mean()
            sns.histplot(selected, label=NAMES[sol], ax=ax[idx],
                         color=COLORS[sol],  bins=20,  multiple="dodge")
            
            # ax[0].set_ylim(0.75, 0.92)
            # ax[0].set_ylim(0.5, 0.92)
            # ax[1].set_ylim(0.5, 1.01)
            # ax[2].set_ylim(0.7, 0.92)
            ax[idx].legend()
            # ax[idx].set_ylim(0.4, 1)
            ax[idx].grid(True, linestyle=':')
            ax[idx].set_title(f'{ds}')
            ax[idx].set_xlabel(f'Communication Round (#)', size=14)
            ax[idx].set_ylabel(f'Lantecy Reduction (%)', size=14)

DATASETS   = ['UCIHAR', 'MotionSense', 'ExtraSensory']
SOLUTIONS2 = ['DEEV-PER-DEEV-0.01', 'DEEV-PER-SHARED-DEEV-0.01', 'DEEV-PER-SHARED-2-DEEV-0.01', 'DEEV-PER-SHARED-3-DEEV-0.01']
NAMES2     = {'DEEV-PER-DEEV-0.01' : 'Personalization shared layers = 0', 
              'DEEV-PER-SHARED-DEEV-0.01' : 'Personalization shared layers = 1', 
              'DEEV-PER-SHARED-2-DEEV-0.01' : 'Personalization shared layers = 2', 
              'DEEV-PER-SHARED-3-DEEV-0.01': 'Personalization shared layers = 3'}
COLORS2    = {'DEEV-PER-DEEV-0.01' : 'r', 'DEEV-PER-SHARED-DEEV-0.01' : 'k', 'DEEV-DEEV-0.01' : 'gray', 'DEEV-PER-SHARED-2-DEEV-0.01' : 'b', 'DEEV-PER-SHARED-3-DEEV-0.01' : 'lime'}
LINE2      = {'DEEV-PER-DEEV-0.01' : '-', 'DEEV-PER-SHARED-DEEV-0.01' : '-.', 'DEEV-DEEV-0.01' : ':', 'DEEV-PER-SHARED-2-DEEV-0.01' : '--', 'DEEV-PER-SHARED-3-DEEV-0.01' : '-'}
MARKER2    = {'DEEV-PER-DEEV-0.01' : 'o', 'DEEV-PER-SHARED-DEEV-0.01' : '+', 'DEEV-DEEV-0.01' : 's', 'DEEV-PER-SHARED-2-DEEV-0.01' : 'o', 'DEEV-PER-SHARED-3-DEEV-0.01' : '*'}

def plot_acc_devper():
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
    for sol in SOLUTIONS2:
        for idx, ds in enumerate(DATASETS):
            df = pd.read_csv(f'../logs/{ds}/{sol}/DNN/server.csv',
                             names=['timestamp', 'round', 'acc', 'acc2', 'acc3'])
            
            sns.lineplot(x='round', y=df['acc'].rolling(3).mean(), data=df, ax=ax[idx], label=NAMES2[sol], 
                         color=COLORS2[sol], linestyle=LINE2[sol],
                         linewidth=2.5)
            # ax[0].set_ylim(0.75, 0.92)
            # ax[0].set_ylim(0.75, 0.95)
            # ax[1].set_ylim(0.5, 0.8)
            # ax[2].set_ylim(0.75, 0.92)
            ax[idx].grid(True, linestyle=':')
            ax[idx].set_title(f'{ds}')
            ax[idx].set_xlabel(f'Communication Round (#)', size=14)
            ax[idx].set_ylabel(f'Test Accuracy (%)', size=14)

def plot_net_devper():
    fig, ax  = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
    for sol in SOLUTIONS2:
        if 'DEEV-PER-DEEV-0.01' == sol: continue
        for idx, ds in enumerate(DATASETS):
            df_fedavg = pd.read_csv(f'../logs/{ds}/FedAvg-None/DNN/train_client.csv',
                             names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
            df_fedavg = df_fedavg.groupby('round').sum()

            df_poc = pd.read_csv(f'../logs/{ds}/POC-POC-0.5/DNN/train_client.csv',
                             names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
            df_poc = df_poc.groupby('round').sum()
            
            df = pd.read_csv(f'../logs/{ds}/{sol}/DNN/train_client.csv',
                             names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
            
            
            poc_value = np.mean(df_poc['param'].values/df_fedavg['param'].values)

            df_grouped = df.groupby('round').sum()
            sol_param  = df_grouped['param'].rolling(5).mean()
            ax[idx].plot(range(100), 1 - sol_param/df_fedavg['param'].values, label=f'{NAMES2[sol]} vs FedAvg', 
                         color=COLORS2[sol], linestyle=LINE2[sol],
                         linewidth=2.5, )
            # ax[idx].axhline(poc_value, linestyle='--', color='k', label='POC vs FedAvg')
            # ax[idx].plot(range(100), 1 - df_grouped['param'].values/df_poc['param'].values, label=f'{NAMES[sol]} vs POC', 
            #              color=COLORS[sol], linestyle='--',
            #              linewidth=2.5, )
            
            # ax[0].set_ylim(0.75, 0.92)
            # ax[0].set_ylim(0.5, 0.92)
            # ax[1].set_ylim(0.5, 1.01)
            # ax[2].set_ylim(0.7, 0.92)
            ax[idx].legend()
            # ax[idx].set_yscale('log')
            ax[idx].grid(True, linestyle=':')
            ax[idx].set_title(f'{ds}')
            ax[idx].set_xlabel(f'Communication Round (#)', size=14)
            ax[idx].set_ylabel(f'Reduction in Communication (%)', size=14)

def plot_time_devper():
    fig, ax  = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
    for sol in SOLUTIONS2:
        if 'FedAvg' in sol: continue
        for idx, ds in enumerate(DATASETS):
            df_fedavg = pd.read_csv(f'../logs/{ds}/FedAvg-None/DNN/train_client.csv',
                             names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
            df_fedavg = df_fedavg.groupby('round').sum()
            avg_time  = df_fedavg['time'].mean()
            
            df = pd.read_csv(f'../logs/{ds}/{sol}/DNN/train_client.csv',
                             names=['round', 'cid', 'selected', 'time', 'param', 'loss', 'acc'])
            
            
            df_grouped = df.groupby('round').sum()
            sol_time   = df_grouped['time'].rolling(5).mean()
            ax[idx].plot(range(100), 1 - sol_time/df_fedavg['time'].values, label=NAMES2[sol], 
                         color=COLORS2[sol], linestyle=LINE2[sol],
                         linewidth=2.5, )
            
            # ax[0].set_ylim(0.75, 0.92)
            # ax[0].set_ylim(0.5, 0.92)
            # ax[1].set_ylim(0.5, 1.01)
            # ax[2].set_ylim(0.7, 0.92)
            ax[idx].legend()
            ax[idx].set_ylim(0.4, 1)
            ax[idx].grid(True, linestyle=':')
            ax[idx].set_title(f'{ds}')
            ax[idx].set_xlabel(f'Communication Round (#)', size=14)
            ax[idx].set_ylabel(f'Lantecy Reduction (%)', size=14)