#!/usr/bin/python3

import os
import fnmatch
from shutil import copyfile
from pprint import pprint
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value
 
def clustering_analysis(pca_df):
    pcs = list(pca_df)
    K = range(1,len(pcs))
    X = []
    for pc, i in zip(pcs, range(len(pcs))): X.append(pca_df[str(pc)])
    X = np.array([X])
    KM = [KMeans(n_clusters=k).fit(X[0]) for k in K]
    centroids = [k.cluster_centers_ for k in KM]
    D_k = [cdist(X[0], cent, 'euclidean') for cent in centroids]
    dist = [np.min(D,axis=1) for D in D_k]
    avgWithinSS = [sum(d)/X[0].shape[0] for d in dist]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(X[0])**2)/X[0].shape[0]
    bss = tss-wcss
    return (avgWithinSS, tss, bss)

def plot_SumOfSquares(K, avgWithinSS, tss, bss, root, fn):    
    output_dir = os.path.join(root, fn)    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, avgWithinSS, 'b*-')
    ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
            markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    fn = os.path.splitext(filename)[0]
    fig.savefig(os.path.join(output_dir, fn + '_someofsquares.png'))
    fig.savefig(os.path.join(output_dir, fn + '_someofsquares.pdf'))
    plt.close()
    plt.clf()
    
def plot_VarianceExplained(K, avgWithinSS, tss, bss, root, fn):    
    output_dir = os.path.join(root, fn)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, bss/tss*100, 'b*-')
    ax.plot(K[kIdx], (bss/tss*100)[kIdx], marker='o', markersize=12, 
            markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained')
    plt.title('Elbow for KMeans clustering')
    fig.savefig(os.path.join(output_dir, fn + '_%variance_explained.png'))
    fig.savefig(os.path.join(output_dir, fn + '_%_variance_explained.pdf'))
    plt.close()
    plt.clf()

def dist(i, centers, ctrl, r):
    d = (i[r] - centers[ctrl][r])**2
    return(d)

def plotColoredPCA(rootfn, pca_df, clusters, kIdx):
    colored_pca = os.path.join(rootfn, 'colored_pca')
    if not os.path.exists(colored_pca):
        os.makedirs(colored_pca)
        
    norm = plt.Normalize()
    colors = plt.cm.jet(norm(clusters))
    
    d3 = os.path.join(colored_pca, '3D')
    if not os.path.exists(d3):
        os.makedirs(d3)
        
    for i in range(1,kIdx+2):
        for j in range(1,kIdx+2):
            for k in range(1,kIdx+2):
                if i >= j or i >= k or j >= k:
                    continue
                fig = plt.figure()
                ax = fig.add_subplot(111, projection = '3d')
                for l in range(len(pca_df)): #plot each point + it's index as text above
                              ax.scatter(pca_df[str(i)][l],pca_df[str(j)][l],pca_df[str(k)][l],color=colors[l])  
                              ax.text(pca_df[str(i)][l],pca_df[str(j)][l],pca_df[str(k)][l],  '%s' % (str(pca_df.index[l])), zorder=1,  color=colors[l]) 
                ax.set_xlabel('PC'+str(i))
                ax.set_ylabel('PC'+str(j))
                ax.set_zlabel('PC'+str(k))
                plt.savefig(os.path.join(d3, 'PC'+str(i)+'_PC'+str(j)+'_PC'+str(k)+'.png'))
                plt.close()
    d2 = os.path.join(colored_pca, '2D')
    if not os.path.exists(d2):
        os.makedirs(d2)
        
    for n in range(1,kIdx+2):
        for m in range(1,kIdx+2):
            if n >= m:
                continue
            fig, ax = plt.subplots()
            for i in range(len(pca_df)): #plot each point + it's index as text above
                ax.scatter(pca_df[str(n)][i],pca_df[str(m)][i],color=colors[i]) 
                ax.text(pca_df[str(n)][i],pca_df[str(m)][i],  '%s' % (str(pca_df.index[i])), zorder=1,  color=colors[i]) 
            ax.set_xlabel('PC'+str(n))
            ax.set_ylabel('PC'+str(m))
            plt.savefig(os.path.join(d2, 'PC'+str(n)+'_PC'+str(m)+'.png'))
            plt.close()
            
def clustering(K, pca_df, dia_df, rootfn, kIdx):
    indices = []
    for i in pca_df.index: indices.append(i)
    s = indices.index('control')
    pcs = list(pca_df)
    K = len(pcs)
    X = []
    for pc, i in zip(pcs, range(len(pcs))): X.append(pca_df[str(pc)])
    X = np.array([X]) 
    kmeans_c = KMeans(n_clusters=kIdx+1, random_state=0).fit(X[0].T)
    clusters = kmeans_c.labels_
    centers = kmeans_c.cluster_centers_
    ctrl = kmeans_c.labels_[s]
    cluster_dists = []
    
    for i in centers:
        centsum = dist(i,centers,ctrl,0)
        for j in range(1,K):
            centsum+=(dist(i,centers,ctrl,j))
        cluster_dists.append((centsum)**0.5)

    
    sorted_dists = np.sort(cluster_dists)
    sorted_index = []
    for i in sorted_dists:
        sorted_index.append(cluster_dists.index(i))
    
    new_clusters = []
    for i in clusters:
        new_clusters.append(sorted_index.index(i))
    plotColoredPCA(rootfn, pca_df, clusters, kIdx)
    cmpds = dia_df.loc['Compound']
    pos = pd.to_numeric(dia_df.loc['Positive'])
    neg = pd.to_numeric(dia_df.loc['Negative'])
    if s == 0:   
        new_dict = {"Compounds": cmpds, "Positive" : pos, "Negative" : neg, "Clusters" : new_clusters[1:len(new_clusters)]}        
    
    if s != 0:
        new_dict = {"Compounds": cmpds, "Positive" : pos, "Negative" : neg, "Clusters" : new_clusters[0:len(new_clusters)-1]}
    
    new_df = pd.DataFrame(data=new_dict)
    return(new_df)

def posVneg(df):
    sub = df.Positive - df.Negative
    subNorm = max(sub)/sub
    posSubNorm = df.Positive*subNorm
    negSubNorm = df.Negative*subNorm
    subAbs = abs(posSubNorm + negSubNorm)
    df['subAbs'] = subAbs
    return (df)

def plot_bars(df, pth, nm):
    norm = plt.Normalize()
    colors = plt.cm.jet(norm(df['Clusters'].values))
    plt.bar(range(1, len(df.Positive) + 1, 1), df.Positive, align = 'center',
                color=colors)#, label = 'Positive')
    plt.bar(range(1, len(df.Positive) + 1, 1), df.Negative, align = 'center',
                color=colors)#, label = 'Negative')
    plt.axhline(np.std(df.Positive) + np.mean(df.Positive)/2, c ='k', ls = '-')#, label = '1 std')
    plt.axhline(-1*np.std(df.Negative) + np.mean(df.Negative)/2, c ='k', ls = '-')
    plt.xlim(0,len(df.Positive) + 1)
    plt.axhline(2*np.std(df.Positive) + np.mean(df.Positive)/2, c ='k', ls = '--')#, label = '2 std')
    plt.axhline(-2*np.std(df.Negative) + np.mean(df.Negative)/2, c ='k', ls = '--')
    plt.xlabel('Compound')
    plt.ylabel('Magnitudes')
    plt.title('Clustered DIA')
    plt.xticks(np.arange(start = 1, stop = len(df.Compounds) + 1), df.Compounds, rotation = 70, fontsize = 5)
#    plt.legend(loc = 4)
    plt.grid()
    plt.savefig(os.path.join(pth, nm + '.pdf'), dpi=600)
    plt.savefig(os.path.join(pth, nm + '.png'), dpi=600)
    plt.close()
    plt.clf()
    
def sort_overlays(overlay_root, cluster_root, new_df):
    directory = os.path.join(cluster_root, 'clustered_overlays')
    clust_pngDir = os.path.join(directory, 'png')
    clust_pdfDir = os.path.join(directory, 'pdf')
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(clust_pngDir)
        os.makedirs(clust_pdfDir)
    
    pngDir = os.path.join(overlay_root, 'png')
    pdfDir = os.path.join(overlay_root, 'pdf')
    for i in new_df.Clusters.unique():
          cmpds = new_df.loc[new_df.Clusters == i, 'Compounds']
          png_cluster = os.path.join(clust_pngDir, str(i))
          pdf_cluster = os.path.join(clust_pdfDir, str(i))
          if not os.path.exists(png_cluster):
              os.makedirs(png_cluster)
          
          if not os.path.exists(pdf_cluster):
              os.makedirs(pdf_cluster)
          
          for c in cmpds:
              pattpng = '*-' + str(c) + '-*.png'    
              pattpdf = '*-' + str(c) + '-*.pdf'
              for filename in fnmatch.filter(os.listdir(pngDir), pattpng):
                      copyfile(os.path.join(pngDir, filename), os.path.join(png_cluster, filename))
                      
              for filename in fnmatch.filter(os.listdir(pdfDir), pattpdf):
                      copyfile(os.path.join(pdfDir, filename), os.path.join(pdf_cluster, filename))
   
def write_summary(df):
   summary = Vividict()
   for i in df.Clusters.unique():
       cmpds = df.loc[df.Clusters == i, 'Compounds']
       for cmpd in cmpds:
           posmag = df.loc[df.Compounds == cmpd, 'Positive']
           if posmag[0] > (np.mean(df.Positive)/2 + 2*np.std(df.Positive)):
               summary['cluster ' + str(i)]['posmag > 2std'][cmpd]
           
           if posmag[0] < (np.mean(df.Positive)/2 + np.std(df.Positive)):
               summary['cluster ' + str(i)]['posmag < 1std'][cmpd]
                       
           if posmag[0] < (np.mean(df.Positive)/2 + 2*np.std(df.Positive)): 
               if posmag[0] > (np.mean(df.Positive)/2 + np.std(df.Positive)):
                   summary['cluster ' + str(i)]['posmag > 1std'][cmpd]
   return(summary)

def dia_color(K, filename, rootPath, pca_df, fn): 
    dia_pattern = '*magnitudes.csv'
    rootfn = os.path.join(rootPath, fn)
    if not os.path.exists(rootfn):
        os.makedirs(rootfn)
    
    dia_root = os.path.join(rootPath, '../DIA/')
    overlay_root = os.path.join(rootPath, '../overlays')
    for root, dirs, files in os.walk(dia_root):
        for filename1 in fnmatch.filter(files, dia_pattern):         
            dia_df = pd.read_csv(os.path.join(root, filename1), header = 0, index_col = 0)
            new_df = clustering(K, pca_df, dia_df, rootfn, kIdx)
            sortClust_df = new_df.sort_values(by='Clusters', ascending=False)
            sortDia_df = new_df.sort_values(by='Positive',  ascending=False)
            sortSub_df = posVneg(new_df).sort_values(by='subAbs', ascending=True)
            pca_fn = os.path.splitext(filename)[0]
            dia_fn = os.path.splitext(filename1)[0]
            path = pca_fn + "_" + dia_fn
            rootfn_path = os.path.join(rootfn, path + '_clustering')
            if not os.path.exists(rootfn_path):
                os.makedirs(rootfn_path)
            plot_bars(new_df, rootfn_path, "DIA_no_sorting")
            plot_bars(sortClust_df, rootfn_path, "DIA_cluster_sorted")
            plot_bars(sortDia_df, rootfn_path, "DIA_magnitude_sorted")
            plot_bars(sortSub_df, rootfn_path, "DIA_positive_vs_negative_sorted")
            new_df.to_csv(rootfn_path + '/' + path + '.xlsx')
            summary = write_summary(new_df)
            with open(rootfn_path + '/' + path + '_summary.txt', 'wt') as out:
                pprint(summary, stream=out)
            sort_overlays(overlay_root, rootfn_path, new_df)    
        
    

rootPath = "./"
pca_pattern = 'PCA*.csv'
kIdx = 5
for root, dirs, files in os.walk(rootPath):
    for filename in fnmatch.filter(files, pca_pattern):
        print(root + '/' + filename)
        fn = os.path.splitext(filename)[0] + '_clustering'
        if not os.path.exists(os.path.join(root, fn)):
            os.makedirs(os.path.join(root, fn))
        
        pca_df = pd.read_csv(os.path.join(root, filename), header = 0, index_col = 0)
        K = range(1, len(list(pca_df)))
        for tries in range(99999):       
            avgWithinSS, tss, bss = clustering_analysis(pca_df)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(K, bss/tss*100, 'b*-')
            ax.plot(K[kIdx], (bss/tss*100)[kIdx], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
            plt.grid(True)
            plt.xlabel('Number of clusters')
            plt.ylabel('Percentage of variance explained')
            plt.title('Elbow for KMeans clustering')
            plt.show()
            #plt.close()
            print('Type different number of clusters or leave blank to continue with current value.')
            try:
                inp = input()
            except:
                inp = raw_input()
                pass
            
            if inp == "":
                break
            else:
                kIdx = int(inp) - 1
                continue
        plt.close()
        plot_SumOfSquares(K, avgWithinSS, tss, bss, root, fn)        
        print('Clustering PCA...')
        dia_color(kIdx+1, filename, root, pca_df, fn)

