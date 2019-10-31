from skimage.morphology import skeletonize
from skimage import data
import sknw
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import os
from shapely.geometry import LineString

def render_graph(img):
    '''
    input  --> binary thresholded mask : x[x>0]=1
    output --> skeleton graph
    '''
    ### skeletonize the input mask
    ske = skeletonize(img).astype(np.uint16)
    ### create graph
    graph = sknw.build_sknw(ske)
    return graph

def graph2linestring(graph,img_size,thresh,figsize,verbose=False,plot=True,write2csv=False,path=None,ret_graph=False):
    '''
    input  --> skeleton graph
    output --> list of (start,end) nodes
    if graph is to be returned, set ret_graph to True
    '''
    data=[]
    ### create sample mask for input image size
    map_=np.zeros(shape=img_size)
    ### iter over edges of graph
    for iter_,(s,e) in enumerate(graph.edges()):
        #calc num of points for current edge
        num_points=len(graph[s][e]['pts'])
        #print(num_points)
        #start pixel and end pixel of current edge
        start,end = graph[s][e]['pts'][0],graph[s][e]['pts'][-1]
        if plot:
            cv2.circle(map_,(start[0],start[1]),10, (255,255,255),-1)
        if verbose:
            print(start,end)
        #check if length of edge exceeds smoothening thresh
        if (num_points>thresh):
            sub_data=[]
            #start with initial point of  graph edge
            curr=graph[s][e]['pts'][0]
            sub_data.append(choose_neighbour(curr[0],curr[1]))
            for i in range(thresh,num_points+thresh,thresh):
                #detct end pixel
                if i>=num_points-thresh:
                    #plot line for latest updated smmother pixel and end pixel
                    cv2.line(map_,pt1=(curr[0],curr[1]),pt2=(end[0],end[1]),color=255,thickness=2)
                    sub_data.append(choose_neighbour(end[0],end[1]))
                    if plot:
                        cv2.circle(map_,(end[0],end[1]),10, (255,255,255),-1)
                    if verbose:
                        print(start,end,(curr[0],curr[1]),(end[0],end[1]),i,num_points)
                    break
                #find next smoother pixel
                next_=graph[s][e]['pts'][i]
                #plot line between smoothening pixel
                if plot:
                    cv2.line(map_,pt1=(curr[0],curr[1]),pt2=(next_[0],next_[1]),color=255,thickness=2)
                #append
                sub_data.append((next_[0],next_[1]))
                if verbose:
                    print(start,end,curr,next_,i,num_points)
                curr=next_
                #plot node
                cv2.circle(map_,(curr[0],curr[1]),10, (255,255,255), -1)
            data.append(LineString(sub_data))
            if verbose:
                print(LineString(sub_data))
        #if current pixel does not exceed threshold
        else:
            #plot line between start pixel and end pixel
            cv2.line(map_,pt1=(start[0],start[1]),pt2=(end[0],end[1]),color=255,thickness=2)
            cv2.circle(map_,(end[0],end[1]),10, (255,255,255),-1)
            #line = LineString([(start[0],start[1]), (end[0],end[1])])
            data.append(LineString([choose_neighbour(start[0],start[1]),choose_neighbour(end[0],end[1])]))
            if verbose:
                print(LineString([choose_neighbour(start[0],start[1]),choose_neighbour(end[0],end[1])]))
            #data.append(((start[0],start[1]), (end[0],end[1])))
        ### a helper statement :ignore
        #if iter_==1:
        #    break
        ####
    if plot:
        fig,ax=plt.subplots(1,1,figsize=figsize)
        ax.imshow(map_,cmap='gray')
    if write2csv:
        df=pd.DataFrame({'linestring':data})
        df.to_csv(path.split('/')[-1][:-4]+'.csv')
    if ret_graph:
        return graph
    else:
        return data

def create_nodes_pool(graph):   
    global valid_nodes
    valid_nodes=[]
    node, nodes = graph.node, graph.nodes()
    for i in nodes :
        valid_nodes.append(node[i]['o'])
    return valid_nodes

def choose_neighbour(x,y):
    point=(x,y)
    dist=[]
    for point_ in valid_nodes:
        line=LineString([point,point_])
        dist.append(line.length)
    min_dist=np.argmin(dist)
    #print('valid_neighbour for {} is {}'.format(point,valid_nodes[min_dist]))
    return valid_nodes[min_dist]

def mask2linestring(img,thresh,verbose=False,plot=True,write2csv=False,path=None,figsize=(10,10),ret_graph=False):
    if thresh == [] or thresh == None:
        print('Thresh is mandatory ,use default 80 . Returning None')
        return None
    graph=render_graph(img)
    img_size=(img.shape[0],img.shape[1])
    create_nodes_pool(graph)
    return graph2linestring(graph,img_size,thresh,figsize,verbose,plot,write2csv,path,ret_graph)
