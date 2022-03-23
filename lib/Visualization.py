import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

nColors=10
colorsRGB=[cm.tab10(x/(nColors-1))[:3] for x in range(nColors)]


def PlotXPartitions(atomicCells,partitionDataA,partitionDataB,shapeXL):
    fig=plt.figure()

    fig.add_subplot(1,3,1)

    img=np.zeros([np.prod(shapeXL),3],dtype=np.double)
    for i,cell in enumerate(atomicCells):
        img[cell]=colorsRGB[i%len(colorsRGB)]
    plt.imshow(img.reshape(shapeXL+[3,]))
    plt.title("atomic cells")
    plt.axis("off")
    
    fig.add_subplot(1,3,2)

    img=np.zeros([np.prod(shapeXL),3],dtype=np.double)
    for i,cell in enumerate(partitionDataA[0]):
        img[cell]=colorsRGB[i%len(colorsRGB)]
    plt.imshow(img.reshape(shapeXL+[3,]))
    plt.title("partition A")
    plt.axis("off")

    fig.add_subplot(1,3,3)

    img=np.zeros([np.prod(shapeXL),3],dtype=np.double)
    for i,cell in enumerate(partitionDataB[0]):
        img[cell]=colorsRGB[i%len(colorsRGB)]
    plt.imshow(img.reshape(shapeXL+[3,]))
    plt.title("partition B")
    plt.axis("off")

    plt.show()


def PlotYPartitions_SparseY(atomicCells,muYAtomicListData,muYAtomicListIndices,muY,cellColors=None):
    
    if cellColors is None:
        _cellColors=np.zeros((len(atomicCells),3),dtype=np.double)
        for i in range(len(atomicCells)):
            _cellColors[i]=colorsRGB[i%len(colorsRGB)]
    else:
        _cellColors=cellColors

    img=np.zeros(muY.shape+(3,),dtype=np.double)
    for i in range(len(atomicCells)):
        img[muYAtomicListIndices[i]]+=\
                np.einsum(\
                muYAtomicListData[i]/muY[muYAtomicListIndices[i]],[0],\
                _cellColors[i],[1],[0,1])
    img=np.minimum(np.maximum(img,0),1)
    
    return img
