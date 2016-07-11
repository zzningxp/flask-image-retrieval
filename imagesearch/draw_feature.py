import image_retrieval 
import matplotlib.pylab as plt

def draw(name, feature4096):
    plt.figure(name)
    feature4096 = feature4096.reshape(64,64)
    for i, x in enumerate(feature4096):
        for j, y in enumerate(x):
            #if y <= 0:
            #    print '   ',
            #else:
            #    print '%3.1f'%y,
            plt.scatter([j],[i], s=[y*1000])
        #print
    plt.axis([-1, 65, -1, 65])
    plt.show()

if __name__ == '__main__':
    worddict = {}
    worddict['piano'] = 'n03452741'
    retri = image_retrieval.Retriever()
    drawset = []
    print retri.cate
    for i, x in enumerate(retri.cate):
        if x == worddict['piano']:
            drawset.append(i)

    for i in drawset[:2]:
        draw(retri.path[i], retri.queryDB[i])
