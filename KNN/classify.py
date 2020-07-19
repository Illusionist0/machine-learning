from os import listdir
import numpy as np
import struct
import matplotlib.pyplot as plt
import time

#读取图片
def read_image(file_name):
    file_handle=open(file_name,"rb")                        
    file_content=file_handle.read()                        
    offset=0
    head = struct.unpack_from('>IIII', file_content, offset) 
    offset += struct.calcsize('>IIII')                      
    imgNum = head[1]                                         
    rows = head[2]                                      
    cols = head[3]                                          
    image_size = rows * cols                                 
    images=np.empty((imgNum , image_size))
    fmt='>' + str(image_size) + 'B'                          
    for i in range(imgNum):
        images[i] = np.array(struct.unpack_from(fmt, file_content, offset))
        offset += struct.calcsize(fmt)                    
    return images

#读取标签
def read_label(file_name):
    file_handle = open(file_name, "rb")                      
    file_content = file_handle.read()                        
    head = struct.unpack_from('>II', file_content, 0)    
    offset = struct.calcsize('>II')
    labelNum = head[1]                                       
    fmt = '>' + str(labelNum) + 'B'
    label = struct.unpack_from(fmt, file_content, offset)    
    return np.array(label)

#KNN算法
def KNN_classify(test_data, train_data, labels, k):
    distance1 = train_data - test_data                      
    distance2 = ((distance1**2).sum(axis=1))**0.5 
    sortedDistances = distance2.argsort()                    

    select_list = [0]*10                                     
    for i in range(k):
        votelabel = labels[sortedDistances[i]]               
        select_list[votelabel] += 1

    global output_count
    fig, ax = plt.subplots(nrows=k//10 +1, ncols=10, sharex='all', sharey='all')    
    ax = ax.flatten()
    field_index = 0
    for i in sortedDistances[:k]:                        
       near_image = np.array(train_data[i]).reshape(28, 28)
       ax[field_index].imshow(near_image, cmap='Greys', interpolation='nearest')
       field_index += 1                                        
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    output_count += 1
    plt.savefig('./test[i]_KNN_neighbour/test%d_%dNN_neighbour.jpg' % (output_count, k))
    plt.show()

    return select_list.index(max(select_list))             

def test_KNN_classify():
    time_start = time.time()
    # 文件获取
    train_image = "train-images.idx3-ubyte"
    test_image = "t10k-images.idx3-ubyte"
    train_label = "train-labels.idx1-ubyte"
    test_label = "t10k-labels.idx1-ubyte"
    # 读取数据
    train_x = read_image(train_image)
    test_x = read_image(test_image)
    train_y = read_label(train_label)
    test_y = read_label(test_label)
    testRatio = 0.01                                           
    test_row=test_x.shape[0]                               
    testNum = int(test_row * testRatio)                      

    k_set = [20]          # 存储的k池（可自己改变）
    misRate_set = []                                         

    global output_count

    for k in k_set:
        output_count = 0
        errorCount = 0                                       
        for i in range(testNum):
            result = KNN_classify(test_x[i], train_x, train_y, k)
            print('The classifier came back with: %d, the real answer is: %d' % (result, test_y[i]))
            if result != test_y[i]:
                errorCount += 1.0                            
        error_rate = errorCount / float(testNum)            
        accuracy = 1.0 - error_rate
        print("\nCurrent k value is: %d, the total number of tested is: %d" % (k, testNum))
        print("\nthe total number of error is: %d" % errorCount)
        print("\nthe total misclassification rate is: %f" % (error_rate))
        print("\nthe total accuracy rate is: %f" % (accuracy))
        misRate_set.append(error_rate)
        print("\n-------------------------------------------------------------\n")
    time_end = time.time()
    print('Finished %d times Testing, test %d images each time, time_cost is %ds in all.'
          %(len(k_set), testNum, time_end-time_start))
    l1 = plt.plot(k_set, misRate_set, label='misclassification rate', linewidth=2, color='b', marker='o',
         markerfacecolor='red', markersize=5)
    plt.title('The misclassification rate of different k (10000 images)')           
    plt.xlabel('k value')
    plt.ylabel('misclassification rate')
    plt.xlim(0, max(k_set)+1)
    plt.ylim(0, max(misRate_set)+0.01)
    for a, b in zip(k_set, misRate_set):
        plt.text(a, b+0.000007, '%.3f' % b, ha='center', va='bottom', fontsize=6)   
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_KNN_classify()