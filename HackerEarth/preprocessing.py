import csv
import os
from shutil import copy2,move

label_file = "C:\\Users\\yashc\\Downloads\\hackerearth\\meta-data\\train.csv"
output_train_dir = "C:\\Users\\yashc\\Downloads\\hackerearth\\output-train"
output_test_dir = "C:\\Users\\yashc\\Downloads\\hackerearth\\output-test"
src_dir = "C:\\Users\\yashc\\Downloads\\hackerearth\\train"

def make_label_files(dir=output_train_dir,copy=True):
    label_set = set()
    c = 0
    with open(label_file) as fs:
        csvreader = csv.reader(fs)
        for row in csvreader:
            if c != 0:
                newPath = dir + "//" + row[1]
                if not os.path.exists(newPath):
                    os.makedirs(newPath)
                if copy:
                    copy2(src_dir + "//" + row[0] , newPath)
            c+=1
    print ("done")

def split_training_data():
    for subdirs,dirs,files in os.walk(output_train_dir):
        for name in dirs:
            newPath = output_test_dir + "//" + name
            c=0
            #print(output_train_dir + "//" + name)
            for subdirs,dirs,files in os.walk(output_train_dir + "//" + name):
                for filename in files:
                    c+=1
            print(c)
            c = int(c*0.25)
            cur = 0
            for subdirs,dirs,files in os.walk(output_train_dir + "//" + name):
                for filename in files:
                    if cur != c:
                        temp = 1
                        move(os.path.join(subdirs,filename),newPath)
                        #print("moving-> " + os.path.join(subdirs,filename) + " -> " + newPath)
                        cur += 1
                    else:
                        break
            #print("moved -> " + str(cur))


if __name__ == '__main__':
    #make_label_files(output_test_dir,False)
    split_training_data()
