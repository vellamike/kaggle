import io
import pdb
def make_sub():
    with open("SampleSubmission.csv", 'r') as file:
        data = file.readlines()
    for (i, line) in enumerate(data):
        if i % 100 == 0:
            print(i)
        part = line.split(",")
        ''' part[0] id
            part[1] {H1, ..., H4}
            part[2] device id
            part[3] timestep
            part[4] prediction '''
        if part[1] == "H4":
            if part[2] == "28":
                stamp = int(part[3])
                # PC on test day Sep 12. See H4 observations sheet.
                if 1347510000 <= stamp <= 1347540000:
                    data[i] = "{0},{1},{2},{3},{4}".format(part[0], part[1],
                                                           part[2], part[3],
                                                           1)
                    pdb.set_trace()
    with open('submission.csv','w') as file:
        file.writelines(data)

if __name__ == "__main__":
    make_sub()
