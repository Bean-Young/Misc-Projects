with open("demo6.py", "r") as f:                                                                    #1
    lines = f.readlines()                                                                           #2
                                                                                                    #3
with open("demo6_new.py", "w") as f:                                                                #4
    for index, line in enumerate(lines):                                                            #5
        f.write(line.strip('\n').ljust(100) + "#"+str(index + 1) + "\n")                            #6
