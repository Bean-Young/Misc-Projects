with open("demo6.py", "r") as f:
    lines = f.readlines()

with open("demo6_new.py", "w") as f:
    for index, line in enumerate(lines):
        f.write(line.strip('\n').ljust(100) + "#"+str(index + 1) + "\n")
