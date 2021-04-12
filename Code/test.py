

with open("./requirements.txt", 'r') as f:
    lol = f.read().splitlines()
    lol = ["lol" + element + "hey" for element in lol]



print(lol)



