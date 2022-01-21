with open('input.wav', 'rb') as f:
    data = f.read()
dataH = ["{:02x}".format(c) for c in data[44:]]
samples = []
for i in range(0, len(dataH), 2):
    a = int(dataH[i+1]+dataH[i], 16)
    b = bin(a)
    if(len(bin(a)) == 18):
        a = -32768 + int(bin(a)[3:],2)
    else:
        print('')
    samples.append(a)

d = ''
