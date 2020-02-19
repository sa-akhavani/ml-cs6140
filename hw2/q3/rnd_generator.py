import random
new_path = 'random3.csv'
ff = open(new_path,'w')
for i in range (100):
    # line = '' + round(random.uniform(-1.00,1.00), 2) + ',' + round(random.uniform(-1.00,1.00), 2)
    line = f'{round(random.uniform(-1.00,1.00), 2)},{round(random.uniform(-10.00,40.00), 2)},{round(random.uniform(1.00,10.00), 2)},{round(random.uniform(-20.00,30.00), 2)},{round(random.uniform(0.00,1.00), 2)}'
    # print(line)
    ff.write(line)
    ff.write('\n')
    