
import random

def rand_test(seed):
    random.seed(seed)
    negitem = random.choice(range(100))
    print(negitem)
    random.seed(2020)

rand_test(2020)

negitem = random.choice(range(100))
print(negitem)


rand_test(1010)

negitem = random.choice(range(100))
print(negitem)

rand_test(2020)

negitem = random.choice(range(100))
print(negitem)

rand_test(1)

negitem = random.choice(range(100))
print(negitem)


