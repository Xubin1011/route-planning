from test import test1

env = test1()

env.a = 2
print(env.a)
print(env.b)
env.com(env.a, env.b)