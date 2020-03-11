import math
def factorial(n):
    if n == 0:       
        return 1
    else:
        return (n * factorial(n-1))

def poisson(k):
    la=4
    return math.pow(la,k)/factorial(k)*pow(math.e,-la)
def gpoisson(k,left,right):
    sum=0
    for i in range(left,right):
        sum+=poisson(i)
    result=poisson(k)/sum
    return result
for i in range(1,6):
    print(i,end=":")
    print(gpoisson(i,1,5))
