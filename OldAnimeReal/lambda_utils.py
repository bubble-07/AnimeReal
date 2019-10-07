from itertools import repeat

#Shamelessly pulled from https://stackoverflow.com/questions/22102933/iterated-function-in-python  
#This obtains [x, func(x), func(func(x)), ...] with times+1 elements
def iterate(func, x, times):
    result = [x]
    result_append = result.append
    for _ in repeat(None, times):
        x = func(x)
        result_append(x)
    return result
