import time

def RateLimited(maxPerSecond):

    """
    Decorator function to limit the rate at which a function is called.    
    :param maxPerSecond: The number of function calls a second.
    :return: None
    """

    minInterval = 1.0 / float(maxPerSecond)

    def decorate(func):
        lastTimeCalled = [0.0]

        def rateLimitedFunction(*args, **kargs):
            elapsed = time.clock() - lastTimeCalled[0]
            leftToWait = minInterval - elapsed
            if leftToWait > 0:
                time.sleep(leftToWait)
            ret = func(*args, **kargs)
            lastTimeCalled[0] = time.clock()
            return ret
        return rateLimitedFunction
    return decorate


if __name__ == "__main__":

    @RateLimited(2)  # call per second
    def printNumber(num):
        print("num: " + str(num))

    for i in range(1, 100):
        printNumber(i)
