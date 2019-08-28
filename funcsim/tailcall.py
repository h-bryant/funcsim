# credit: http://www.kylem.net/programming/tailcall.html


class TailCaller(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        ret = self.f(*args, **kwargs)
        while type(ret) is TailCall:
            ret = ret.handle()
        return ret


class TailCall(object):
    def __init__(self, call, *args, **kwargs):
        self.__call__ = call
        self.args = args
        self.kwargs = kwargs

    def handle(self):
        if type(self.__call__) is TailCaller:
            return self.__call__.f(*self.args, **self.kwargs)
        else:
            return self.__call__(*self.args, **self.kwargs)


# useage example:
@TailCaller
def fact(n, r=1):
    if n <= 1:
        return r
    else:
        return TailCall(fact, n-1, n*r)