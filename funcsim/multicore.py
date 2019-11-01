from __future__ import print_function
import multiprocessing
import sys

if (sys.version_info > (3, 0)):
    import queue
else:
    import Queue as queue


def fun(f, q_in, q_out):
    try:
        while True:
            # print("%s : seeking work" % os.getpid())
            i, x = q_in.get(True, 1)
            try:
                ret = f(x)
                q_out.put((i, ret), True)
            except Exception as e:
                msg = "WARNING: function passed to multi.parmap "
                msg += "encountered an exception: "
                print("%s\n%s" (msg, e), file=sys.stderr)
                sys.stderr.flush()
                q_out.put((i, None), True)
    except queue.Empty:
        sys.stdout.flush()
        sys.exit(0)


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    manager = multiprocessing.Manager()

    # set up and fill work queue
    q_in = manager.Queue()
    sent = [q_in.put((i, x), True) for i, x in enumerate(X)]

    q_out = manager.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out)) for
            _ in range(nprocs)]

    for p in proc:
        p.daemon = True
        p.start()

    for p in proc:
        if p.is_alive():
            sys.stdout.flush()
            p.join()

    res = []
    try:
        while True:
            res.append(q_out.get(False))
    except queue.Empty:
        pass

    lost = len(sent) - len(res)
    if lost > 0:
        print("WARNING: multi.parmap encountered %s lost jobs" % lost,
              file=sys.stderr)
        sys.stderr.flush()

    problems = sum([1 for i, x, in res if x is None])
    if problems > 0:
        print("WARNING: multi.parmap encountered %s failed jobs" % problems,
              file=sys.stderr)
        sys.stderr.flush()

    ret = [x for i, x in sorted(res)]
    manager.shutdown()
    return ret
