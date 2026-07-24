import multiprocessing
import queue
import sys


class _JobError:
    # picklable marker carrying a worker exception message back to the
    # parent process
    def __init__(self, msg):
        self.msg = msg


def fun(f, q_in, q_out):
    try:
        while True:
            # print("%s : seeking work" % os.getpid())
            i, x = q_in.get(True, 1)
            try:
                ret = f(x)
                q_out.put((i, ret), True)
            except Exception as e:
                # report the failure to the parent, which will raise;
                # swallowing it here would silently corrupt the results
                q_out.put((i, _JobError(f"{type(e).__name__}: {e}")), True)
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

    manager.shutdown()

    # any failed or lost job invalidates the whole result set; raise (as
    # single-process mode would) rather than passing None values through
    errors = [x for _, x in res if isinstance(x, _JobError)]
    if errors:
        raise RuntimeError(f"multicore.parmap: {len(errors)} of {len(sent)} "
                           f"jobs raised an exception; first error: "
                           f"{errors[0].msg}")
    lost = len(sent) - len(res)
    if lost > 0:
        raise RuntimeError(f"multicore.parmap: {lost} of {len(sent)} jobs "
                           f"were lost (worker process died?)")

    return [x for i, x in sorted(res)]
