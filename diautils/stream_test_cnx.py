from diautils.stream import *

if __name__ == '__main__':

    num_send = 1000 * 32 + 3
    num_loops = 20
    src = list(range(num_send))
    flow = BasicWorkerFlow()
    # flow = None

    def run_fetch(multiprocess, blocking, max_chunk_size, max_queue_size, num_fetch, ctx):
        print('[{}][{}][{}][MCS:{}][MQS:{}][fetch:{}]'.format('PROCESS' if multiprocess else 'THREAD', ctx if ctx else '', 'blocking' if blocking else 'non-blocking', max_chunk_size, max_queue_size, num_fetch))
        s = pipe(src).queue(max_chunk_size, max_queue_size, multiprocess, blocking, flow, ctx=ctx).stream()
        for _ in range(num_loops):
            with iter(s) as ite:
                # print('NUMBER OF THREADS: {}'.format(threading.active_count()))
                counter = help.SimpleCounter()
                # start = time.time()
                if num_fetch > 1:
                    ite.drain(out=counter.map_count_len, chunk_size=num_fetch)
                else:
                    ite.drain(out=counter.map_count)
                # end = time.time()
                assert counter.value == num_send
                # print('time: {}'.format(end - start))

    def run_max_queue_size(mp, blocking, max_chunk_size, max_queue_size, ctx):
        if max_chunk_size == 1 and max_queue_size > 0:
            max_queue_size *= 32
        run_fetch(mp, blocking, max_chunk_size, max_queue_size, 1, ctx)
        run_fetch(mp, blocking, max_chunk_size, max_queue_size, 8, ctx)
        run_fetch(mp, blocking, max_chunk_size, max_queue_size, 64, ctx)

    def run_max_chunk_size(mp, blocking, max_chunk_size, ctx):
        run_max_queue_size(mp, blocking, max_chunk_size, -1, ctx)
        run_max_queue_size(mp, blocking, max_chunk_size, 32, ctx)
        run_max_queue_size(mp, blocking, max_chunk_size, 128, ctx)

    def run_blocking(mp, blocking, ctx):
        run_max_chunk_size(mp, blocking, 1, ctx)
        run_max_chunk_size(mp, blocking, 4, ctx)
        run_max_chunk_size(mp, blocking, 16, ctx)
        run_max_chunk_size(mp, blocking, 64, ctx)

    def run(mp, ctx):
        run_blocking(mp, True, ctx)
        run_blocking(mp, False, ctx)

    run(False, None)
    run(True, 'fork')
    run(True, 'spawn')
