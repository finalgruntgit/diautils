from diautils.stream import *

num_send = 1024 * 1024
num_loops = 2
src = list(range(num_send))

if __name__ == '__main__':

    def run_fetch(multiprocess, blocking, max_chunk_size, max_queue_size, num_fetch, ctx):
        print('[{}][{}][{}][MCS:{}][MQS:{}][fetch:{}]'.format('PROCESS' if multiprocess else 'THREAD', ctx if ctx else '', 'blocking' if blocking else 'non-blocking', max_chunk_size, max_queue_size, num_fetch))
        s = pipe(src).queue(max_chunk_size, max_queue_size, multiprocess, blocking, ctx=ctx).stream()
        for _ in range(num_loops):
            with iter(s) as ite:
                timer = help.Timer().start()
                ite.drain(chunk_size=num_fetch)
                timer.stop()
                print('time: {}'.format(timer.time()))

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