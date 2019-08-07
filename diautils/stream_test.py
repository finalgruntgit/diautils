from unittest import TestCase
from diautils.stream import *

class TestStream(TestCase):

    def test_base(self):

        print('test_base')
        vals = list(range(100))
        s = pipe(vals).stream()
        assert(s.one() == 0)
        assert(s.many(5) == list(range(5)))
        assert (s.all() == vals)

    def test_iterate(self):

        print('test_iterate')
        c = help.SimpleCounter()
        total = 100
        num_loop = 20
        s = pipe(range(total)).stream()
        for _ in range(num_loop):
            # print(_)
            c.reset()
            for _ in s:
                c.incr()
            c.check(total)

    def test_drain(self):

        print('test_drain')
        c = help.SimpleCounter()
        total = 100
        num_loop = 20
        s = pipe(range(total)).stream()
        for _ in range(num_loop):
            c.reset()
            s.drain(c.map_count)
            c.check(total)

    def test_store(self):

        print('test_store')
        num_msg = 100
        msg = 'hello'
        qsize = 128
        num_loop = 20
        s = pipe([msg]).loop(num_msg).queue(qsize)
        for _ in range(num_loop):
            s.drain()

    def test_send_receive_parallel(self):
        c = help.SimpleCounter()
        msg = 'hello'
        num_msg = 1024
        num_worker = 3
        qsize = 1024
        num_loop = 4
        s = pipe(help.repeatable(msg, num_msg)).parallel(num_worker, max_chunk_size=8).reduce(max_chunk_size=4).queue(qsize, max_queue_size=2)
        for _ in range(num_loop):
            c.reset()
            s.drain(c.map_count_len, chunk_size=16)
            c.check(num_msg)

    def test_loop(self):

        print('test_loop')
        c = help.SimpleCounter()
        total = 25
        num_loop = 20
        num_repeat = 4
        s = pipe(range(total)).loop(num_repeat).map(c.map_count).stream()
        for _ in range(num_loop):
            c.reset()
            s.drain()
            c.check(total * num_repeat)

    def test_map(self):

        print('test_map')
        c = help.SimpleCounter()
        total = 20
        num_loop = 20
        vals = list(range(total))
        s = pipe(vals).map(lambda x: x+1).map(c.map_count).map(lambda x: x - 1).stream()
        for _ in range(num_loop):
            c.reset()
            assert s.all() == vals
            c.check(total)


    def test_each(self):

        print('test_each')
        c = help.SimpleCounter()
        total = 20
        num_repeat = 5
        num_loop = 10
        s = pipe([list(range(total))]).loop(num_repeat).each(c.map_count).stream()
        for _ in range(num_loop):
            c.reset()
            s.drain()
            c.check(total * num_repeat)

    def test_filter(self):

        print('test_filter')
        c = help.SimpleCounter()
        c2 = help.SimpleCounter()
        total = 20
        num_repeat = 5
        num_loop = 10
        s = pipe(range(total)).loop(num_repeat).map(c.map_count).filter(lambda x: x % 2 == 0).map(c2.map_count).stream()
        for _ in range(num_loop):
            c.reset()
            c2.reset()
            s.drain()
            c.check(total * num_repeat)
            c2.check(total * num_repeat // 2)

    def test_batch(self):

        print('test_batch')
        c = help.SimpleCounter()
        c2 = help.SimpleCounter()
        total = 20
        num_repeat = 5
        num_loop = 10
        batch_size = 4
        s = pipe(range(total)).loop(num_repeat).map(c.map_count).batch(batch_size).map(c2.map_count)
        for _ in range(num_loop):
            c.reset()
            c2.reset()
            s.drain()
            c.check(total * num_repeat)
            c2.check((total * num_repeat + batch_size - 1) // batch_size)

    def test_unpack(self):

        print('test_unpack')
        c = help.SimpleCounter()
        c2 = help.SimpleCounter()
        c3 = help.SimpleCounter()
        total = 20
        num_repeat = 5
        num_loop = 10
        batch_size = 4
        s = pipe(range(total)).loop(num_repeat).map(c.map_count).batch(batch_size).map(c2.map_count).unpack().map(c3.map_count).stream()
        for _ in range(num_loop):
            c.reset()
            c2.reset()
            c3.reset()
            s.drain()
            c.check(total * num_repeat)
            c2.check((total * num_repeat + batch_size - 1) // batch_size)
            c3.check(total * num_repeat)

    def test_limit(self):

        print('test_limit')
        c = help.SimpleCounter()
        c2 = help.SimpleCounter()
        total = 20
        num_repeat = 5
        num_loop = 10
        num_limit = 50
        s = pipe(range(total)).loop(num_repeat).map(c.map_count).limit(num_limit).map(c2.map_count).stream()
        for _ in range(num_loop):
            c.reset()
            c2.reset()
            s.drain()
            c.check(num_limit)
            c2.check(num_limit)

    def test_window(self):

        print('test_window')
        c = help.SimpleCounter()
        c2 = help.SimpleCounter()
        total = 20
        num_loop = 4
        window_size = 5
        vals = list(range(total))
        s = pipe(vals).window(window_size).map(c.map_count).map(c2.map_incr).stream()
        for _ in range(num_loop):
            c.reset()
            c2.reset()
            res = s.all()
            print(res)
            assert res != vals
            c.check(total)
            c2.check(sum(vals))

    def test_reshape(self):

        print('test_reshape')
        c = help.SimpleCounter()
        total = 20
        num_loop = 4
        vals = np.zeros(shape=(total, 4, 5)).tolist()
        s = pipe(vals).reshape((2, 5, 2)).stream()
        for _ in range(num_loop):
            c.reset()
            s.drain(c.map_count)
            c.check(total)

    def test_reshape_mp(self):

        print('test_reshape_mp')
        c = help.SimpleCounter()
        total = 20
        num_loop = 4
        vals = np.zeros(shape=(total, 4, 5))
        s = pipe(vals).parallel().reshape((2, 5, 2)).stream()
        for _ in range(num_loop):
            c.reset()
            s.drain(c.map_count)
            c.check(total)

    def test_transpose(self):

        print('test_transpose')
        vals = [[[1, 2, 3], [4, 5]]]
        s = pipe(vals).transpose(fill_value=-1).stream()
        print(s.all())

    def test_ndarray(self):

        print('test_ndarray')
        vals = [[[1, 2, 3], [4, 5, 6]]]
        s = pipe(vals).nd().stream()
        arr = s.one()
        print(arr, arr.shape, arr.dtype)