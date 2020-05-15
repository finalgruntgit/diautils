import collections as cols
import itertools
import numpy as np
import multiprocessing as mp
import threading
import time
from diautils import help


def pipe(data):
    if type(data) in (int, np.int, np.uint, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64):
        return StreamNode(IterablePipe(np.arange(data)))
    elif callable(data):
        return StreamNode(CallableDataPipe(data))
    elif help.is_arraylike(data) and not isinstance(data, range):
        return StreamNode(ArrayPipe(data))
    elif help.is_iterable(data):
        return StreamNode(IterablePipe(data))
    else:
        raise Exception('Cannot create stream data pipe from this type of data source')


class WorkerData:

    def __init__(self, segid, multiprocess=False, supervised=True, wid=0):
        self.segid = segid
        self.multiprocess = multiprocess
        self.supervised = supervised
        self.id = wid


class WorkerFlow:

    def on_start(self, w):
        pass

    def on_stop(self, w, reason):
        pass

    def on_error(self, w, err):
        pass


class BasicWorkerFlow(WorkerFlow):

    def __init__(self, log=True):
        self.log = log

    def on_start(self, w):
        if self.log:
            print('[{}][{}][{}][{}] start'.format(help.getpid(), 'P' if w.multiprocess else 'T', w.segid, 'supervis' if w.supervised else 'parallel-{}'.format(w.id)))

    def on_stop(self, w, reason):
        if self.log:
            print('[{}][{}][{}][{}] stop ({})'.format(help.getpid(), 'P' if w.multiprocess else 'T', w.segid, 'supervis' if w.supervised else 'parallel-{}'.format(w.id), reason))

    def on_error(self, w, err):
        if self.log:
            print('[{}][{}][{}][{}] error'.format(help.getpid(), 'P' if w.multiprocess else 'T', w.segid, 'supervis' if w.supervised else 'parallel-{}'.format(w.id)))
            print(help.format_error_ml(err))


class WorkerStopReason(help.NamedEnum):
    END_OF_PIPE = 0,
    INTERRUPTED = 1,
    ERROR = 2,


def worker_run(task, wdata, flows):
    if flows:
        if help.is_arraylike(flows):
            for flow in flows:
                flow.on_start(wdata)
        else:
            flows.on_start(wdata)
    stop_reason = WorkerStopReason.END_OF_PIPE
    ex = None
    while True:
        try:
            task.run()
        except EmptyPipeException:
            time.sleep(1e-4)
        except StopIteration:
            break
        except KeyboardInterrupt:
            stop_reason = WorkerStopReason.INTERRUPTED
            break
        except Exception as err:
            if flows:
                if help.is_arraylike(flows):
                    ahead = True
                    for flow in flows:
                        ahead &= flow.on_error(wdata, err)
                    if ahead:
                        continue
                elif flows.on_error(wdata, err):
                    continue
            stop_reason = WorkerStopReason.ERROR
            ex = err
            break
    if flows:
        if help.is_arraylike(flows):
            for flow in flows:
                flow.on_stop(wdata, stop_reason)
        else:
            flows.on_stop(wdata, stop_reason)
    if ex is not None:
        raise ex


class PipeIterator:

    def __next__(self):
        return self.one()

    def __iter__(self):
        return self

    def one(self):
        raise NotImplementedError()

    def many(self, num):
        out = []
        try:
            while len(out) < num:
                out.append(self.one())
        except StopIteration:
            if not len(out):
                raise
        return out

    def set_main(self):
        return self


class ChildPipeIterator(PipeIterator):

    def __init__(self, ite):
        self.ite = ite

    def set_main(self):
        self.ite.set_main()
        return self


class PipeTask(ChildPipeIterator):

    def __init__(self, ite):
        super().__init__(ite)

    def run(self):
        raise NotImplementedError()


class PipeLevel(help.NamedEnum):
    MAIN = 0,
    SUPERVISED = 1,
    PARALLEL = 2


class Pipe:

    def __init__(self, segid=0, level=PipeLevel.MAIN, is_link=False):
        self.level = level
        self.segid = segid
        self.is_link = is_link

    def create(self, input):
        raise NotImplementedError()

    def create_link(self, input=None):
        return self.create(input), [], input


class ChildPipe(Pipe):

    def __init__(self, parent):
        super().__init__(parent.segid, parent.level, False)
        self.parent = parent


class LinkPipe(Pipe):

    def __init__(self, segid, level=PipeLevel.SUPERVISED):
        super().__init__(segid, level, True)


###########
# FILTERS #
###########

def filter_empty(v):
    return v


###########
# MAPPERS #
###########

class ReshapeMapper:

    def __init__(self, shape):
        self.shape = shape

    def apply(self, v):
        return np.reshape(v, self.shape)


class TransposeMapper:

    def __init__(self, fill_value):
        self.fill_value = fill_value

    def apply(self, v):
        return list(itertools.zip_longest(*v, fillvalue=self.fill_value))


class ColumnStackMapper:

    def __init__(self, columns):
        self.columns = np.zeros(max(columns), bool)
        self.columns[columns] = True

    def apply(self, v):
        out = []
        for i, c in enumerate(v):
            if self.columns[i]:
                out.append(help.flatten(c))
            else:
                out.append(c)
        return out


def map_ndarray(v):
    return np.array(v)


class StreamNode:

    def __init__(self, pipe, parent=None):
        self.parent = parent
        self.pipe = pipe

    # Pipeline compiler
    def stream(self):
        return Stream(self)

    # Quick accessors
    def __iter__(self):
        return iter(self.stream())

    def one(self):
        return self.stream().one()

    def many(self, num):
        return self.stream().many(num)

    def all(self, chunk_size=1):
        return self.stream().all(chunk_size)

    def drain(self, out=None, chunk_size=1):
        return self.stream().drain(out, chunk_size)

    # Generics
    def attach(self, processor):
        return StreamNode(processor, self)

    # Links
    def queue(self, max_chunk_size=1, max_queue_size=-1, multiprocess=False, blocking=False, flow=None, clear_on_interrupt=True, ctx='spawn'):
        if self.pipe.level == PipeLevel.PARALLEL:
            raise Exception('Storing only allowed on a main or supervised stream segment')
        node = self.attach(InputQueueLinkPipe(self.pipe, max_chunk_size, max_queue_size, multiprocess, blocking, flow, clear_on_interrupt, ctx)).attach(OutputQueuePipe(self.pipe.segid + 1))
        if max_chunk_size > 1:
            node = node.unpack()
        return node

    def parallel(self, num=1, max_chunk_size=1, max_queue_size=-1, fair=True, multiprocess=True, blocking=True, flow=None, clear_on_interrupt=True, ctx='spawn'):
        if not self.pipe.level == PipeLevel.MAIN:
            raise Exception('Storing only allowed on a main stream segment')
        node = self.attach(InputParallelLinkPipe(self.pipe, num, max_chunk_size, max_queue_size, fair, multiprocess, blocking, flow, clear_on_interrupt, ctx)).attach(OutputParallelPipe(self.pipe.segid + 1))
        if max_chunk_size == 1:
            return node
        else:
            return node.unpack()

    def reduce(self, max_chunk_size=1, max_queue_size=-1, fair=True, blocking=False, flow=None, clear_on_interrupt=True):
        node = self.attach(InputReduceLinkPipe(self.pipe, max_chunk_size, max_queue_size, flow, clear_on_interrupt)).attach(OutputReducePipe(self.pipe.segid + 1, fair, blocking))
        if max_chunk_size == 1:
            return node
        else:
            return node.unpack()

    # Basics
    def map(self, fc_map):
        if fc_map is None:
            return self
        return self.attach(MapPipe(self.pipe, fc_map))

    def each(self, fc_map):
        if fc_map is None:
            return self
        return self.attach(EachPipe(self.pipe, fc_map))

    def yielder(self, class_yielder):
        if class_yielder is None:
            return self
        return self.attach(YielderPipe(self.pipe, class_yielder))

    def batch(self, num):
        assert num > 0
        return self.attach(BatchPipe(self.pipe, num))

    def filter(self, fc_filter=filter_empty):
        if fc_filter is None:
            return self
        return self.attach(FilterPipe(self.pipe, fc_filter))

    def loop(self, num=-1):
        if self.pipe.segid > 0:
            raise Exception('Looping only supported inside a primary stream segment')
        return self.attach(LoopPipe(self.pipe, num))

    def unpack(self):
        return self.attach(UnpackPipe(self.pipe))

    def limit(self, num):
        if num > 0:
            return self.attach(LimitPipe(self.pipe, num))
        else:
            return self

    # Combines
    def shuffle(self):
        return self.map(help.randomize)

    def window(self, size):
        return self.batch(size).shuffle().unpack()

    def reshape(self, shape):
        return self.map(ReshapeMapper(shape).apply)

    def transpose(self, fill_value=None):
        return self.map(TransposeMapper(fill_value).apply)

    def stack(self, columns=None):
        if columns is None:
            return self.map(help.stack)
        else:
            return self.map(ColumnStackMapper(columns).apply)

    def nd(self):
        return self.map(map_ndarray)

    def map_filter(self, fc_map):
        if fc_map is None:
            return self
        return self.map(fc_map).filter()


class StreamIterator(PipeIterator):

    def __init__(self, stream):
        assert len(stream.segments)
        ctx = None
        self.stores = []
        for segment in stream.segments:
            self.task, stores, ctx = segment.create_link(ctx)
            self.stores.extend(stores)
        self.task.set_main()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.interrupt()

    def one(self):
        while True:
            try:
                return self.task.one()
            except EmptyPipeException:
                continue
            except KeyboardInterrupt:
                self.interrupt()
                raise

    def many(self, num):
        while True:
            try:
                return self.task.many(num)
            except EmptyPipeException:
                continue
            except KeyboardInterrupt:
                self.interrupt()
                raise

    def all(self, chunk_size=1):
        try:
            if chunk_size == 1:
                return list(self)
            else:
                out = []
                try:
                    while True:
                        out.extend(self.many(chunk_size))
                except StopIteration:
                    if not len(out):
                        raise
                return out
        except KeyboardInterrupt:
            self.interrupt()
            raise

    def drain(self, out=None, chunk_size=1):
        try:
            if out is None:
                if chunk_size == 1:
                    try:
                        while True:
                            self.one()
                    except StopIteration:
                        pass
                else:
                    try:
                        while True:
                            self.many(chunk_size)
                    except StopIteration:
                        pass
            else:
                if chunk_size == 1:
                    try:
                        while True:
                            out(self.one())
                    except StopIteration:
                        pass
                else:
                    try:
                        while True:
                            out(self.many(chunk_size))
                    except StopIteration:
                        pass
        except KeyboardInterrupt:
            self.interrupt()
            raise

    def interrupt(self):
        if len(self.stores):
            for store in self.stores:
                store.interrupt()
            self.stores = []
        return self


class Stream:

    def __init__(self, node):
        if node.pipe.level == PipeLevel.PARALLEL:
            node = node.reduce()
        if node.pipe.level == PipeLevel.SUPERVISED:
            node = node.queue()
        self.segments = [node.pipe]
        while node.parent is not None:
            node = node.parent
            if node.pipe.is_link:
                self.segments.append(node.pipe)
        self.segments.reverse()

    def one(self):
        with iter(self) as ite:
            return ite.one()

    def many(self, num):
        with iter(self) as ite:
            return ite.many(num)

    def all(self, chunk_size=1):
        with iter(self) as ite:
            return ite.all(chunk_size)

    def drain(self, out=None, chunk_size=1):
        with iter(self) as ite:
            ite.drain(out, chunk_size)
        return self

    def __iter__(self):
        return StreamIterator(self)


###############
# CONNECTIONS #
###############

class EmptyPipeException(StopIteration):
    pass


class FullPipeException(StopIteration):
    pass


class ClosedPipeException(StopIteration):
    pass


class InputCnx:

    def send(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class OutputCnx(PipeIterator):

    def __init__(self, blocking=False):
        self.blocking = blocking

    def set_main(self):
        self.blocking = True
        return self


class MtStore:

    def __init__(self, clear_on_interrupt=True):
        self.open = help.SharedMtBool(True)
        self.interrupted = help.SharedMtBool(False)
        self.queue = cols.deque()
        self.clear_on_interrupt = clear_on_interrupt

    def close(self):
        self.open.set(False)
        return self

    def interrupt(self):
        self.interrupted.set()
        if self.clear_on_interrupt:
            self.queue.clear()
        return self.close()


class MtInputCnx(InputCnx):

    def __init__(self, store, max_queue_size):
        self.store = store
        self.max_queue_size = max_queue_size
        self.send = self.send_capped if self.max_queue_size > 0 else self.send_uncapped
        self.count = 0

    def check(self):
        if self.store.open.is_set():
            return self
        if self.store.interrupted.is_set():
            raise KeyboardInterrupt()
        raise StopIteration()

    def send_capped(self, v):
        self.check()
        if len(self.store.queue) >= self.max_queue_size:
            raise FullPipeException()
        self.store.queue.append(v)
        return v

    def send_uncapped(self, v):
        self.check().store.queue.append(v)
        return v

    def close(self):
        self.store.close()
        return self


class MtOutputCnx(OutputCnx):

    def __init__(self, store, blocking=False):
        super().__init__(blocking)
        self.store = store

    def one(self):
        while True:
            try:
                return self.store.queue.popleft()
            except IndexError:
                if self.store.open.is_set():
                    if self.blocking:
                        time.sleep(1e-4)
                    else:
                        raise EmptyPipeException()
                else:
                    if self.store.interrupted.is_set():
                        raise KeyboardInterrupt()
                    elif not len(self.store.queue):
                        raise StopIteration()


def create_mt_cnx(max_queue_size=-1, blocking=False, clear_on_interrupt=True):
    store = MtStore(clear_on_interrupt)
    return MtInputCnx(store, max_queue_size), MtOutputCnx(store, blocking)


class MpStore:

    def __init__(self, ctx=mp.get_context('spawn'), clear_on_interrupt=True):
        self.open = help.SharedMpBool(True, ctx=ctx)
        self.interrupted = help.SharedMpBool(False, ctx=ctx)
        self.counter = help.SharedMpCounter(ctx=ctx)
        self.clear_on_interrupt = clear_on_interrupt

    def close(self):
        self.open.set(False)
        return self

    def interrupt(self):
        self.interrupted.set()
        if self.clear_on_interrupt:
            self.counter.set(0)
        return self.close()


class MpInputCnx(InputCnx):

    def __init__(self, cnx, store, max_queue_size):
        self.cnx = cnx
        self.store = store
        self.max_queue_size = max_queue_size
        self.send = self.send_capped if self.max_queue_size > 0 else self.send_uncapped

    def check(self):
        if self.store.open.is_set():
            return self
        if self.store.interrupted.is_set():
            raise KeyboardInterrupt()
        raise StopIteration()

    def send_capped(self, v):
        self.check()
        if self.store.counter.value >= self.max_queue_size:
            raise FullPipeException()
        self.store.counter.incr()
        self.cnx.send(v)
        return v

    def send_uncapped(self, v):
        self.check()
        self.store.counter.incr()
        self.cnx.send(v)
        return v

    def close(self, owner=True):
        if owner:
            self.store.close()
        self.cnx.close()
        return self


class MpOutputCnx(OutputCnx):

    def __init__(self, cnx, store, blocking=False):
        super().__init__(blocking)
        self.cnx = cnx
        self.store = store

    def one(self):
        try:
            if self.blocking or self.store.counter.value:
                self.store.counter.decr()
                v = self.cnx.recv()
                return v
            if not self.store.open.is_set() and not self.store.counter.value:
                raise StopIteration()
            raise EmptyPipeException()
        except EOFError:
            raise StopIteration()

    def close(self):
        self.cnx.close()
        return self


def create_mp_cnx(max_queue_size=-1, blocking=False, clear_on_interrupt=True, ctx=mp.get_context('spawn')):
    store = MpStore(ctx, clear_on_interrupt)
    cnx_out, cnx_in = ctx.Pipe(duplex=False)
    return MpInputCnx(cnx_in, store, max_queue_size), MpOutputCnx(cnx_out, store, blocking)


class CnxDrainPipeTask(PipeTask):

    def __init__(self, ite, cnx, max_chunk_size):
        super().__init__(ite)
        self.cnx = cnx
        self.max_chunk_size = max_chunk_size
        self.run = self.drain_one if self.max_chunk_size == 1 else self.drain_many

    def drain_one(self):
        try:
            v = self.ite.one()
        except EmptyPipeException:
            raise
        except StopIteration:
            self.cnx.close()
            raise
        while True:
            try:
                self.cnx.send(v)
                break
            except FullPipeException:
                help.sleep(1e-4)

    def drain_many(self):
        try:
            vs = self.ite.many(self.max_chunk_size)
        except EmptyPipeException:
            raise
        except StopIteration:
            self.cnx.close()
            raise
        while True:
            try:
                self.cnx.send(vs)
                break
            except FullPipeException:
                help.sleep(1e-4)


############
# CALLABLE #
############

class DeferredPipeIterator(PipeIterator):

    def __init__(self, fc_data):
        self.fc_data = fc_data
        self.ite = None

    def check(self):
        if self.ite is None:
            data = self.fc_data()
            if help.is_arraylike(data) and not isinstance(data, range):
                self.ite = ArrayPipeIterator(data)
            elif help.is_iterable(data):
                self.ite = IterablePipeIterator(iter(data))
            elif help.is_iterator(data):
                self.ite = IterablePipeIterator(data)
            else:
                raise Exception('Cannot stream data from this type of data source')
        return self.ite

    def many(self, num):
        return self.check().many(num)

    def one(self):
        return self.check().one()


class CallableDataPipe(Pipe):

    def __init__(self, fc_data):
        super().__init__()
        self.fc_data = fc_data

    def create(self, _):
        return DeferredPipeIterator(self.fc_data)


############
# ITERABLE #
############

class IterablePipeIterator(PipeIterator):

    def __init__(self, ite):
        self.ite = ite

    def one(self):
        return next(self.ite)


class IterablePipe(Pipe):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def create(self, _):
        return IterablePipeIterator(iter(self.data))


#########
# ARRAY #
#########

class ArrayPipeIterator(PipeIterator):

    def __init__(self, data):
        self.data = data
        self.cursor = 0

    def one(self):
        try:
            v = self.data[self.cursor]
            self.cursor += 1
            return v
        except IndexError:
            raise StopIteration

    def many(self, num):
        end = self.cursor + num
        vs = self.data[self.cursor:end]
        if len(vs):
            self.cursor = end
            return vs
        else:
            raise StopIteration


class ArrayPipe(Pipe):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def create(self, _):
        return ArrayPipeIterator(self.data)


###########
# YIELDER #
###########

class BasicYielder:

    def __init__(self):
        self.queue = cols.deque()

    def fill(self):
        raise NotImplementedError

    def extract(self, v):
        raise NotImplementedError


class YielderPipeIterator(ChildPipeIterator):

    def __init__(self, ite, yielder):
        super().__init__(ite)
        self.yielder = yielder

    def one(self):
        while True:
            try:
                return self.yielder.queue.popleft()
            except IndexError:
                try:
                    self.yielder.fill()
                except StopIteration:
                    self.yielder.extract(self.ite.one())

    def many(self, num):
        out = []
        while len(out) < num:
            try:
                out.append(self.one())
            except StopIteration:
                if not len(out):
                    raise
                else:
                    break
        return out


class YielderPipe(ChildPipe):

    def __init__(self, parent, class_yielder):
        super().__init__(parent)
        self.class_yielder = class_yielder

    def create(self, input):
        return YielderPipeIterator(self.parent.create(input), self.class_yielder())


#######
# MAP #
#######

class MapPipeIterator(ChildPipeIterator):

    def __init__(self, ite, fc_map):
        super().__init__(ite)
        self.fc_map = fc_map

    def one(self):
        return self.fc_map(self.ite.one())

    def many(self, num):
        return [self.fc_map(v) for v in self.ite.many(num)]


class MapPipe(ChildPipe):

    def __init__(self, parent, fc_map):
        super().__init__(parent)
        self.fc_map = fc_map

    def create(self, input):
        return MapPipeIterator(self.parent.create(input), self.fc_map)


##########
# FILTER #
##########

class FilterPipeIterator(ChildPipeIterator):

    def __init__(self, ite, fc_filter):
        super().__init__(ite)
        self.fc_filter = fc_filter

    def one(self):
        while True:
            v = self.ite.one()
            if self.fc_filter(v):
                return v

    def many(self, num):
        out = []
        try:
            while len(out) < num:
                out.extend([v for v in self.ite.many(num - len(out)) if self.fc_filter(v)])
        except StopIteration:
            if not len(out):
                raise
        return out


class FilterPipe(ChildPipe):

    def __init__(self, parent, fc_filter):
        super().__init__(parent)
        self.fc_filter = fc_filter

    def create(self, input):
        return FilterPipeIterator(self.parent.create(input), self.fc_filter)


#########
# BATCH #
#########

class BatchPipeIterator(ChildPipeIterator):

    def __init__(self, ite, num):
        super().__init__(ite)
        self.num = num
        self.buffer = []

    def one(self):
        if len(self.buffer):
            vs = self.buffer
            try:
                vs.extend(self.ite.many(self.num - len(self.buffer)))
            except EmptyPipeException:
                pass
            except StopIteration:
                self.buffer = []
                return vs
            if len(vs) == self.num:
                self.buffer = []
                return vs
            raise EmptyPipeException()
        else:
            vs = self.ite.many(self.num)
            if len(vs) == self.num:
                return vs
            if isinstance(vs, np.ndarray):
                self.buffer = vs.tolist()
            else:
                self.buffer = vs
            raise EmptyPipeException()


class BatchPipe(ChildPipe):

    def __init__(self, parent, num):
        super().__init__(parent)
        self.num = num

    def create(self, input):
        return BatchPipeIterator(self.parent.create(input), self.num)


########
# LOOP #
########

class LoopPipeIterator(PipeIterator):

    def __init__(self, parent, ctx, num):
        self.parent = parent
        self.ctx = ctx
        self.num = num
        self.count = 1
        self.ite = self.parent.create(self.ctx)

    def next_loop(self):
        if self.num > 0:
            if self.count == self.num:
                self.ite = None
                raise StopIteration
            self.count += 1
        self.ite = self.parent.create(self.ctx)
        return self

    def one(self):
        while True:
            try:
                return self.ite.one()
            except EmptyPipeException:
                raise
            except StopIteration:
                self.next_loop()
            except AttributeError:
                raise StopIteration

    def many(self, num):
        out = []
        while len(out) < num:
            try:
                out.extend(self.ite.many(num - len(out)))
            except EmptyPipeException:
                if not len(out):
                    raise
                else:
                    break
            except StopIteration:
                try:
                    self.next_loop()
                except StopIteration:
                    if not len(out):
                        raise
                    else:
                        break
            except AttributeError:
                raise StopIteration
        return out

    def interrupt(self):
        if self.ite is not None:
            self.ite.interrupt()
        return self


class LoopPipe(ChildPipe):

    def __init__(self, parent, num):
        super().__init__(parent)
        self.num = num

    def create(self, input):
        return LoopPipeIterator(self.parent, input, self.num)


########
# EACH #
########

class EachPipeIterator(ChildPipeIterator):

    def __init__(self, ite, fc_map):
        super().__init__(ite)
        self.fc_map = fc_map

    def apply(self, v):
        return list(map(self.fc_map, v))

    def one(self):
        return self.apply(self.ite.one())

    def many(self, num):
        return [self.apply(v) for v in self.ite.many(num)]


class EachPipe(ChildPipe):

    def __init__(self, parent, fc_map):
        super().__init__(parent)
        self.fc_map = fc_map

    def create(self, input):
        return EachPipeIterator(self.parent.create(input), self.fc_map)


###########
# UNPACK  #
###########

class UnpackPipeIterator(ChildPipeIterator):

    def __init__(self, ite):
        super().__init__(ite)
        self.buffer = []
        self.buffer_index = 0

    def one(self):
        try:
            v = self.buffer[self.buffer_index]
            self.buffer_index += 1
            return v
        except IndexError:
            self.buffer = self.ite.one()
            self.buffer_index = 1
            return self.buffer[0]

    def many(self, num):
        while True:
            offset = self.buffer_index
            self.buffer_index += num
            out = self.buffer[offset:self.buffer_index]
            while len(out) < num:
                try:
                    self.buffer = self.ite.one()
                    self.buffer_index = num - len(out)
                    out.extend(self.buffer[:self.buffer_index])
                except StopIteration:
                    if not len(out):
                        raise
                    else:
                        break
            return out


class UnpackPipe(ChildPipe):

    def __init__(self, parent):
        super().__init__(parent)

    def create(self, input):
        return UnpackPipeIterator(self.parent.create(input))


#########
# LIMIT #
#########

class LimitPipeIterator(ChildPipeIterator):

    def __init__(self, ite, num):
        super().__init__(ite)
        self.num = num
        self.cursor = 0

    def check(self):
        if self.cursor == self.num:
            raise StopIteration
        return self

    def one(self):
        v = self.check().ite.one()
        self.cursor += 1
        return v

    def many(self, num):
        vs = self.check().ite.many(min(num, self.num - self.cursor))
        self.cursor += len(vs)
        return vs


class LimitPipe(ChildPipe):

    def __init__(self, parent, num):
        super().__init__(parent)
        self.num = num

    def create(self, input):
        return LimitPipeIterator(self.parent.create(input), self.num)


#########
# QUEUE #
#########

class InputQueueLinkPipe(LinkPipe):

    def __init__(self, parent, max_chunk_size=1, max_queue_size=-1, multiprocess=False, blocking=False, flow=None, clear_on_interrupt=True, ctx='spawn'):
        super().__init__(parent.segid, PipeLevel.PARALLEL if multiprocess else PipeLevel.SUPERVISED)
        self.parent = parent
        self.max_chunk_size = max_chunk_size
        self.max_queue_size = max_queue_size
        self.multiprocess = multiprocess
        self.blocking = blocking
        self.flow = flow
        self.clear_on_interrupt = clear_on_interrupt
        self.ctx = ctx

    def create_link(self, input=None):
        if self.multiprocess:
            ctx = mp.get_context(self.ctx)
            cnx_in, cnx_out = create_mp_cnx(self.max_queue_size, self.blocking, self.clear_on_interrupt, ctx)
            task = CnxDrainPipeTask(self.parent.create(input), cnx_in, self.max_chunk_size)
            worker = ctx.Process(target=worker_run, args=(task, WorkerData(self.segid, True, True), self.flow))
            worker.start()
            if input is not None:
                for cnx in input:
                    cnx.close()
            cnx_in.close(False)
        else:
            cnx_in, cnx_out = create_mt_cnx(self.max_queue_size, self.blocking, self.clear_on_interrupt)
            task = CnxDrainPipeTask(self.parent.create(input), cnx_in, self.max_chunk_size)
            worker = threading.Thread(target=worker_run, args=(task, WorkerData(self.segid, False, True), self.flow))
            worker.start()
        return None, [cnx_in.store], [cnx_out]


class OutputQueuePipe(Pipe):

    def __init__(self, segid):
        super().__init__(segid)

    def create(self, input):
        return input[0]


############
# PARALLEL #
############

class ParallelPipeTask(PipeTask):

    def __init__(self, ite, cnxs, max_chunk_size, fair=True):
        super().__init__(ite)
        self.cnxs = cnxs
        self.max_chunk_size = max_chunk_size
        self.fair = fair
        self.cursor = 0
        self.fetch = self.fetch_one if self.max_chunk_size == 1 else self.fetch_many

    def close(self):
        for cnx in self.cnxs:
            cnx.close()
        self.cnxs.clear()
        return self

    def adv(self):
        self.cursor += 1
        if self.cursor == len(self.cnxs):
            self.cursor = 0
        return self

    def run(self):
        v = self.fetch()
        offset = self.cursor
        while True:
            try:
                self.cnxs[self.cursor].send(v)
                if self.fair:
                    self.adv()
                break
            except FullPipeException:
                self.adv()
                if self.cursor == offset:
                    help.sleep(1e-4)
        return self

    def fetch_one(self):
        try:
            return self.ite.one()
        except EmptyPipeException:
            raise
        except StopIteration:
            self.close()
            raise

    def fetch_many(self):
        try:
            return self.ite.many(self.max_chunk_size)
        except EmptyPipeException:
            raise
        except StopIteration:
            self.close()
            raise


class InputParallelLinkPipe(LinkPipe):

    def __init__(self, parent, num, max_chunk_size=1, max_queue_size=-1, fair=True, multiprocess=True, blocking=True, flow=None, clear_on_interrupt=True, ctx='spawn'):
        super().__init__(parent.segid)
        self.num = num
        self.parent = parent
        self.max_chunk_size = max_chunk_size
        self.max_queue_size = max_queue_size
        self.fair = fair
        self.multiprocess = multiprocess
        self.blocking = blocking
        self.flow = flow
        self.clear_on_interrupt = clear_on_interrupt
        self.ctx = ctx

    def create_cnx(self):
        if self.multiprocess:
            return create_mp_cnx(self.max_queue_size, self.blocking, self.clear_on_interrupt, mp.get_context(self.ctx))
        else:
            return create_mt_cnx(self.max_queue_size, self.blocking, self.clear_on_interrupt)

    def create_link(self, input=None):
        ite = self.parent.create(input)
        cnxs = np.transpose([self.create_cnx() for _ in range(self.num)]).tolist()
        # task = ParallelPipeTask(ite, [CnxDrainPipeTask(ite, cnx, self.max_chunk_size) for cnx in cnxs[0]], self.fair)
        task = ParallelPipeTask(ite, cnxs[0], self.max_chunk_size, self.fair)
        worker = threading.Thread(target=worker_run, args=(task, WorkerData(self.segid, False, True), self.flow))
        worker.start()
        return None, [cnx.store for cnx in cnxs[1]], (cnxs[1], self.multiprocess, self.ctx)


class OutputParallelPipe(Pipe):

    def __init__(self, segid):
        super().__init__(segid, PipeLevel.PARALLEL)

    def create(self, input):
        return input


##########
# REDUCE #
##########

class InputReduceLinkPipe(LinkPipe):

    def __init__(self, parent, max_chunk_size=1, max_queue_size=-1, flow=None, clear_on_interrupt=True):
        super().__init__(parent.segid, PipeLevel.PARALLEL)
        self.parent = parent
        self.max_chunk_size = max_chunk_size
        self.max_queue_size = max_queue_size
        self.flow = flow
        self.clear_on_interrupt = clear_on_interrupt

    def create_link(self, input=None):
        cnxs, multiprocess, ctx = input
        cnxs_out = []
        if multiprocess:
            ctx = mp.get_context(ctx)
            for wid, cnx in enumerate(cnxs):
                cnx_in, cnx_out = create_mp_cnx(self.max_queue_size, False, self.clear_on_interrupt, ctx)
                worker = ctx.Process(target=worker_run, args=(CnxDrainPipeTask(self.parent.create(cnx), cnx_in, self.max_chunk_size), WorkerData(self.segid, True, False, wid), self.flow))
                worker.start()
                cnx.close()
                cnx_in.close(False)
                cnxs_out.append(cnx_out)
        else:
            for wid, cnx in enumerate(cnxs):
                cnx_in, cnx_out = create_mt_cnx(self.max_queue_size, False, self.clear_on_interrupt)
                worker = threading.Thread(target=worker_run, args=(CnxDrainPipeTask(self.parent.create(cnx), cnx_in, self.max_chunk_size), WorkerData(self.segid, False, False, wid), self.flow))
                worker.start()
                cnxs_out.append(cnx_out)
        return None, [cnx.store for cnx in cnxs_out], cnxs_out


class OutputReducePipeIterator(PipeIterator):

    def __init__(self, cnxs, fair, blocking=False):
        self.cnxs = cnxs
        self.fair = fair
        self.cnx_index = 0
        self.blocking = blocking

    def one(self):
        start_index = self.cnx_index
        while len(self.cnxs):
            try:
                if self.fair and (len(self.cnxs) > 1):
                    v = self.cnxs[self.cnx_index].one()
                    self.cnx_index += 1
                    if self.cnx_index == len(self.cnxs):
                        self.cnx_index = 0
                    return v
                else:
                    return next(self.cnxs[self.cnx_index])
            except EmptyPipeException:
                self.cnx_index += 1
            except StopIteration:
                del self.cnxs[self.cnx_index]
                if not len(self.cnxs):
                    break
                if self.cnx_index < start_index:
                    start_index -= 1
            if self.cnx_index == len(self.cnxs):
                self.cnx_index = 0
            if self.cnx_index != start_index:
                continue
            if self.blocking:
                time.sleep(1e-4)
            else:
                raise EmptyPipeException()
        raise StopIteration

    def set_main(self):
        self.blocking = True
        return self


class OutputReducePipe(Pipe):

    def __init__(self, segid, fair=True, blocking=False):
        super().__init__(segid)
        self.fair = fair
        self.blocking = blocking

    def create(self, input):
        return OutputReducePipeIterator(input, self.fair, self.blocking)
