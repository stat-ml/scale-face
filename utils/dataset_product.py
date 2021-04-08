import os
from collections import namedtuple
from typing import Union
import numpy as np
import pandas as pd
import multiprocessing as mp
import ctypes
from multiprocessing import Process, Queue, Lock

DPath = namedtuple("DPath", ["path", "prefix"])


class DatasetProduct:
    def __init__(
        self, path_query: Union[str, DPath], path_distractor: Union[str, DPath]
    ):
        if isinstance(path_query, str):
            path_query = DPath(path=path_query, prefix=None)
        if isinstance(path_distractor, str):
            path_distractor = DPath(path=path_distractor, prefix=None)
        self.init_from_path(path_query, is_query=True)
        self.init_from_path(path_distractor, is_query=False)
        self.batch_queue = None
        self.batch_workers = None

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
        return self.data[key]

    def _delitem(self, key):
        self.data.__delitem__(key)

    @property
    def num_classes(self):
        return len(self.data["label"].unique())

    @property
    def classes(self):
        return self.data["label"].unique()

    @property
    def size(self):
        return self.data.shape[0]

    @property
    def loc(self):
        return self.data.loc

    @property
    def iloc(self):
        return self.data.iloc

    def init_from_path(self, path: DPath, is_query):
        path = os.path.expanduser(path.path)
        _, ext = os.path.splitext(path)

        if os.path.isdir(path):
            self.init_from_folder(path, is_query=is_query)
        elif ext == ".txt":
            self.init_from_list(path, is_query=is_query)
        else:
            raise ValueError(
                "Cannot initialize dataset from path: %s\n\
                It should be either a folder, .txt or .hdf5 file"
                % path
            )

    def init_from_folder(self, folder, is_query):
        folder = os.path.abspath(os.path.expanduser(folder))
        class_names = os.listdir(folder)
        class_names.sort()
        paths = []
        labels = []
        names = []
        for label, class_name in enumerate(class_names):
            classdir = os.path.join(folder, class_name)
            if os.path.isdir(classdir):
                images_class = os.listdir(classdir)
                images_class.sort()
                images_class = [os.path.join(class_name, img) for img in images_class]
                paths.extend(images_class)
                labels.extend(len(images_class) * [label])
                names.extend(len(images_class) * [class_name])
        abspaths = [os.path.join(folder, p) for p in paths]
        if is_query is True:
            self.data_query = pd.DataFrame(
                {"path": paths, "abspath": abspaths, "label": labels, "name": names}
            )
            self.prefix_query = folder
        else:
            self.data_distractor = pd.DataFrame(
                {"path": paths, "abspath": abspaths, "label": labels, "name": names}
            )
            self.prefix_distractor = folder

    def init_from_list(self, filename, is_query, folder_depth=2):
        with open(filename, "r") as f:
            lines = f.readlines()
        lines = [line.strip().split(" ") for line in lines]
        abspaths = [os.path.abspath(line[0]) for line in lines]
        paths = ["/".join(p.split("/")[-folder_depth:]) for p in abspaths]
        if len(lines[0]) == 2:
            labels = [int(line[1]) for line in lines]
            names = [str(lb) for lb in labels]
        elif len(lines[0]) == 1:
            names = [p.split("/")[-folder_depth] for p in abspaths]
            _, labels = np.unique(names, return_inverse=True)
        else:
            raise ValueError(
                'List file must be in format: "fullpath(str) \
                                        label(int)" or just "fullpath(str)"'
            )

        if is_query is True:
            self.data_query = pd.DataFrame(
                {"path": paths, "abspath": abspaths, "label": labels, "name": names}
            )
            self.prefix_query = abspaths[0].split("/")[:-folder_depth]
        else:
            self.data_distractor = pd.DataFrame(
                {"path": paths, "abspath": abspaths, "label": labels, "name": names}
            )
            self.prefix_distractor = abspaths[0].split("/")[:-folder_depth]

    def get_batch_in_range(self, l, r):
        if l >= len(self):
            return None
        indices = np.arange(l, r)
        batch = {}
        for column in self.data.columns:
            batch[column] = self.data[column].values[indices]
        return batch

    def start_sequential_batch_queue(
        self, batch_format, proc_func=None, maxsize=1, num_threads=3
    ):
        self.batch_queue = Queue(maxsize=maxsize)

        def batch_queue_worker(seed, lock, cur_idx):
            while True:
                lock.acquire()
                batch = self.get_batch_in_range(
                    cur_idx.value, cur_idx.value + batch_format["size"]
                )
                if batch is None:
                    # TODO: break worker, queue won't stop
                    break
                cur_idx.value += batch_format["size"]
                lock.release()
                if proc_func is not None:
                    batch["image"] = proc_func(batch["abspath"])
                self.batch_queue.put(batch)

        self.batch_workers = []
        lock = Lock()
        cur_idx = mp.Value(ctypes.c_int, 0)
        for i in range(num_threads):
            worker = Process(target=batch_queue_worker, args=(i, lock, cur_idx))
            worker.daemon = True
            worker.start()
            self.batch_workers.append(worker)

    def pop_batch_queue(self, timeout):
        return self.batch_queue.get(block=True, timeout=timeout)

    def release_queue(self):
        if self.index_queue is not None:
            self.index_queue.close()
        if self.batch_queue is not None:
            self.batch_queue.close()
        if self.index_worker is not None:
            self.index_worker.terminate()
            del self.index_worker
            self.index_worker = None
        if self.batch_workers is not None:
            for w in self.batch_workers:
                w.terminate()
                del w
            self.batch_workers = None
