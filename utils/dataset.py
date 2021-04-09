import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import ctypes
from multiprocessing import Process, Queue, Condition, Lock
from typing import Callable, Optional, List

queue_timeout = 600


class Dataset(object):
    def __init__(
        self, path: str, preprocess_func: Optional[Callable] = None, seed: int = 0
    ):
        self.init_from_path(path)
        self.base_seed = seed
        self.batch_queue = None
        self.batch_workers = None
        self._preprocess_func = preprocess_func

    def __getitem__(self, ind: int) -> np.ndarray:
        abspath: List = [self.data.iloc[ind]["abspath"]]
        image = self._preprocess_func(abspath)[0]
        return image

    def get_item_by_the_path(self, path: str) -> np.ndarray:
        abspath: List = list(self.data.query(f"path == '{path}'")["abspath"])[0:1]
        if len(abspath) == 0:
            raise NotImplementedError
        image = self._preprocess_func(abspath)[0]
        return image

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

    def init_from_path(self, path):
        path = os.path.expanduser(path)
        _, ext = os.path.splitext(path)
        if os.path.isdir(path):
            self.init_from_folder(path)
        elif ext == ".txt":
            self.init_from_list(path)
        else:
            raise ValueError(
                "Cannot initialize dataset from path: %s\n\
                It should be either a folder, .txt or .hdf5 file"
                % path
            )

    def init_from_folder(self, folder):
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
        self.data = pd.DataFrame(
            {"path": paths, "abspath": abspaths, "label": labels, "name": names}
        )

    def init_from_list(self, filename, folder_depth=2):
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

        self.data = pd.DataFrame(
            {"path": paths, "abspath": abspaths, "label": labels, "name": names}
        )

    def __len__(self):
        return len(self.data["path"])

    def set_base_seed(self, base_seed=0):
        self.base_seed = base_seed

    def random_samples_from_class(self, label, num_samples, exception=None):
        indices_temp = list(np.where(self.data["label"].values == label)[0])

        if exception is not None:
            indices_temp.remove(exception)
            assert len(indices_temp) > 0
        indices = []
        iterations = int(np.ceil(1.0 * num_samples / len(indices_temp)))
        for i in range(iterations):
            sample_indices = np.random.permutation(indices_temp)
            indices.append(sample_indices)
        indices = list(np.concatenate(indices, axis=0)[:num_samples])
        return indices

    def get_batch_indices(self, batch_format):
        indices_batch = []
        batch_size = batch_format["size"]

        num_classes = batch_format["num_classes"]
        assert batch_size % num_classes == 0
        num_samples_per_class = batch_size // num_classes
        idx_classes = np.random.permutation(self.classes)[:num_classes]
        indices_batch = []
        for c in idx_classes:
            indices_batch.extend(
                self.random_samples_from_class(c, num_samples_per_class)
            )

        return indices_batch

    def get_batch(self, batch_format):
        indices = self.get_batch_indices(batch_format)
        batch = {}
        for column in self.data.columns:
            batch[column] = self.data[column].values[indices]
        return batch

    def get_batch_in_range(self, l, r):
        if l >= len(self):
            return None
        indices = np.arange(l, r)
        batch = {}
        for column in self.data.columns:
            batch[column] = self.data[column].values[indices]
        return batch

    def start_batch_queue(self, batch_format, maxsize=1, num_threads=3):
        self.batch_queue = Queue(maxsize=maxsize)

        def batch_queue_worker(seed):
            np.random.seed(seed + self.base_seed)
            while True:
                batch = self.get_batch(batch_format)
                if self._preprocess_func is not None:
                    batch["image"] = self._preprocess_func(batch["abspath"])
                self.batch_queue.put(batch)

        self.batch_workers = []
        for i in range(num_threads):
            worker = Process(target=batch_queue_worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.batch_workers.append(worker)

    def start_sequential_batch_queue(self, batch_size, maxsize=1, num_threads=3):
        self.batch_queue = Queue(maxsize=maxsize)

        def batch_queue_worker(seed, lock, cur_idx):
            while True:
                lock.acquire()
                batch = self.get_batch_in_range(
                    cur_idx.value, cur_idx.value + batch_size
                )
                if batch is None:
                    # TODO: break worker, queue won't stop
                    break
                cur_idx.value += batch_size
                lock.release()
                if self._preprocess_func is not None:
                    batch["image"] = self._preprocess_func(batch["abspath"])
                self.batch_queue.put(batch)

        self.batch_workers = []
        lock = Lock()
        cur_idx = mp.Value(ctypes.c_int, 0)
        for i in range(num_threads):
            worker = Process(target=batch_queue_worker, args=(i, lock, cur_idx))
            worker.daemon = True
            worker.start()
            self.batch_workers.append(worker)

    def pop_batch_queue(self, timeout=queue_timeout):
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
