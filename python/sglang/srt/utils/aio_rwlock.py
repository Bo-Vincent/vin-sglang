import asyncio


class RWLock:
    def __init__(self):
        # Protects internal state
        self._lock = asyncio.Lock()

        # Condition variable used to wait for state changes
        self._cond = asyncio.Condition(self._lock)

        # Number of readers currently holding the lock
        self._readers = 0

        # Snapshot readers keep ordinary inference readers admissible while an
        # update queues, but do not let later snapshots extend that window.
        self._snapshot_readers = 0

        # Whether a writer is currently holding the lock
        self._writer_active = False

        # How many writers are queued waiting for a turn
        self._waiting_writers = 0

    @property
    def reader_lock(self):
        """
        A context manager for acquiring a shared (reader) lock.

        Example:
            async with rwlock.reader_lock:
                # read-only access
        """
        return _ReaderLock(self)

    @property
    def snapshot_reader_lock(self):
        """A snapshot reader that preserves inference service while active."""
        return _ReaderLock(self, snapshot=True)

    @property
    def writer_lock(self):
        """
        A context manager for acquiring an exclusive (writer) lock.

        Example:
            async with rwlock.writer_lock:
                # exclusive access
        """
        return _WriterLock(self)

    async def acquire_reader(self, *, snapshot: bool = False):
        async with self._lock:
            while self._writer_active or self._reader_waits_for_writer(snapshot):
                await self._cond.wait()
            self._readers += 1
            if snapshot:
                self._snapshot_readers += 1

    def _reader_waits_for_writer(self, snapshot: bool) -> bool:
        if self._waiting_writers == 0:
            return False
        if snapshot:
            return True
        return self._snapshot_readers == 0

    async def release_reader(self, *, snapshot: bool = False):
        async with self._lock:
            self._readers -= 1
            if snapshot:
                self._snapshot_readers -= 1
            # If this was the last reader, wake up anyone waiting
            # (potentially a writer or new readers). Ending the last snapshot
            # also closes the inference admission window for a queued writer.
            if self._readers == 0 or (snapshot and self._snapshot_readers == 0):
                self._cond.notify_all()

    async def acquire_writer(self):
        async with self._lock:
            # Increment the count of writers waiting
            self._waiting_writers += 1
            try:
                # Wait while either a writer is active or readers are present
                while self._writer_active or self._readers > 0:
                    await self._cond.wait()
                self._writer_active = True
            finally:
                # Decrement waiting writers only after we've acquired the writer lock
                self._waiting_writers -= 1

    async def release_writer(self):
        async with self._lock:
            self._writer_active = False
            # Wake up anyone waiting (readers or writers)
            self._cond.notify_all()

    async def is_locked(self):
        async with self._lock:
            return self._writer_active or self._readers > 0


class _ReaderLock:
    def __init__(self, rwlock: RWLock, *, snapshot: bool = False):
        self._rwlock = rwlock
        self._snapshot = snapshot

    async def __aenter__(self):
        await self._rwlock.acquire_reader(snapshot=self._snapshot)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._rwlock.release_reader(snapshot=self._snapshot)


class _WriterLock:
    def __init__(self, rwlock: RWLock):
        self._rwlock = rwlock

    async def __aenter__(self):
        await self._rwlock.acquire_writer()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._rwlock.release_writer()
