import os
import numpy as np
import threading


class FP32ToFP16ConvertingBytesIO:
    def __init__(self, num_floats):
        self._fp16_np = np.zeros(num_floats, dtype=np.float16)
        self._curr_offset = 0
        self._incomplete_vector_value = dict()
        self._lock = threading.Lock()

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        with self._lock:
            if whence == os.SEEK_SET:
                self._curr_offset = offset
            elif whence == os.SEEK_CUR:
                self._curr_offset += offset
            elif whence == os.SEEK_END:
                self._curr_offset = 4 * len(self._fp16_np)
            else:
                raise ValueError(f"Unexpected whence={whence}")

    def getbuffer(self):
        return memoryview(self._fp16_np)

    def write(self, b):
        with self._lock:
            len_bytes = len(b)
            # Determine the boundary of the first float value
            # if byte_idx1 == 0, meaning the offset is located at the multiple of sizeof(float), the start offset of
            # float value. Otherwise, it is pointing to incomplete bytes within one float value.
            # For example, value_idx1=55, byte_idx1=2 then the offset is pointing to b2 in
            # [...54 float values, [?, ?, b2, b3]]
            value_idx1, byte_idx1 = FP32ToFP16ConvertingBytesIO._get_index(
                self._curr_offset
            )

            # We skip incomplete float value for now when having non-zero byte_idx1
            # Otherwise, we can use the given value_idx1
            copy_start_index = value_idx1 if byte_idx1 == 0 else value_idx1 + 1

            # Determine the boundary of the last float value
            # if byte_idx2 == 0, the offset is located at the multiple of sizeof(float), the start offset of
            # float value. Otherwise, it is pointing to incomplete bytes within one float value.
            # For example, value_idx2=55, byte_idx2=2 then the offset is pointing to b2 in
            # [...54 float values, [?, ?, b2, b3]]
            actual_end_offset = self._curr_offset + len_bytes
            value_idx2, byte_idx2 = FP32ToFP16ConvertingBytesIO._get_index(
                actual_end_offset
            )
            copy_end_index = value_idx2

            # Clip bytes to have complete float values
            clip_start = 0 if byte_idx1 == 0 else 4 - byte_idx1
            clip_end = len_bytes - byte_idx2
            fp32_vector_values = np.frombuffer(b[clip_start:clip_end], dtype=np.float32)

            # Convert FP32 values to FP16
            self._fp16_np[copy_start_index:copy_end_index] = fp32_vector_values

            # Try to assemble incomplete float value from leading and trailing
            self._append_incomplete_bytes(value_idx1, b, 0, clip_start, byte_idx1)
            self._append_incomplete_bytes(value_idx2, b, clip_end, len_bytes, 0)

            self._curr_offset += len_bytes
            return len_bytes

    @staticmethod
    def _get_index(offset):
        return int(offset / 4), int(offset % 4)

    def _append_incomplete_bytes(
        self, value_idx, buffer, start_offset, end_offset, byte_idx
    ):
        if start_offset == end_offset:
            return

        bytes_count = self._incomplete_vector_value.get(value_idx)
        if bytes_count is None:
            bytes_count = {"count": 0, "bytes": [0] * 4}
            self._incomplete_vector_value[value_idx] = bytes_count
        four_bytes = bytes_count["bytes"]

        offset = start_offset
        while offset < end_offset:
            four_bytes[byte_idx] = buffer[offset]
            offset += 1
            byte_idx += 1

        bytes_count["count"] += end_offset - start_offset
        if bytes_count["count"] == 4:
            self._fp16_np[value_idx] = np.frombuffer(
                bytes(four_bytes), dtype=np.float32
            )[0]
            del self._incomplete_vector_value[value_idx]
