import multiprocessing
import regex as re
import os
from typing import BinaryIO, Iterator
from rich.progress import track
import pickle
from cs336_basics.region_timer import RegionTimer

is_main_file: bool = __name__ == "__main__"
pretokenize_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pretokenize_pattern_compiled = re.compile(pretokenize_pattern)
global_special_tokens = ["<|endoftext|>"]
split_pattern = "|".join(list(map(re.escape, global_special_tokens)))
split_pattern_compiled = re.compile(split_pattern)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_doc(doc: str, pre_tokens_dict: dict[bytes, int]) -> None:
    for m in pretokenize_pattern_compiled.finditer(doc):
        word = m.group()
        key = word.encode("utf-8")
        pre_tokens_dict[key] = pre_tokens_dict.get(key, 0) + 1


def process_chunk(idx: int, file_path: str, start: int, end: int) -> dict[bytes, int]:
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk: str = f.read(end - start).decode("utf-8")
        pre_tokens_dict: dict[bytes, int] = {}
        if idx == 0:
            print(f"Processing chunk from {start} to {end}, size {len(chunk)} bytes")
            total_count = sum([1 for _ in split_pattern_compiled.splititer(chunk)])
            print(f"Total documents in chunk {idx}: {total_count}")
            iterator: Iterator[str] = split_pattern_compiled.splititer(chunk)
            for _ in track(range(total_count), description=f"Processing chunk {idx}..."):
                process_doc(next(iterator), pre_tokens_dict)
        else:
            total_count = 0
            for doc in split_pattern_compiled.splititer(chunk):
                process_doc(doc, pre_tokens_dict)
                total_count += 1
            print(f"Total documents in chunk {idx}: {total_count}")
    return pre_tokens_dict


def get_pre_token_counts(file_path: str) -> dict[bytes, int]:
    pre_tokens_dict = dict[bytes, int]()
    region_timer = RegionTimer()
    num_processes: int = 4
    with open(file_path, "rb") as f:
        # get file size
        f.seek(0, os.SEEK_END)
        file_size = f.tell() / 1024 / 1024  # in MB
        f.seek(0)
        print(f"File size: {file_size} MB")
        num_processes = max(num_processes, int(file_size // 400) + 1)
        print(f"Using {num_processes} processes")
        region_timer.start("find boundaries")
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        region_timer.stop("find boundaries")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        print(f"Found {len(boundaries)-1} chunks")

    # use multiprocessing to process each chunk in parallel
    pre_tokens_dict: dict[bytes, int] = {}
    with multiprocessing.Pool(processes=6) as pool:
        region_timer.start("process chunks")
        all_dicts = pool.starmap(
            process_chunk,
            [(t, file_path, start, end) for t, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))],
        )
        region_timer.stop("process chunks")

        region_timer.start("merge results")
        for idx, d in enumerate(all_dicts):
            print(f"Merging {len(d)} unique pre-tokens from chunk {idx}")
            for k, v in d.items():
                pre_tokens_dict[k] = pre_tokens_dict.get(k, 0) + v
        region_timer.stop("merge results")
        print(f"Total unique pre-tokens so far: {len(pre_tokens_dict)}")
    if is_main_file:
        region_timer.report()

    return pre_tokens_dict


if __name__ == "__main__":
    import argparse
    import time
    from rich import print

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Path to the input text file.",
    )
    args = parser.parse_args()

    now = time.time()
    pre_tokens_dict = get_pre_token_counts(args.file)
    print(f"Time taken: {int(time.time() - now)} seconds")

    # get file directory of the input file
    file_dir = os.path.dirname(args.file)
    with open(os.path.join(file_dir, "pretokenization_output.dat"), "wb") as out_f:
        # Save the pre-token counts to a file, use simple binary serialization
        pickle.dump(pre_tokens_dict, out_f)
