from dataclasses import dataclass
import gc
import multiprocessing
import os
import pickle
import regex as re
from typing import Iterable, Iterator, Self
from cs336_basics.pretokenization import (
    find_chunk_boundaries,
    pretokenize_pattern_compiled,
    global_special_tokens,
    split_pattern_compiled,
)
from cs336_basics.region_timer import ContextTimer, RegionTimer
from cs336_basics.token_pair import Pair, PairIndex, PairIndexState
from rich.live import Live
from rich.text import Text
from rich import print
import numpy as np

is_main_file: bool = __name__ == "__main__"
logging_enabled: bool = False and is_main_file


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""

    vocab: dict[int, bytes]
    merges: dict[Pair, int]
    merges_list: list[Pair]


class Tokenizer:
    """Abstract base class for tokenizers."""

    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points."""

    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    """Represent a string as a sequence of bytes."""

    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")
        indices = list(map(int, string_bytes))
        return indices

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)
        string = string_bytes.decode("utf-8")
        return string


def convert_special_token_to_regex(special_tokens: list[str]) -> re.Pattern[str]:
    split_pattern = "|".join(list(map(re.escape, special_tokens)))
    split_pattern_compiled = re.compile(split_pattern)
    return split_pattern_compiled


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""

    vocab: dict[int, bytes]
    merges: list[Pair]
    merges_index_list: list[tuple[PairIndex, int]]
    merges_index_dict: dict[PairIndex, int]
    token_to_index: dict[bytes, int]
    special_tokens: list[str] | None
    end_token_str: str = "<|endoftext|>"
    end_token_id: int
    pattern_compiled: re.Pattern[str]
    timer: RegionTimer
    encode_cache: dict[str, tuple[int, ...]] = {}
    is_debug: bool = False
    pre_tokens_count: int = 0
    sp_tokens_count: int = 0
    all_pre_tokens: list[str] = []

    def __init__(self, vocab: dict[int, bytes], merges: list[Pair], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or global_special_tokens
        self.special_tokens = sorted(self.special_tokens, reverse=True)
        sp_tokens_count: int = len(self.special_tokens)
        assert (
            len(merges) == len(vocab) - 256 - sp_tokens_count
        ), f"Expected {len(vocab)-256-sp_tokens_count} merges, got {len(merges)}"

        self.token_to_index = dict()
        for token_index, token in self.vocab.items():
            self.token_to_index[token] = token_index
        
        self.end_token_id = self.token_to_index[self.end_token_str.encode("utf-8")]

        split_pattern = "(" + "|".join(list(map(re.escape, self.special_tokens))) + ")"
        self.pattern_compiled = re.compile(split_pattern)

        for token_str in special_tokens or []:
            token: bytes = token_str.encode("utf-8")
            if token in self.token_to_index:
                continue
            print(f"Adding special token {token} to vocabulary at index {len(self.vocab)}")
            index = len(self.vocab)
            self.vocab[index] = token
            self.token_to_index[token] = index

        token_to_index = {v: k for k, v in vocab.items()}
        self.merges_index_list = []
        self.merges_index_dict = {}
        for t, merge in enumerate(merges):
            index = t + 256
            pair_left_index = token_to_index[merge[0]]
            pari_right_index = token_to_index[merge[1]]
            merge_index = (pair_left_index, pari_right_index)
            self.merges_index_list.append((merge_index, index))
            self.merges_index_dict[merge_index] = index

        self.timer = RegionTimer()
        self.pre_tokens_count = 0
        self.sp_tokens_count = 0
        self.all_pre_tokens = []

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Self:
        vocab: dict[int, bytes] = dict()
        with open(vocab_filepath, "rb") as vocab_f:
            vocab = pickle.load(vocab_f)

        merges: list[Pair] = []
        with open(merges_filepath, "rb") as merges_f:
            merges = pickle.load(merges_f)

        return cls(vocab, merges, special_tokens)

    def start_timer(self, region_name: str) -> None:
        if not is_main_file:
            return
        self.timer.start(region_name)

    def stop_timer(self, region_name: str) -> None:
        if not is_main_file:
            return
        self.timer.stop(region_name)

    def _convert_pretoken_to_indices(self, str: str) -> list[int]:
        indices: list[int] = []
        pre_tokens: bytes = str.encode("utf-8")
        for b in pre_tokens:
            index: int = self.token_to_index[bytes([b])]
            indices.append(index)
        return indices

    def _convert_to_pretokens(self, string: str) -> list[str]:
        with ContextTimer(self.timer, "Get byte token indices", is_main_file):
            if logging_enabled:
                print(f"Converting simple string to indices: '{string}'")
            pretokens: list[str] = []
            for mat in pretokenize_pattern_compiled.finditer(string):
                pretokens.append(mat.group())
        self.pre_tokens_count += len(pretokens)
        if self.is_debug:
            self.all_pre_tokens.extend(pretokens)
        return pretokens

    def _process_string(self, string: str) -> list[int]:
        pretokens: list[str] = self._convert_to_pretokens(string)
        indices: list[int] = []
        for pretoken in pretokens:
            if pretoken not in self.encode_cache:
                part_indices = self._convert_pretoken_to_indices(pretoken)
                part_indices = self._merge_indices(part_indices)
                self.encode_cache[pretoken] = tuple(part_indices)
                indices.extend(part_indices)
            else:
                indices.extend(self.encode_cache[pretoken])
        return indices

    def encode(self, string: str) -> list[int]:
        indices: list[int] = []
        for index in self.encode_stream(string):
            indices.append(index)
        return indices

    def encode_stream(self, string: str) -> Iterator[int]:
        if not self.special_tokens:
            yield from self._process_string(string)
            return

        chunks = self.pattern_compiled.splititer(string)
        for part in chunks:
            assert isinstance(part, str), f"Expected str, got {type(part)}"
            if part in self.special_tokens:
                if logging_enabled:
                    print(f"Found special token: '{part}'")
                if self.is_debug:
                    self.all_pre_tokens.append(part)
                token: bytes = part.encode("utf-8")
                index: int = self.token_to_index[token]
                self.sp_tokens_count += 1
                yield index
            else:
                part_indices = self._process_string(part)
                yield from part_indices

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            yield from self.encode(string)

    def decode(self, indices: list[int]) -> str:
        bytes_list: list[bytes] = []
        for i in indices:
            assert i in self.vocab, f"Index {i} not in vocabulary"
            bytes_list.append(self.vocab[i])
        byte_string = b"".join(bytes_list)
        string = byte_string.decode("utf-8", errors="replace")
        return string

    def _merge_indices(self, indices: list[int]) -> list[int]:
        with ContextTimer(self.timer, "Merging indices", is_main_file):
            if len(indices) < 2:
                return indices

            pair_state: PairIndexState = PairIndexState(self.merges_index_dict, self.timer, is_main_file)
            pair_state.rebuild(indices)

            while True:
                pair, new_index = pair_state.get_next()
                if new_index < 0:
                    break
                indices = self._merge(indices, pair, new_index, pair_state)
                if len(indices) < 2:
                    return indices
            return indices

    def _merge(self, indices: list[int], pair: PairIndex, new_index: int, pair_state: PairIndexState) -> list[int]:
        """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
        count: int = pair_state.count(pair)
        if count == 0:
            return indices
        i: int = 0
        last_found_i: int = -1
        new_indices: list[int] = []
        new_pair_indices: list[int] = []
        # print(f"pair = {pair} indices {indices} pair_state {dict(pair_state)}")
        while i < len(indices):
            if i + 1 >= len(indices) or len(new_pair_indices) >= count:
                new_indices.append(indices[i])
                i += 1
                continue
            if indices[i] != pair[0] or indices[i + 1] != pair[1]:
                new_indices.append(indices[i])
                i += 1
                continue

            pair_state.remove(pair)
            if i > 0 and last_found_i < 0 or i - 1 > last_found_i + 1:
                prev_pair = (indices[i - 1], indices[i])
                pair_state.remove(prev_pair)

            if i + 2 < len(indices):
                next_pair = (indices[i + 1], indices[i + 2])
                pair_state.remove(next_pair)

            last_found_i = i
            new_pair_indices.append(len(new_indices))
            new_indices.append(new_index)
            i += 2

        if is_main_file and len(new_pair_indices) == 0:
            # print(f"Warning: no pairs found for {pair} in {indices}. indices_pair_state: {dict(pair_state)}")
            return indices

        for idx in new_pair_indices:
            if idx > 0:
                pair_index = new_indices[idx - 1], new_indices[idx]
                pair_state.add(pair_index)
            if idx + 1 < len(new_indices):
                pair_index = new_indices[idx], new_indices[idx + 1]
                pair_state.add(pair_index)
        return new_indices


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """Given `string` that has been tokenized into `indices`, ."""
    num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens


def process_chunk(idx: int, file_path: str, start: int, end: int, tokenizer: BPETokenizer) -> int:
    # check np file exists
    output_file = file_path.replace(".txt", f"-bpe-{idx}.npy")
    now = time.time()
    chunk_size = end - start
    token_count: int = 0
    estimated_tokens = int(chunk_size / 3.5)
    mm = np.memmap(output_file, dtype=np.uint16, mode="w+", shape=(estimated_tokens,))
    # 1. first solution, read line by line
    if arg.solution == 1:
        # this solution isn't correct, because we split by lines, which could affect the pretokenization
        with open(file_path, "r") as f:
            f.seek(start)
            while f.tell() < end:
                line = f.readline()
                line_tokens = tokenizer.encode(line)
                for token in line_tokens:
                    mm[token_count] = token
                    token_count += 1
            read_bytes = f.tell() - start
            assert read_bytes == chunk_size, f"Chunk {idx+1} read {read_bytes} bytes, expected {chunk_size} bytes"
    else:
        # 2. second solution, read the whole chunk
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_bytes: bytes = f.read(chunk_size)
            chunk: str = chunk_bytes.decode("utf-8")
            assert f.tell() == end, f"Chunk {idx+1} read {f.tell()-start} bytes, expected {chunk_size} bytes"

            for token_id in tokenizer.encode(chunk):
                mm[token_count] = token_id
                token_count += 1
    if tokenizer.is_debug:
        with open(f"chunk-{idx+1}-pre-tokens_{arg.solution}.pkl", "wb") as f:
            pickle.dump(tokenizer.all_pre_tokens, f)
    print(
        f"Chunk {idx+1} pre_tokens_count: {tokenizer.pre_tokens_count} special_tokens_count: {tokenizer.sp_tokens_count}"
    )
    time_taken = time.time() - now
    compress_ratio = chunk_size / token_count if token_count > 0 else 0
    bytes_per_sec = chunk_size / time_taken / 1024 / 1024
    print(
        f"Processed chunk {idx+1}, time_taken={time_taken:.2f} seconds, speed={bytes_per_sec:.2f} MB/s, compression ratio={compress_ratio:.2f}"
    )
    mm.flush()
    print(f"Saved {token_count:,} tokens to {output_file}")

    del mm
    gc.collect()

    pre_size = os.path.getsize(output_file)
    # Truncate the file to the real size (2 bytes per token)
    with open(output_file, "r+b") as f:
        f.truncate(token_count * 2)
    after_size = os.path.getsize(output_file)
    assert (
        after_size == token_count * 2
    ), f"File size after truncation {after_size} bytes, expected {token_count*2} bytes"
    print(f"chunk {idx + 1} before truncation: {pre_size:,} bytes, after truncation: {after_size:,} bytes")
    return token_count


def split_file_to_chunks(file_path: str, num_chunks: int) -> list[tuple[int, int]]:
    boundaries: list[int] = []
    with open(file_path, "rb") as f:
        # get file size
        f.seek(0, os.SEEK_END)
        total_bytes = f.tell()
        file_size = total_bytes / 1024 / 1024  # in MB
        f.seek(0)
        print(f"File size: {file_size:.2f} MB")
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
    return list(zip(boundaries[:-1], boundaries[1:]))


def process_file_multi_process(file_path: str, tokenizer: BPETokenizer) -> int:
    from rich.progress import track

    now = time.time()
    total_bytes: int = os.path.getsize(file_path)
    desired_num_chunk = max(10, int(total_bytes / 1024 / 1024 // 50) + 1)
    boundaries = split_file_to_chunks(file_path, desired_num_chunk)
    print(f"Found {len(boundaries)} chunks, desired {desired_num_chunk} chunks")
    assert len(boundaries) == desired_num_chunk, f"Expected {desired_num_chunk} chunks, got {len(boundaries)}"

    token_count: int = 0
    with multiprocessing.Pool(processes=10) as pool:
        all_token_lists = pool.starmap(
            process_chunk,
            [(t, file_path, start, end, tokenizer) for t, (start, end) in enumerate(boundaries)],
        )
        for c in all_token_lists:
            token_count += c
    time_taken = time.time() - now
    compress_ratio: float = total_bytes / token_count
    bytes_per_sec = total_bytes / time_taken / 1024 / 1024
    print("-" * 80)
    print(
        f"Generate {token_count:,} tokens, time_taken {time_taken:.2f} seconds, {bytes_per_sec:.2f} MB/s, compression ratio {compress_ratio:.2f}"
    )
    return len(boundaries)


def process_file(file_path: str, tokenizer: BPETokenizer) -> None:
    from rich.progress import track

    now = time.time()
    total_bytes: int = os.path.getsize(file_path)
    desired_num_chunk = max(10, int(total_bytes / 1024 / 1024 // 50) + 1)
    boundaries = split_file_to_chunks(file_path, desired_num_chunk)
    print(f"Found {len(boundaries)} chunks, desired {desired_num_chunk} chunks")
    assert len(boundaries) == desired_num_chunk, f"Expected {desired_num_chunk} chunks, got {len(boundaries)}"

    with open(file_path, "rb") as f:
        start, end = boundaries[0]
        f.seek(start)
        sample_bytes = f.read(end - start)
    sample_str = sample_bytes.decode("utf-8", errors="replace")
    sample_tokens = tokenizer.encode(sample_str)
    sample_compress_ratio = get_compression_ratio(sample_str, sample_tokens)
    print(
        f"Sampled {end - start} bytes, got {len(sample_tokens)} tokens, compression ratio {sample_compress_ratio:.2f}"
    )

    approx_total_tokens = int(total_bytes / sample_compress_ratio)
    print(f"Estimated total tokens: {approx_total_tokens / 1_000_000:.2f}M")
    print("-" * 80)
    output_file = file_path.replace(".txt", f"-bpe.npy")
    mm = np.memmap(output_file, dtype=np.uint16, mode="w+", shape=(int(approx_total_tokens * 1.15),))
    idx = 0
    with open(file_path, "rb") as f:
        for t, (start, end) in enumerate(boundaries):
            print(f"Chunk {t + 1}: {start:,} to {end:,}, size {end - start:,} bytes")
            f.seek(start)
            chunk_bytes: bytes = f.read(end - start)
            chunk: str = chunk_bytes.decode("utf-8", errors="replace")
            estimated_chunk_tokens = int((end - start) / sample_compress_ratio) * 1.1
            for token_id in track(
                tokenizer.encode_stream(chunk), total=estimated_chunk_tokens, description=f"Processing chunk {t+1}..."
            ):
                mm[idx] = token_id
                idx += 1
    time_taken = time.time() - now
    bytes_per_sec = total_bytes / time_taken / 1024 / 1024
    compress_ratio = total_bytes / idx
    print(f"Time taken: {time_taken:.2f} seconds, {bytes_per_sec:.2f} MB/s, compression ratio {compress_ratio:.2f}")
    print(f"Saved {idx:,} tokens, saved to {output_file}")
    mm.flush()
    del mm
    gc.collect()

    pre_size = os.path.getsize(output_file)
    # Truncate the file to the real size (2 bytes per token)
    with open(output_file, "r+b") as f:
        f.truncate(idx * 2)
    after_size = os.path.getsize(output_file)
    print(f"File size before truncation: {pre_size:,} bytes, after truncation: {after_size:,} bytes")
    return


if is_main_file:
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Path to the input text file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not actually run the tokenizer, just print the first 10 documents.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="",
        help="If set, run a test on the given string.",
    )
    parser.add_argument(
        "--multi-process",
        action="store_true",
        default=False,
        help="If set, use multiple processes to tokenize the file.",
    )
    parser.add_argument(
        "--solution",
        type=int,
        default=2,
        choices=[1, 2],
        help="Which solution to use for processing the file. 1 or 2.",
    )

    arg = parser.parse_args()

    now = time.time()
    file_path = arg.file

    # Load the tokenizer
    file_prefix = file_path.replace("train.txt", "train")
    file_prefix = file_prefix.replace("valid.txt", "train")
    vocab_file: str = f"{file_prefix}-vocab.dat"
    merges_file: str = f"{file_prefix}-merges.dat"
    print(f"Loading tokenizer from {vocab_file} and {merges_file}")
    tokenizer = BPETokenizer.from_files(vocab_file, merges_file, special_tokens=["<|endoftext|>"])
    tokenizer.is_debug = arg.dry_run

    if not arg.multi_process:
        file_path = arg.test if arg.test else arg.file
        process_file(file_path, tokenizer)
        tokenizer.timer.report()
    else:
        file_path = arg.test if arg.test else arg.file
        part_file_count: int = process_file_multi_process(file_path, tokenizer)
        if arg.dry_run:
            exit(0)
        # merge all part files
        output_file = file_path.replace(".txt", f"-bpe-merged.npy")
        total_tokens = 0
        with open(output_file, "wb") as out_f:
            for i in range(part_file_count):
                part_file = file_path.replace(".txt", f"-bpe-{i}.npy")
                part_size = os.path.getsize(part_file)
                total_tokens += part_size // 2
                with open(part_file, "rb") as in_f:
                    out_f.write(in_f.read())
                os.remove(part_file)
        print(f"Merged {part_file_count} part files into {output_file}, total tokens {total_tokens / 1_000_000:.2f}M")
