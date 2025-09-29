import os
import pickle
from rich.progress import track
from cs336_basics.region_timer import RegionTimer
from cs336_basics.pretokenization import get_pre_token_counts, global_special_tokens
from cs336_basics.token_pair import TokenPairCounter, PreTokens, Pair
from cs336_basics.tokenizer import BPETokenizerParams

is_main_file: bool = __name__ == "__main__"


def update_pair_counts_opt(
    new_tokens_list: list[tuple[bytes, ...]],
    affected_docs_list: list[int],
    pre_tokens: PreTokens,
    token_pair_counter: TokenPairCounter,
    region_timer: RegionTimer,
):
    region_timer.start("bulk update pairs")
    all_removed_pairs: dict[Pair, int] = {}
    all_added_pairs: dict[Pair, int] = {}
    for idx, new_tokens in zip(affected_docs_list, new_tokens_list):
        tokens_count = pre_tokens.get_item_count(idx)
        tokens = pre_tokens[idx]

        # remove old pairs
        for i in range(len(tokens) - 1):
            pair: Pair = (tokens[i], tokens[i + 1])
            count: int = token_pair_counter.remove_pair(pair, idx, tokens_count, False)
            all_removed_pairs[pair] = count

        # add new pairs
        for i in range(len(new_tokens) - 1):
            pair: Pair = (new_tokens[i], new_tokens[i + 1])
            count: int = token_pair_counter.add_pair(pair, idx, tokens_count, False)
            all_added_pairs[pair] = count
    region_timer.stop("bulk update pairs")

    # bulk update heap
    region_timer.start("bulk update heap")
    for pair, count in all_removed_pairs.items():
        if count == 0:
            continue
        token_pair_counter.push_heap(pair, count)
    for pair, count in all_added_pairs.items():
        token_pair_counter.push_heap(pair, count)
    region_timer.stop("bulk update heap")


def train(
    vocab_size: int,
    pre_tokens_dict: dict[bytes, int],
    timer: RegionTimer,
    fast: bool = True,
) -> BPETokenizerParams:
    timer.start("initialize")
    pre_tokens = PreTokens(pre_tokens_dict)

    pre_tokens_count = len(pre_tokens)
    if is_main_file:
        print(f"Loaded {pre_tokens_count} unique pre-tokens")

    bpe_params = BPETokenizerParams(vocab=dict(), merges=dict(), merges_list=[])
    for i in range(256):
        bpe_params.vocab[i] = bytes([i])

    assert len(bpe_params.vocab) == 256

    token_pair_counter: TokenPairCounter = TokenPairCounter(is_main_file=is_main_file)
    token_pair_counter.init_from(pre_tokens)
    timer.stop("initialize")

    for iter in track(range(vocab_size)):
        timer.start("find max")
        (merged_tokens, merged_count) = token_pair_counter.get_max_pair()
        if merged_count == 0:
            print("No more pairs to merge.")
            break
        timer.stop("find max")

        timer.start("add new token")
        new_token = merged_tokens[0] + merged_tokens[1]
        new_token_count: int = merged_count

        new_token_idx: int = len(bpe_params.vocab)
        bpe_params.vocab[new_token_idx] = new_token

        # add message with this assertion failure
        assert (
            merged_tokens not in bpe_params.merges
        ), f"Merge {merged_tokens} already exists in merges. merge_tokens={merged_tokens} new_token={new_token}"
        bpe_params.merges[merged_tokens] = new_token_idx
        bpe_params.merges_list.append(merged_tokens)
        if bpe_params.vocab and len(bpe_params.vocab) >= vocab_size:
            break

        if is_main_file and (iter < 5 or iter % 500 == 99):
            print(f"New token: {new_token} with count {new_token_count}.\tnew_token_idx={new_token_idx}")
        timer.stop("add new token")

        # update current_tokens_dict and token_pair_counts
        timer.start("update tokens")
        new_pre_tokens_list: list[tuple[bytes, ...]] = []
        affected_docs_list: list[int] = []
        for idx in token_pair_counter.get_pretokens(merged_tokens):
            old_pre_tokens: tuple[bytes, ...] = pre_tokens[idx]
            new_pre_tokens: list[bytes] = []
            i = 0
            while i < len(old_pre_tokens):
                if i < len(old_pre_tokens) - 1 and (old_pre_tokens[i], old_pre_tokens[i + 1]) == merged_tokens:
                    new_pre_tokens.append(new_token)
                    i += 2
                else:
                    new_pre_tokens.append(old_pre_tokens[i])
                    i += 1

            assert len(new_pre_tokens) != len(old_pre_tokens), f"Token {new_token} not found in pre_tokens"

            new_pre_tokens_list.append(tuple(new_pre_tokens))
            affected_docs_list.append(idx)
        timer.stop("update tokens")

        if fast:
            update_pair_counts_opt(
                new_pre_tokens_list,
                affected_docs_list,
                pre_tokens,
                token_pair_counter,
                region_timer=timer,
            )
            assert (
                token_pair_counter.get_token_count(merged_tokens) == 0
            ), f"After update, {merged_tokens} has count {token_pair_counter.get_token_count(merged_tokens)}, expected 0"

        timer.start("update pre_tokens")
        for idx, new_tokens_tuple in zip(affected_docs_list, new_pre_tokens_list):
            pre_tokens[idx] = new_tokens_tuple
        timer.stop("update pre_tokens")

        if not fast:
            timer.start("rebuild pairs")
            token_pair_counter.init_from(pre_tokens)
            count: int = token_pair_counter.get_token_count(merged_tokens)
            assert count == 0, f"After rebuilding, {merged_tokens} has count {count}, expected 0"
            timer.stop("rebuild pairs")

    assert len(bpe_params.vocab) <= vocab_size
    return bpe_params


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> BPETokenizerParams:
    timer = RegionTimer()
    timer.start("get pre-token counts")
    if input_path:
        pre_tokens_dict: dict[bytes, int] = get_pre_token_counts(input_path)
    else:
        pre_tokens_dict = dict[bytes, int]()
        with open("data/pretokenization_output.dat", "rb") as in_f:
            pre_tokens_dict = pickle.load(in_f)
    timer.stop("get pre-token counts")

    bpe_params: BPETokenizerParams = train(vocab_size - len(special_tokens), pre_tokens_dict, timer, fast=True)
    # Add special tokens at the end of the vocab
    for token in special_tokens:
        token = token.encode("utf-8")
        bpe_params.vocab[len(bpe_params.vocab)] = token
    if is_main_file:
        print(f"Final vocab size: {len(bpe_params.vocab)}")
    # find longest token
    timer.start("find longest token")
    longest_token = max(bpe_params.vocab.values(), key=len)
    if is_main_file:
        print(f"Longest token: {longest_token} str={longest_token.decode('utf-8')} with length {len(longest_token)}")
    timer.stop("find longest token")

    if is_main_file:
        timer.report()
    return bpe_params


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
    vocab_size = 10000 if "TinyStories" in args.file else 32000

    bpe_params: BPETokenizerParams = train_bpe(args.file, vocab_size, global_special_tokens)
    print(f"Time taken: {time.time() - now} seconds")
    data_dir = os.path.dirname(args.file)

    if "train.txt" in args.file:
        file_name = os.path.basename(args.file).replace(".txt", "")
        with open(os.path.join(data_dir, f"{file_name}-vocab.dat"), "wb") as out_f:
            pickle.dump(bpe_params.vocab, out_f)

        with open(os.path.join(data_dir, f"{file_name}-merges.dat"), "wb") as out_f:
            pickle.dump(bpe_params.merges_list, out_f)
