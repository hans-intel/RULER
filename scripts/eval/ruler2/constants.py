import re

try:
    import editdistance

    def _levenshtein(seq1, seq2):
        return editdistance.eval(seq1, seq2)
except ModuleNotFoundError:
    def _levenshtein(seq1, seq2):
        m = len(seq1)
        n = len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
        return dp[m][n]


def wer(hypotheses, references):
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference lists must have the same number of elements."
        )

    for hyp, ref in zip(hypotheses, references):
        hyp_words = hyp.split()
        ref_words = ref.split()
        words += len(ref_words)
        scores += _levenshtein(hyp_words, ref_words)

    if words == 0:
        return float("inf")
    return 1.0 * scores / words


def _sample_score_all(pred, ref_list):
    pred_l = pred.lower()
    return sum(max(1.0 if r.lower() in pred_l else 0.0, 1 - wer([pred_l], [r.lower()])) for r in ref_list) / len(ref_list)


def string_match_ruler2_all(preds, refs):
    score = sum(_sample_score_all(pred, ref) for pred, ref in zip(preds, refs)) / len(preds) * 100
    return round(score, 2)


def string_match_ruler2_2steps(preds, refs):
    last_step_preds = [pred.split("\n\n")[-1] for pred in preds]
    score = sum(_sample_score_all(pred, ref) for pred, ref in zip(last_step_preds, refs)) / len(last_step_preds) * 100
    return round(score, 2)


def string_match_ruler2_part(preds, refs):
    cleaned_preds = [re.sub(r"Document \d+:(?:.*\n)+?\n", "", pred) for pred in preds]
    score = (
        sum(
            max(max(1.0 if r.lower() in pred.lower() else 0.0, 1 - wer([pred.lower()], [r.lower()])) for r in ref)
            for pred, ref in zip(cleaned_preds, refs)
        )
        / len(cleaned_preds)
        * 100
    )
    return round(score, 2)


TASKS = {
    "mk_niah_basic": {"metric_fn": string_match_ruler2_all},
    "mk_niah_easy": {"metric_fn": string_match_ruler2_all},
    "mk_niah_medium": {"metric_fn": string_match_ruler2_all},
    "mk_niah_hard": {"metric_fn": string_match_ruler2_all},
    "mv_niah_basic": {"metric_fn": string_match_ruler2_all},
    "mv_niah_easy": {"metric_fn": string_match_ruler2_all},
    "mv_niah_medium": {"metric_fn": string_match_ruler2_2steps},
    "mv_niah_hard": {"metric_fn": string_match_ruler2_all},
    "qa_basic": {"metric_fn": string_match_ruler2_part},
    "qa_easy": {"metric_fn": string_match_ruler2_part},
    "qa_medium": {"metric_fn": string_match_ruler2_part},
    "qa_hard": {"metric_fn": string_match_ruler2_part},
}
