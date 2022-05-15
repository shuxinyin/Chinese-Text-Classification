import torch


def beam_search_decoder(post, top_k):
    """
    Parameters:
        post(Tensor) – the output probability of decoder. shape = (batch_size, seq_length, vocab_size).
        top_k(int) – beam size of decoder. shape
    return:
        indices(Tensor) – a beam of index sequence. shape = (batch_size, beam_size, seq_length).
        log_prob(Tensor) – a beam of log likelihood of sequence. shape = (batch_size, beam_size).
    """

    batch_size, seq_length, vocab_size = post.shape
    log_post = post.log()
    log_prob, indices = log_post[:, 0, :].topk(top_k, sorted=True)  # first word top-k candidates
    indices = indices.unsqueeze(-1)
    for i in range(1, seq_length):
        log_prob = log_prob.unsqueeze(-1) + log_post[:, i, :].unsqueeze(1).repeat(1, top_k, 1)  # word by word
        log_prob, index = log_prob.view(batch_size, -1).topk(top_k, sorted=True)
        indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)
    return indices, log_prob


class Solution:
    def nthUglyNumber(self, n: int) -> int:
        dp = [1] * n
        a, b, c, = 0, 0, 0

        for i in range(1, n):
            n2, n3, n5 = dp[a] * 2, dp[b] * 3, dp[c] * 5
            dp[i] = min(n2, n3, n5)

            if dp[i] == n2:
                a += 1
            if dp[i] == n3:
                b += 1
            if dp[i] == n5:
                c += 1
        return dp[-1]


class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        # Time: O(n^2)
        # Space: O(n)
        dp = [1 / 6] * 6
        for i in range(2, n + 1):
            tmp_dp = [0] * (5 * i + 1)
            for j in range(len(dp)):
                for k in range(6):
                    tmp_dp[j + k] += dp[j] / 6
            dp = tmp_dp
        return dp


if __name__ == '__main__':
    post = torch.softmax(torch.randn([32, 20, 1000]), -1)
    print(post)
    indices, log_prob = beam_search_decoder(post, top_k=3)
    print(indices.shape)
