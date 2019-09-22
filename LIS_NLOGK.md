# Longest Increasing Subsequence: Greedy nLogk solution

This is basically known as output-sensitive $O(nlogk)$ greedy+ D&C algorithm, where `k` is the length of the longest increasing subsequence.

This algorithm finds the LIS in $O(nlogk)$ instead of $O(n^2)$ by keeping a sorted array $L$, which is used for binary search during the process of finding the LIS.

Here, $L$ is an array that contains the smallest ending values of the various LISs that occurs in the whole array. In other words, $L[i]=$ the smallest ending value of all length $i$ LIS that is found so far. $L$ does not contain the actual LIS, but the ending of the LIS of various sizes.
