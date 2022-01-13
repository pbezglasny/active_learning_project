from collections import defaultdict
import heapq


class DialogStats:
    def __init__(self):
        self.correct_ans = 0
        self.total_ans = 0

    def add_ans(self, correct):
        self.total_ans += 1
        if correct:
            self.correct_ans += 1

    @property
    def ratio(self):
        if self.total_ans == 0:
            return 0
        else:
            return self.correct_ans / self.total_ans

    def __repr__(self):
        return f'{self.correct_ans}/{self.total_ans}'


class DialogPrediction:
    def __init__(self):
        self.answers = None
        self.reset()

    def reset(self):
        self.answers = defaultdict(lambda: DialogStats())

    def add_answer(self, dialog_id, correct):
        self.answers[dialog_id].add_ans(correct)

    def get_bottom_k_percents(self, k):
        answer = []
        result_count = len(self.answers) * k // 100
        result_count = max(result_count, 1)
        for k, v in self.answers.items():
            if len(answer) < result_count:
                heapq.heappush(answer, (-v.ratio, k))
            else:
                prev_ratio, dialog_id = heapq.heappop(answer)
                if prev_ratio > -v.ratio:
                    heapq.heappush(answer, (prev_ratio, dialog_id))
                else:
                    heapq.heappush(answer, (-v.ratio, k))
        return [dialog_id for _, dialog_id in answer]

    def __repr__(self):
        return str(self.answers)


