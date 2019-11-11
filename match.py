# coding=utf-8
import os, time, jieba

VERBOSE = False


class Trie:
    def __init__(self, dict_file=None, suffix_num=2):
        self.words = []
        self.words_idx = {}
        self.tnext = [{}]
        self.tword = [[]]
        if dict_file and os.path.exists(dict_file):
            self.add_words([line.strip("\n\r").split("\t")[0] for line in open(dict_file, encoding="utf-8")],
                           suffix_num)
        self.stopwords = {"的", "是", "有", "多少", "哪些", "和", "什么", "谁", "这"}

    def add_words(self, words, suffix_num=2):
        '''
        将一个词语（及其后缀）加入字典树
        :param words: 词语列表，每个元素包括词语和一个特征
        :param suffix_num: 加入字典树的后缀个数，0表示模糊串的第一个字和实体要一致，1表示模糊串的第一个字在实体的前两个字内，依次类推
        :return:
        '''
        ii = 0
        for w in words:
            ii += 1
            if VERBOSE and ii % 10000 == 0: print('Add %d words' % ii)
            id = len(self.words)
            self.words.append(w)
            w = w.lower()
            self.words_idx[w] = id
            for offset in range(min(suffix_num + 1, len(w))):
                z = 0
                for c in w[offset:]:
                    if not c in self.tnext[z]:  # 不存在节点，添加
                        self.tnext[z][c] = len(self.tword)
                        self.tword.append([])
                        self.tnext.append({})
                    z = self.tnext[z][c]
                self.tword[z].append((id, offset))
        if VERBOSE: print('Added %d words, %d nodes' % (ii, len(self.tword)))

    def approx(self, word, rep_pun=1, del_pun=0.3, add_pun=1, pun_limit=2, thres=0.6, order_pun=0.01, approx_flag=True):
        '''
        近似单词匹配
        :param word: 匹配词
        :param rep_pun: 替换惩罚
        :param del_pun: 删除惩罚
        :param add_pun: 添加惩罚
        :param pun_limit: 惩罚上限
        :param thres: 编辑距离相似度阈值
        :param order_pun: 顺序惩罚，即前面的错误会有更多的惩罚
        :param verbose: 输出详细信息
        :param approx_flag: 若置为False，则假设第一个字是对的，可提升效率但是会miss掉前缀丢失的匹配，需要配合后缀食用
        :return: 列表，每个元素包含一个匹配结果和其编辑距离相似度分数
        '''

        def push(q, arrv_dict, n, l, p):  # 将新节点判重后放入队列
            if arrv_dict.get((n, l), pun_limit + 1e-6) > p:
                q.append((n, l, p))
                arrv_dict[(n, l)] = p

        if pun_limit >= len(word): pun_limit = len(word) - 1
        qh = 0  # 队列头
        w = word.lower()
        q = [(0, 0, 0)]  # 队列，元素分别为(当前前缀树节点，当前匹配到的模糊串位置，当前惩罚)
        obj = {}  # 搜索结果
        arrv_dict = {}  # 状态判重字典
        ll = len(w)
        while qh < len(q):
            z, i, b = q[qh]
            qh += 1
            # 如果当前状态惩罚超过限制，跳过
            if b > pun_limit: continue
            if i >= len(w):  # 如果模糊串已经匹配完全，则开始找对应实体并计算相似度
                for tw, offset in self.tword[z]:
                    mw = self.words[tw]
                    b += offset * del_pun
                    s = 1 - b / max(len(w), len(mw))
                    if s > thres and s > obj.get(mw, 0):  # 分数大于阈值和已有的串，更新
                        obj[mw] = s
            c = w[i] if i < len(w) else None
            next = self.tnext[z].get(c, -1)
            # 精确匹配
            if next >= 0: push(q, arrv_dict, next, i + 1, b)
            if approx_flag:
                for ch, nx in self.tnext[z].items():
                    # 删除操作
                    push(q, arrv_dict, nx, i, b + del_pun + order_pun * max(ll - i, 0))
                    # 替换操作
                    if c != ch: push(q, arrv_dict, nx, i + 1, b + rep_pun + order_pun * max(ll - i, 0))
                # 添加操作
                push(q, arrv_dict, z, i + 1, b + add_pun + order_pun * (ll - i))
            approx_flag = True
        ret = sorted(obj.items(), key=lambda x: -x[1])
        if VERBOSE: print(word, qh, ret[:10])
        return ret

    def fuzzy_search(self, Q, match_word_num=5, min_len=4, blacklist=set(), hmm=True, **fuzzy_params):
        '''
        模糊搜索
        :param Q: 待匹配文本，字符串或者分词后的词列表
        :param match_word_len: 最长匹配词数
        :param min_len: 最短匹配词长度
        :param hmm: 设置为False则分词粒度更细，若改为False建议提升match_word_num至少为6
        :param fuzzy_params: 模糊匹配参数
        :return: 模糊搜索结果字典，key为模糊词；value为匹配词列表，每个元素包含一个匹配结果和其编辑距离相似度分数。
        '''
        ss = jieba.lcut(Q, HMM=hmm) if type(Q) == str else Q
        ret = {}
        overlaps = 0  # 前面已匹配的串覆盖到的最后的位置，用来去除掉被覆盖的匹配（匹配到了长的就不要短的了）
        for i in range(len(ss)):
            if ss[i] not in self.stopwords:
                for j in range(min(i + match_word_num, len(ss)), i, -1):
                    if j <= overlaps: break  # 如果已经有一个长的匹配了，就跳过这个
                    if ss[j - 1] in self.stopwords: continue
                    subs = "".join(ss[i:j])
                    if len(subs) < min_len or subs in blacklist or subs.startswith("基金"): continue  # 去掉差的匹配
                    res = self.approx(subs, approx_flag=False, **fuzzy_params)
                    r = [(m, s) for m, s in res if m.lower() in self.words_idx]
                    if r:
                        ret[subs] = r
                        overlaps = j
        return ret

    def exact_search(self, Q, longest_mention=20):  # TODO: optimization with ac-machine
        splits = [0]
        lasteng = False
        Q = Q.lower()
        for ii in range(len(Q)):
            if 'a' <= Q[ii] <= 'z' or '0' <= Q[ii] <= '9':  # 连续的数字或英文当作一个整体
                if lasteng: continue
                lasteng = True
            else:
                lasteng = False
            if ii > 0: splits.append(ii)
        splits.append(len(Q))

        ret = set()
        upper = longest_mention + 1
        for ik, ii in enumerate(splits):
            for jk in range(ik + 1, ik + upper):
                if jk >= len(splits): break
                jj = splits[jk]
                subs = Q[ii:jj]
                if len(subs) > len(Q) or len(subs) <= 1: continue  # min len = 2
                ret.add(subs)
        return [r for r in ret if r.lower() in self.words_idx]


if __name__ == "__main__":
    trie = Trie("data/ent.txt")
    jieba.lcut("abc")
    exams = ["诺安基金管理有限公司怎么样", "建信战略的经理是谁？", "赎回华夏长成需要多少钱？", "汇添富年丰的基金号是什么？", "交银300价值Etf连接是什么基金？", "大成景H恒现在托管费如何收"]
    for sent in exams:
        t0 = time.time()
        ret = trie.exact_search(sent)
        if not ret: ret = trie.fuzzy_search(sent)
        print("%6.4f" % (time.time() - t0), sent, ret)
