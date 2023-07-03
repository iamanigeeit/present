import re


class G2PAligner:

    def __init__(self, g2p_dict_path, g2p_letters_path=''):
        self.g2p = G2PAligner.load_graph2phone(g2p_dict_path)
        self.graph_to_phoneset, self.max_graph_seq, self.max_phone_seq = self.g2p
        if g2p_letters_path:
            self.letters_pron, _, _ = G2PAligner.load_graph2phone(g2p_letters_path)
        else:
            self.letters_pron = None
        graphset = set()
        for graph_seq in self.graph_to_phoneset:
            graphset |= set(graph_seq)
        self.invalid_regex = re.compile('[^' + ''.join(sorted(graphset)) + ']')

    @staticmethod
    def load_graph2phone(dict_path):
        graph_to_phoneset = {}
        with open(dict_path) as f:
            for line in f:
                line = line.rstrip()
                if line:
                    try:
                        graph, phoneset = line.split('  ')
                        graph_to_phoneset[graph] = set(tuple(phones.split('+')) for phones in phoneset.split(' '))
                    except ValueError:
                        print('Invalid line:', line)
        max_graph_seq = max(map(len, graph_to_phoneset.keys()))
        max_phone_seq = max([len(phones) for phoneset in graph_to_phoneset.values() for phones in phoneset]) + 1
        return graph_to_phoneset, max_graph_seq, max_phone_seq

    def align(self, word, pron, traceback=None):
        if traceback is not None:
            traceback.append((word, pron))
        if word:
            for i in range(self.max_graph_seq, 0, -1):
                graph = word[:i]
                if graph in self.graph_to_phoneset:
                    phoneset = self.graph_to_phoneset[graph]
                    for j in range(self.max_phone_seq, 0, -1):
                        phone = tuple(pron[:j])
                        if phone in phoneset:  # found match
                            alignment = [(graph, phone)]
                            remaining_alignment, valid, _ = self.align(word[i:], pron[j:], traceback)
                            if valid:
                                return alignment + remaining_alignment, True, traceback
                    if ('SIL',) in phoneset:
                        alignment = [(graph, tuple())]
                        remaining_alignment, valid, _ = self.align(word[i:], pron, traceback)
                        if valid:
                            return alignment + remaining_alignment, True, traceback
            return [], False, traceback  # when graph does not match phone
        else:
            if pron:
                return [], False, traceback  # some phones are left over after all graphs are consumed
            else:
                return [], True, traceback  # everything is matched
    
    def align_nonsilent(self, word, pron, traceback=None):
        if traceback is not None:
            traceback.append((word, pron))
        if word:
            for i in range(self.max_graph_seq, 0, -1):
                graph = word[:i]
                if graph in self.graph_to_phoneset:
                    phoneset = self.graph_to_phoneset[graph]
                    for j in range(self.max_phone_seq, 0, -1):
                        phone = tuple(pron[:j])
                        if phone in phoneset:  # found match
                            alignment = [(graph, phone)]
                            remaining_alignment, valid, _ = self.align_nonsilent(word[i:], pron[j:], traceback)
                            if valid:
                                return alignment + remaining_alignment, True, traceback
            return [], False, traceback  # when graph does not match phone
        else:
            if pron:
                return [], False, traceback  # some phones are left over after all graphs are consumed
            else:
                return [], True, traceback  # everything is matched
    
    def align_best_effort(self, word, pron, traceback):
        best_len = 9999999
        best_subword = ''
        best_subpron = ''
        for subword, subpron in traceback:
            total_len = len(subword) + len(subpron)
            if total_len < best_len:
                best_len = total_len
                best_subword = subword
                best_subpron = subpron
        i = len(word) - len(best_subword)
        j = len(pron) - len(best_subpron)
        alignment, _, _ = self.align(word[:i], pron[:j])
        return alignment, best_subword, best_subpron

    def align_spell_letters(self, word, pron, traceback=None):
        if traceback is not None:
            traceback.append((word, pron))
        if word:
            for i in range(self.max_graph_seq, 0, -1):
                graph = word[:i]
                if graph in self.graph_to_phoneset:
                    phoneset = self.graph_to_phoneset[graph]
                    for j in range(self.max_phone_seq, 0, -1):
                        phone = tuple(pron[:j])
                        if phone in phoneset:  # found match
                            alignment = [(graph, phone)]
                            remaining_alignment, valid, traceback = self.align_spell_letters(
                                word[i:], pron[j:], traceback)
                            if valid:
                                return alignment + remaining_alignment, True, traceback
                    if ('SIL',) in phoneset:
                        alignment = [(graph, tuple())]
                        remaining_alignment, valid, traceback = self.align_spell_letters(
                            word[i:], pron, traceback)
                        if valid:
                            return alignment + remaining_alignment, True, traceback
                if graph in self.letters_pron:
                    phoneset = self.letters_pron[graph]
                    j = len(next(iter(phoneset)))
                    phone = tuple(pron[:j])
                    if phone in phoneset:
                        alignment = [(graph, phone)]
                        remaining_alignment, valid, traceback = self.align_spell_letters(
                            word[i:], pron[j:], traceback)
                        if valid:
                            return alignment + remaining_alignment, True, traceback
            return [], False, traceback  # when graph does not match phone
        else:
            if pron:
                return [], False, traceback  # some phones are left over after all graphs are consumed
            else:
                return [], True, traceback  # everything is matched

    def align_fallback(self, word, pron):
        traceback = []
        alignment, valid, traceback = self.align_nonsilent(word, pron, traceback=traceback)
        if valid:
            return alignment, 0
        else:
            alignment, subword, subpron = self.align_best_effort(word, pron, traceback)
            if len(subword) == 1 and len(subpron) == 1:
                return alignment + [(subword, tuple(subpron))], 1.5
            elif len(subword) == 0:
                return alignment + [('', tuple(subpron))], len(subpron)
            elif len(subpron) == 0:
                return alignment + [(subword, tuple())], len(subword)
            else:
                possible_alignments = [(subword[0], (subpron[0],)), ('', subpron[0],), (subword[0], tuple())]
                alignment_penalties = [self.align_fallback(subword[1:], subpron[1:]),
                                       self.align_fallback(subword, subpron[1:]),
                                       self.align_fallback(subword[1:], subpron)]
                sub_alignments = [x[0] for x in alignment_penalties]
                penalties = [x[1] for x in alignment_penalties]
                penalties[0] += 1.5
                penalties[1] += 1
                penalties[2] += 1
                best_choice = min(range(3), key=penalties.__getitem__)
                best_alignment = sub_alignments[best_choice]
                best_penalty = penalties[best_choice]
                return alignment + [possible_alignments[best_choice]] + best_alignment, best_penalty

    @staticmethod
    def expand_alignment(alignment):
        new_alignment = []
        for graph, phone in alignment:
            if len(graph) == len(phone) == 2 and graph not in {'wh', 're', 'le'}:
                new_alignment.append((graph[0], (phone[0],)))
                new_alignment.append((graph[1], (phone[1],)))
            else:
                new_alignment.append((graph, phone))
        return new_alignment

    def remove_invalid_chars(self, word):
        return self.invalid_regex.sub('', word)
    
    def __call__(self, word, pron):
        assert isinstance(word, str) and isinstance(pron, list)
        word = self.remove_invalid_chars(word)
        alignment, _ = self.align_fallback(word, pron)
        return G2PAligner.expand_alignment(alignment)


def convert_cmudict_nltk(cmudict_path, out_path):
    with open(out_path, 'w') as out_file:
        with open(cmudict_path) as dic:
            for line in dic:
                if not line.startswith(';'):
                    word, pron = line.rstrip().split('  ')
                    match = re.fullmatch(r".+\((\d)\)", word)
                    if match:
                        i = int(match.group(1))
                        out_file.write(f'{word} {i+1} {pron}\n')
                    else:
                        out_file.write(f'{word} 1 {pron}\n')