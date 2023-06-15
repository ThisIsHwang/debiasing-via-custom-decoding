
#preprocessing winobias dataset for testing bias.

# making the dataset

import os
from torch.utils.data import Dataset, DataLoader



class WinoBiasDataset(Dataset):
    def __init__(self, wino_data_dir, extra_gendered_words_dir, generalized_swaps_dir):
        # read the text file
        self.wino_data_dir = wino_data_dir
        self.extra_gendered_words_dir = extra_gendered_words_dir
        self.generalized_swaps_dir = generalized_swaps_dir

        # read all the winobias text file in the wino_data_dir
        self.wino_man_sentences = list()
        self.wino_woman_sentences = list()
        if os.path.exists("/home/doubleyyh/bias_mitigation/dataset/preprocessed_wino/wino_man_sentences.txt"):
            with open("/home/doubleyyh/bias_mitigation/dataset/preprocessed_wino/wino_man_sentences.txt", 'r') as f:
                data = f.read().strip().split('\n')
                self.wino_man_sentences = data
            if os.path.exists("/home/doubleyyh/bias_mitigation/dataset/preprocessed_wino/wino_woman_sentences.txt"):
                with open("/home/doubleyyh/bias_mitigation/dataset/preprocessed_wino/wino_woman_sentences.txt", 'r') as f:
                    data = f.read().strip().split('\n')
                    self.wino_woman_sentences = data
        else:
            # make the gendered words list
            self.man_words, self.woman_words = self.gender_words_list()
            self.init_text_data()




    def init_text_data(self):
        # read the text file and make the data in the directory

        # travel the directory
        for root, dirs, files in os.walk(self.wino_data_dir):
            for file in files:
                if file.endswith(".dev") or file.endswith(".test"):
                    # read the file
                    with open(os.path.join(root, file), 'r') as f:
                        data = f.read().strip().split('\n')
                        # make the data into a list
                        origin_list = [" ".join(d.split(' ')[1:]).replace('[', '').replace(']', '') for d in data]

                        man_sentence_list, woman_sentence_list = self.preprocessIt(origin_list)
                        self.wino_man_sentences += man_sentence_list
                        self.wino_woman_sentences += woman_sentence_list

        # save the list into a file
        with open("/home/doubleyyh/bias_mitigation/dataset/preprocessed_wino/wino_man_sentences.txt", 'w') as f:
            for s in self.wino_man_sentences:
                f.write(s + '\n')
        with open("/home/doubleyyh/bias_mitigation/dataset/preprocessed_wino/wino_woman_sentences.txt", 'w') as f:
            for s in self.wino_woman_sentences:
                f.write(s + '\n')

    def preprocessIt(self, sentenceList):
        # remove the [ and ] and cut the sentence into a self.gender_word_change
        result_man_list = []
        result_woman_list = []
        for s in sentenceList:
            temp_list = []
            #temp_woman_list = []
            t = s.split()
            for tt in t:
                if tt in self.man_words:
                    for ttt in self.man_words[tt]:
                        result_woman_list.append(" ".join(temp_list + [ttt]))
                        result_man_list.append(" ".join(temp_list + [tt]))
                    break
                elif tt in self.woman_words:
                    for ttt in self.woman_words[tt]:
                        result_man_list.append(" ".join(temp_list + [ttt]))
                        result_woman_list.append(" ".join(temp_list + [tt]))
                    break
                temp_list.append(tt)


        return result_man_list, result_woman_list


    def gender_words_list(self):
        # read the text file
        from collections import defaultdict
        man_words = defaultdict(set)
        woman_words = defaultdict(set)
        with open(self.generalized_swaps_dir, 'r') as f:
            data = f.read().strip().split('\n')
            tempList = [d.split('\t') for d in data]
            for t in tempList:
                man_words[t[0].strip()].add(t[1].strip())#.append()
                woman_words[t[1].strip()].add(t[0].strip())#append() #+=

        with open(self.extra_gendered_words_dir, 'r') as f:
            data = f.read().strip().split('\n')
            tempList = [d.split('\t') for d in data]
            for t in tempList:
                man_words[t[0].strip()].add(t[1].strip())#append(t[1].strip())
                woman_words[t[1].strip()].add(t[0].strip())#append(t[0].strip())  # +=


        return man_words,woman_words


    def __len__(self):
        return len(self.wino_man_sentences)

    def __getitem__(self, idx):
        return self.wino_man_sentences[idx], self.wino_woman_sentences[idx]


if __name__ == "__main__":
    wino_data_dir = "/home/doubleyyh/bias_mitigation/corefBias/WinoBias/wino/data"
    extra_gendered_words_dir = "/home/doubleyyh/bias_mitigation/corefBias/WinoBias/wino/extra_gendered_words.txt"
    generalized_swaps_dir = "/home/doubleyyh/bias_mitigation/corefBias/WinoBias/wino/generalized_swaps_fix.txt"
    dataset = WinoBiasDataset(wino_data_dir, extra_gendered_words_dir, generalized_swaps_dir)
    print()
