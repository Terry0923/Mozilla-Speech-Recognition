import numpy as np
import re
import codecs
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import pandas as pd

# hyper parameters
#stop_words = set(stopwords.words('english'))
incorrect_deep_accuracy_arr = [] #accuracy of deepspeech with number transcription
correct_deep_accuracy_arr = [] #accuracy of deepspeech with english word transcription
google_accuracy_arr = [] #accuracy of google speech

#for collecting mismatching words
sub_arr = []
ins_arr = []
del_arr = []


stop_words = []
# helper function
def num2words(num):
	nums_20_90 = ['twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
	nums_0_19 = ['zero','one','Two','three','four','five','six','seven','eight',"nine", 'ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
	nums_dict = {100: 'hundred',1000:'thousand', 1000000:'million', 1000000000:'billion'}
	if num < 20:
		return nums_0_19[num]
	if num < 100:
		return nums_20_90[int(num/10)-2] + ('' if num%10 == 0 else ' ' + 	nums_0_19[num%10])
	# find the largest key smaller than num
	maxkey = max([key for key in nums_dict.keys() if key <= num])
	return num2words(int(num/maxkey)) + ' ' + nums_dict[int(maxkey)] + ('' if num%maxkey == 0 else ' ' + num2words(num%maxkey))

#check if the word are numbers
def customizedIsNumerical(str):
    if str.isnumeric():
        return True
    else:
        #sometimes the program miss something like 96. or 96%, it should still be considered numbers
        str = str[:-1]
        if(str.isnumeric()):
            return True
        return False

#convert string containing numbers to string containing english words
def convertScript(str):
    newstr = ""
    for word in str.split():
        if customizedIsNumerical(word):
            if not word.isnumeric():
                word = word[:-1]
            word = num2words(int(word))
        newstr += word + " "
    return newstr

#class that compare text to find WER and ACC
class TextComp(object):
    def __init__(self, original_text, recognition_text, encoding='utf-8'):
        # original_path: path of the original text
        # recognition_path: path of the recognized text
        # encoding: specifies the encoding which is to be used for the file
        self.original_text = original_text
        self.recognition_text = recognition_text
        self.encoding = encoding
        self.I = 0
        self.S = 0
        self.D = 0


    def Preprocess(self, text):
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text.lower())
        filtered_words = list(filter(lambda w: w not in stop_words, words))
        return filtered_words

    def WER(self, debug=False):
        r = self.Preprocess(self.original_text)
        h = self.Preprocess(self.recognition_text)
        # costs will holds the costs, like in the Levenshtein distance algorithm
        costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
        # backtrace will hold the operations we've done.
        # so we could later backtrace, like the WER algorithm requires us to.
        backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

        OP_OK = 0
        OP_SUB = 1
        OP_INS = 2
        OP_DEL = 3

        # First column represents the case where we achieve zero
        # hypothesis words by deleting all reference words.
        for i in range(1, len(r) + 1):
            costs[i][0] = i
            backtrace[i][0] = OP_DEL

        # First row represents the case where we achieve the hypothesis
        # by inserting all hypothesis words into a zero-length reference.
        for j in range(1, len(h) + 1):
            costs[0][j] = j
            backtrace[0][j] = OP_INS

        # computation
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    costs[i][j] = costs[i - 1][j - 1]
                    backtrace[i][j] = OP_OK
                else:
                    substitutionCost = costs[i - 1][j - 1] + 1  # penalty is always 1
                    insertionCost = costs[i][j - 1] + 1  # penalty is always 1
                    deletionCost = costs[i - 1][j] + 1  # penalty is always 1

                    costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                    if costs[i][j] == substitutionCost:
                        backtrace[i][j] = OP_SUB
                    elif costs[i][j] == insertionCost:
                        backtrace[i][j] = OP_INS
                    else:
                        backtrace[i][j] = OP_DEL

        # back trace though the best route:
        i = len(r)
        j = len(h)
        self.S = 0
        self.D = 0
        self.I = 0
        numCor = 0
        if debug:
            print("OP\toriginal\trecognition")
            lines = []
        while i > 0 or j > 0:
            if backtrace[i][j] == OP_OK:
                numCor += 1
                i -= 1
                j -= 1
                if debug:
                    lines.append("OK\t" + r[i] + "\t" + h[j])
            elif backtrace[i][j] == OP_SUB:
                self.S += 1
                i -= 1
                j -= 1
                if debug:
                    lines.append("SUB\t" + r[i] + "\t" + h[j])
                    sub_arr.append("SUB\t" + r[i] + "\t" + h[j])
            elif backtrace[i][j] == OP_INS:
                self.I += 1
                #self.Insertions.append(OP_INS)
                j -= 1
                if debug:
                    lines.append("INS\t" + "****" + "\t" + h[j])
                    ins_arr.append("INS\t" + "****" + "\t" + h[j])
            elif backtrace[i][j] == OP_DEL:
                self.D += 1
                #self.Deletions.append(OP_DEL)
                i -= 1
                if debug:
                    lines.append("DEL\t" + r[i] + "\t" + "****")
                    del_arr.append("DEL\t" + r[i] + "\t" + "****")
        if debug:
            lines = reversed(lines)
            for line in lines:
                print(line)
            print("#cor " + str(numCor))
            print("#sub " + str(self.S))
            print("#del " + str(self.D))
            print("#ins " + str(self.I))
            return (self.S + self.D + self.I) / float(len(r))
        wer_result = round((self.S + self.D + self.I) / float(len(r)), 3)
        return wer_result

    def Accuracy(self):
        return float(len(self.Preprocess(self.original_text)) - self.D - self.S) / len(
            self.Preprocess(self.original_text))

    def getInsertions(self):
        return self.Insertions

    def getSubstitutions(self):
        return self.Substitutions

    def getDeletions(self):
        return self.Deletions


if __name__ == '__main__':
    print("Do you want to see the detail of the performance?")    # Get number of digits
    answer = input()
    debug = False
    if(answer == "yes"):
        debug = True
        #first file
    with open ("1_Paramedic_Smith_Original_deep_2019.txt", 'r') as myfile:
        latest_deep_first = myfile.read().replace('\n', '')
    with open ("1_Paramedic_Smith_Original_google_2019.txt", 'r') as myfile:
        google_first = myfile.read().replace('\n','')
    with open ("1_Paramedic_Smith_Original_Transcript.txt", 'r') as myfile:
        original_first = myfile.read().replace('\n', '')
        original_first_deep = convertScript(original_first)
        #second file
    with open ("2_McLaren_EMT_Radio_Call_Alpha_107_Original_deep_2019.txt", 'r') as myfile:
        latest_deep_second = myfile.read().replace('\n', '')
    with open ("2_McLaren_EMT_Radio_Call_Alpha_107_Original_google_2019.txt", 'r') as myfile:
        google_second = myfile.read().replace('\n','')
    with open ("2_McLaren_EMT_Radio_Call_Alpha_107_Original_Transcript.txt", 'r') as myfile:
        original_second = myfile.read().replace('\n', '')
        original_second_deep = convertScript(original_second)

        #thid file
    with open ("3_McLaren_EMT_Radio_Call_Alpha_117_Original_deep_2019.txt", 'r') as myfile:
        latest_deep_third= myfile.read().replace('\n', '')
    with open ("3_McLaren_EMT_Radio_Call_Alpha_117_Original_google_2019.txt", 'r') as myfile:
        google_third = myfile.read().replace('\n','')
    with open ("3_McLaren_EMT_Radio_Call_Alpha_117_Original_Transcript.txt", 'r') as myfile:
        original_third = myfile.read().replace('\n', '')
        original_third_deep = convertScript(original_third)
        # fourth file
    with open ("4_McLaren_EMT_Radio_Call_Alpha_101_Original_deep_2019.txt", 'r') as myfile:
        latest_deep_fourth = myfile.read().replace('\n', '')
    with open ("4_McLaren_EMT_Radio_Call_Alpha_101_Original_google_2019.txt", 'r') as myfile:
        google_fourth = myfile.read().replace('\n','')
    with open ("4_McLaren_EMT_Radio_Call_Alpha_101_Original_Transcript.txt", 'r') as myfile:
        original_fourth = myfile.read().replace('\n', '')
        original_fourth_deep = convertScript(original_fourth)
        #fifth file
    with open ("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Original_deep_2019.txt", 'r') as myfile:
        latest_deep_fifth = myfile.read().replace('\n', '')
    with open ("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Original_google_2019.txt", 'r') as myfile:
        google_fifth = myfile.read().replace('\n','')
    with open ("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Original_Transcript.txt", 'r') as myfile:
        original_fifth = myfile.read().replace('\n', '')
        original_fifth_deep = convertScript(original_fifth)
        #sixth file
    with open ("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Original_deep_2019.txt", 'r') as myfile:
        latest_deep_sixth = myfile.read().replace('\n', '')
    with open ("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Original_google_2019.txt", 'r') as myfile:
        google_sixth = myfile.read().replace('\n','')
    with open ("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Original_Transcript.txt", 'r') as myfile:
        original_sixth = myfile.read().replace('\n', '')
        original_sixth_deep = convertScript(original_sixth)
        #seventh file
    with open ("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Original_deep_2019.txt", 'r') as myfile:
        latest_deep_seventh = myfile.read().replace('\n', '')
    with open ("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Original_google_2019.txt", 'r') as myfile:
        google_seventh = myfile.read().replace('\n','')
    with open ("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Original_Transcript.txt", 'r') as myfile:
        original_seventh = myfile.read().replace('\n', '')
        original_seventh_deep = convertScript(original_seventh)

    #first
    latest_deep_stats_first = TextComp(latest_deep_first, original_first_deep)
    print("[correct_deepspeech_first_2019] Word Error Rate:"+ str(latest_deep_stats_first.WER(debug)))
    print("[correct_deepspeech_first_2019] Accuracy:"+str(latest_deep_stats_first.Accuracy()))
    correct_deep_accuracy_arr.append(latest_deep_stats_first.Accuracy())

    incorrect_latest_deep_stats_first = TextComp(latest_deep_first, original_first)
    print("[incorrect_deepspeech_first_2019] Word Error Rate:"+ str(incorrect_latest_deep_stats_first.WER(debug)))
    print("[incorrect_deepspeech_first_2019] Accuracy:"+str(incorrect_latest_deep_stats_first.Accuracy()))
    incorrect_deep_accuracy_arr.append(incorrect_latest_deep_stats_first.Accuracy())

    google_stats_first = TextComp(google_first, original_first)
    print("[google_first_2019] Word Error Rate:"+ str(google_stats_first.WER(False)))
    print("[google_first_2019] Accuracy:"+str(google_stats_first.Accuracy()))
    google_accuracy_arr.append(google_stats_first.Accuracy())

    #second
    latest_deep_stats_second = TextComp(latest_deep_second, original_second_deep)
    print("[correct_deepspeech_second_2019] Word Error Rate:"+ str(latest_deep_stats_second.WER(debug)))
    print("[correct_deepspeech_second_2019] Accuracy:"+str(latest_deep_stats_second.Accuracy()))
    correct_deep_accuracy_arr.append(latest_deep_stats_second.Accuracy())

    incorrect_latest_deep_stats_second = TextComp(latest_deep_second, original_second)
    print("[incorrect_deepspeech_second_2019] Word Error Rate:"+ str(incorrect_latest_deep_stats_second.WER(debug)))
    print("[incorrect_deepspeech_second_2019] Accuracy:"+str(incorrect_latest_deep_stats_second.Accuracy()))
    incorrect_deep_accuracy_arr.append(incorrect_latest_deep_stats_second.Accuracy())

    google_stats_second = TextComp(google_second, original_second)
    print("[google_second_2019] Word Error Rate:"+ str(google_stats_second.WER(False)))
    print("[google_second_2019] Accuracy:"+str(google_stats_second.Accuracy()))
    google_accuracy_arr.append(google_stats_second.Accuracy())

    #third
    latest_deep_stats_third = TextComp(latest_deep_third, original_third_deep)
    print("[correct_deepspeech_third_2019] Word Error Rate:"+ str(latest_deep_stats_third.WER(debug)))
    print("[correct_deepspeech_third_2019] Accuracy:"+str(latest_deep_stats_third.Accuracy()))
    correct_deep_accuracy_arr.append(latest_deep_stats_third.Accuracy())

    incorrect_latest_deep_stats_third = TextComp(latest_deep_third, original_third)
    print("[incorrect_deepspeech_third_2019] Word Error Rate:"+ str(incorrect_latest_deep_stats_third.WER(debug)))
    print("[incorrect_deepspeech_third_2019] Accuracy:"+str(incorrect_latest_deep_stats_third.Accuracy()))
    incorrect_deep_accuracy_arr.append(incorrect_latest_deep_stats_third.Accuracy())

    google_stats_third = TextComp(google_third, original_third)
    print("[google_third_2019] Word Error Rate:"+ str(google_stats_third.WER(False)))
    print("[google_third_2019] Accuracy:"+str(google_stats_third.Accuracy()))
    google_accuracy_arr.append(google_stats_third.Accuracy())

    #fourth
    latest_deep_stats_fourth = TextComp(latest_deep_fourth, original_fourth_deep)
    print("[correct_deepspeech_fourth_2019] Word Error Rate:"+ str(latest_deep_stats_fourth.WER(debug)))
    print("[correct_deepspeech_fourth_2019] Accuracy:"+str(latest_deep_stats_fourth.Accuracy()))
    correct_deep_accuracy_arr.append(latest_deep_stats_fourth.Accuracy())

    incorrect_latest_deep_stats_fourth = TextComp(latest_deep_fourth, original_fourth)
    print("[incorrect_deepspeech_fourth_2019] Word Error Rate:"+ str(incorrect_latest_deep_stats_fourth.WER(debug)))
    print("[incorrect_deepspeech_fourth_2019] Accuracy:"+str(incorrect_latest_deep_stats_fourth.Accuracy()))
    incorrect_deep_accuracy_arr.append(incorrect_latest_deep_stats_fourth.Accuracy())

    google_stats_fourth = TextComp(google_fourth, original_fourth)
    print("[google_fourth_2019] Word Error Rate:"+ str(google_stats_fourth.WER(False)))
    print("[google_fourth_2019] Accuracy:"+str(google_stats_fourth.Accuracy()))
    google_accuracy_arr.append(google_stats_fourth.Accuracy())

    #fifth
    latest_deep_stats_fifth = TextComp(latest_deep_fifth, original_fifth_deep)
    print("[correct_deepspeech_fifth_2019] Word Error Rate:"+ str(latest_deep_stats_fifth.WER(debug)))
    print("[correct_deepspeech_fifth_2019] Accuracy:"+str(latest_deep_stats_fifth.Accuracy()))
    correct_deep_accuracy_arr.append(latest_deep_stats_fifth.Accuracy())

    incorrect_latest_deep_stats_fifth = TextComp(latest_deep_fifth, original_fifth)
    print("[incorrect_deepspeech_fifth_2019] Word Error Rate:"+ str(incorrect_latest_deep_stats_fifth.WER(debug)))
    print("[incorrect_deepspeech_fifth_2019] Accuracy:"+str(incorrect_latest_deep_stats_fifth.Accuracy()))
    incorrect_deep_accuracy_arr.append(incorrect_latest_deep_stats_fifth.Accuracy())

    google_stats_fifth = TextComp(google_fifth, original_fifth)
    print("[google_fifth_2019] Word Error Rate:"+ str(google_stats_fifth.WER(False)))
    print("[google_fifth_2019] Accuracy:"+str(google_stats_fifth.Accuracy()))
    google_accuracy_arr.append(google_stats_fifth.Accuracy())

    #sixth
    latest_deep_stats_sixth = TextComp(latest_deep_sixth, original_sixth_deep)
    print("[correct_deepspeech_sixth_2019] Word Error Rate:"+ str(latest_deep_stats_sixth.WER(debug)))
    print("[correct_deepspeech_sixth_2019] Accuracy:"+str(latest_deep_stats_sixth.Accuracy()))
    correct_deep_accuracy_arr.append(latest_deep_stats_sixth.Accuracy())

    incorrect_latest_deep_stats_sixth = TextComp(latest_deep_sixth, original_sixth)
    print("[incorrect_deepspeech_sixth_2019] Word Error Rate:"+ str(incorrect_latest_deep_stats_sixth.WER(debug)))
    print("[incorrect_deepspeech_sixth_2019] Accuracy:"+str(incorrect_latest_deep_stats_sixth.Accuracy()))
    incorrect_deep_accuracy_arr.append(incorrect_latest_deep_stats_sixth.Accuracy())

    google_stats_sixth = TextComp(google_sixth, original_sixth)
    print("[google_sixth_2019] Word Error Rate:"+ str(google_stats_sixth.WER(False)))
    print("[google_sixth_2019] Accuracy:"+str(google_stats_sixth.Accuracy()))
    google_accuracy_arr.append(google_stats_sixth.Accuracy())

    #seventh
    latest_deep_stats_seventh = TextComp(latest_deep_seventh, original_seventh_deep)
    print("[correct_deepspeech_seventh_2019] Word Error Rate:"+ str(latest_deep_stats_seventh.WER(debug)))
    print("[correct_deepspeech_seventh_2019] Accuracy:"+str(latest_deep_stats_seventh.Accuracy()))
    correct_deep_accuracy_arr.append(latest_deep_stats_seventh.Accuracy())

    incorrect_latest_deep_stats_seventh = TextComp(latest_deep_seventh, original_seventh)
    print("[incorrect_deepspeech_seventh_2019] Word Error Rate:"+ str(incorrect_latest_deep_stats_seventh.WER(debug)))
    print("[incorrect_deepspeech_seventh_2019] Accuracy:"+str(incorrect_latest_deep_stats_seventh.Accuracy()))
    incorrect_deep_accuracy_arr.append(incorrect_latest_deep_stats_seventh.Accuracy())

    google_stats_seventh = TextComp(google_seventh, original_seventh)
    print("[google_seventh_2019] Word Error Rate:"+ str(google_stats_seventh.WER(False)))
    print("[google_seventh_2019] Accuracy:"+str(google_stats_seventh.Accuracy()))
    google_accuracy_arr.append(google_stats_seventh.Accuracy())

    #write output to text files
    with open('sub_error.txt', 'w') as f:
        for item in sub_arr:
            f.write("%s\n" % item)
    with open('ins_error.txt', 'w') as f:
        for item in ins_arr:
            f.write("%s\n" % item)
    with open('del_error.txt', 'w') as f:
        for item in del_arr:
            f.write("%s\n" % item)

    # data visualization
    labels = ('audio_1', 'audio_2', 'audio_3', 'audio_4'
    , 'audio_5', 'audio_6', 'audio_7')

    df = pd.DataFrame(np.c_[incorrect_deep_accuracy_arr,correct_deep_accuracy_arr,google_accuracy_arr], index=labels, columns=['Incorrect Mozilla Deepspeech Recognition','correct Mozilla Deepspeech Recognition', 'Google Speech Recognition'])

    ax = df.plot.bar()
    ax.set_xlabel("audio file")
    ax.set_ylabel("Accuracy")
    plt.suptitle("Mozilla Deepspeech Performance without background noise")
    plt.show()
