import numpy as np
import re
import codecs
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import pandas as pd

google_accuracy_arr = []
clean_deep_accuracy_arr = []
cafe_low_deep_accuracy_arr = []
cafe_medium_deep_accuracy_arr = []
cafe_high_deep_accuracy_arr = []
people_low_deep_accuracy_arr = []
people_medium_deep_accuracy_arr = []
people_high_deep_accuracy_arr = []
sirens_low_deep_accuracy_arr = []
sirens_medium_deep_accuracy_arr = []
sirens_high_deep_accuracy_arr = []
stop_words = set(stopwords.words('english'))
# stop_words = []
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
    debug = False

    ###############read files here##################
    #first file
    #google
    with open ("1_Paramedic_Smith_Original_google_2019.txt", 'r') as myfile:
        google_first = myfile.read().replace('\n','')
    #clean
    with open ("1_Paramedic_Smith_Original_deep_2019.txt", 'r') as myfile:
        latest_deep_first = myfile.read().replace('\n', '')
    #cafe-low
    with open ("1_Paramedic_Smith_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_first_cafe_low = myfile.read().replace('\n', '')
    #cafe-medium
    with open ("1_Paramedic_Smith_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_first_cafe_medium = myfile.read().replace('\n', '')
    #cafe-high
    with open ("1_Paramedic_Smith_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_first_cafe_high = myfile.read().replace('\n', '')
    #people-low
    with open ("1_Paramedic_Smith_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_first_people_low = myfile.read().replace('\n', '')
    #people-medium
    with open ("1_Paramedic_Smith_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_first_people_medium = myfile.read().replace('\n', '')
    #people-high
    with open ("1_Paramedic_Smith_People_High_Noise.txt", 'r') as myfile:
        latest_deep_first_people_high = myfile.read().replace('\n', '')
    #sirens-low
    with open ("1_Paramedic_Smith_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_first_sirens_low = myfile.read().replace('\n', '')
    #sirens-medium
    with open ("1_Paramedic_Smith_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_first_sirens_medium = myfile.read().replace('\n', '')
    #sirens-high
    with open ("1_Paramedic_Smith_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_first_sirens_high = myfile.read().replace('\n', '')
    #golden script
    with open ("1_Paramedic_Smith_Original_Transcript.txt", 'r') as myfile:
        original_first = myfile.read().replace('\n', '')
        original_first_deep = convertScript(original_first)

    #fifth file
    #google
    with open ("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Original_google_2019.txt", 'r') as myfile:
        google_fifth = myfile.read().replace('\n','')
    #clean
    with open ("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Original_deep_2019.txt", 'r') as myfile:
        latest_deep_fifth = myfile.read().replace('\n', '')
	#cafe-low
    with open ("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_fifth_cafe_low = myfile.read().replace('\n', '')
	#people-medium
    with open ("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_fifth_people_medium = myfile.read().replace('\n', '')
	#sirens-high
    with open ("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_fifth_sirens_high = myfile.read().replace('\n', '')
	#cafe-medium
    with open ("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_fifth_cafe_medium = myfile.read().replace('\n', '')
	#people-high
    with open ("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_People_High_Noise.txt", 'r') as myfile:
        latest_deep_fifth_people_high = myfile.read().replace('\n', '')
	#sirens-low
    with open ("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_fifth_sirens_low = myfile.read().replace('\n', '')
	#cafe-high
    with open ("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_fifth_cafe_high = myfile.read().replace('\n', '')
	#people-low
    with open ("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_fifth_people_low = myfile.read().replace('\n', '')
	#sirens-medium
    with open ("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_fifth_sirens_medium = myfile.read().replace('\n', '')
    #golden script
    with open ("5_McLaren_EMT_Radio_Call_Alpha_107_Rerecording_Original_Transcript.txt", 'r') as myfile:
        original_fifth = myfile.read().replace('\n', '')
        original_fifth_deep = convertScript(original_fifth)
    #sixth file
    #google
    with open ("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Original_google_2019.txt", 'r') as myfile:
        google_sixth = myfile.read().replace('\n','')

    #clean
    with open ("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Original_deep_2019.txt", 'r') as myfile:
        latest_deep_sixth = myfile.read().replace('\n', '')
	#cafe-low
    with open ("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_sixth_cafe_low = myfile.read().replace('\n', '')
	#cafe-medium
    with open ("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_sixth_cafe_medium = myfile.read().replace('\n', '')
	#cafe-high
    with open ("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_sixth_cafe_high = myfile.read().replace('\n', '')
	#people-low
    with open ("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_sixth_people_low = myfile.read().replace('\n', '')
	#people-medium
    with open ("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_sixth_people_medium = myfile.read().replace('\n', '')
	#people-high
    with open ("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_People_High_Noise.txt", 'r') as myfile:
        latest_deep_sixth_people_high = myfile.read().replace('\n', '')
	#sirens-low
    with open ("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_sixth_sirens_low = myfile.read().replace('\n', '')
	#sirens-medium
    with open ("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_sixth_sirens_medium = myfile.read().replace('\n', '')
	#sirens-high
    with open ("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_sixth_sirens_high = myfile.read().replace('\n', '')
    #golden script
    with open ("6_McLaren_EMT_Radio_Call_Alpha_117_Rerecording_Original_Transcript.txt", 'r') as myfile:
        original_sixth = myfile.read().replace('\n', '')
        original_sixth_deep = convertScript(original_sixth)
    #seventh file
    #google
    with open ("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Original_google_2019.txt", 'r') as myfile:
        google_seventh = myfile.read().replace('\n','')
    #clean
    with open ("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Original_deep_2019.txt", 'r') as myfile:
        latest_deep_seventh = myfile.read().replace('\n', '')
	#cafe-low
    with open ("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Cafeteria_Low_Noise.txt", 'r') as myfile:
        latest_deep_seventh_cafe_low = myfile.read().replace('\n', '')
	#cafe-medium
    with open ("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Cafeteria_Medium_Noise.txt", 'r') as myfile:
        latest_deep_seventh_cafe_medium = myfile.read().replace('\n', '')
	#cafe-high
    with open ("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Cafeteria_High_Noise.txt", 'r') as myfile:
        latest_deep_seventh_cafe_high = myfile.read().replace('\n', '')
	#people-low
    with open ("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_People_Low_Noise.txt", 'r') as myfile:
        latest_deep_seventh_people_low = myfile.read().replace('\n', '')
	#people-medium
    with open ("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_People_Medium_Noise.txt", 'r') as myfile:
        latest_deep_seventh_people_medium = myfile.read().replace('\n', '')
	#people-high
    with open ("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_People_High_Noise.txt", 'r') as myfile:
        latest_deep_seventh_people_high = myfile.read().replace('\n', '')
	#sirens-low
    with open ("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Sirens_Low_Noise.txt", 'r') as myfile:
        latest_deep_seventh_sirens_low = myfile.read().replace('\n', '')
	#sirens-medium
    with open ("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Sirens_Medium_Noise.txt", 'r') as myfile:
        latest_deep_seventh_sirens_medium = myfile.read().replace('\n', '')
	#sirens-high
    with open ("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Sirens_High_Noise.txt", 'r') as myfile:
        latest_deep_seventh_sirens_high = myfile.read().replace('\n', '')
    #golden script
    with open ("7_McLaren_EMT_Radio_Call_Alpha_101_Rerecording_Original_Transcript.txt", 'r') as myfile:
        original_seventh = myfile.read().replace('\n', '')
        original_seventh_deep = convertScript(original_seventh)

    ######################GENERATE WER###########################

    #first file
    #google
    google_stats_first = TextComp(google_first, original_first)
    print("[google_first_2019] Word Error Rate:"+ str(google_stats_first.WER(False)))
    print("[google_first_2019] Accuracy:"+str(google_stats_first.Accuracy()))
    google_accuracy_arr.append(google_stats_first.Accuracy())
    #clean
    latest_deep_stats_first = TextComp(latest_deep_first, original_first_deep)
    print("[clean_deepspeech_first_2019] Word Error Rate:"+ str(latest_deep_stats_first.WER(debug)))
    print("[clean_deepspeech_first_2019] Accuracy:"+str(latest_deep_stats_first.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_first.Accuracy())
    #cafe-low
    latest_deep_stats_first_cafe_low = TextComp(latest_deep_first_cafe_low, original_first_deep)
    print("[cafe_low_deepspeech_first_2019] Word Error Rate:"+ str(latest_deep_stats_first_cafe_low.WER(debug)))
    print("[cafe_low_deepspeech_first_2019] Accuracy:"+str(latest_deep_stats_first_cafe_low.Accuracy()))
    cafe_low_deep_accuracy_arr.append(latest_deep_stats_first_cafe_low.Accuracy())
	#cafe-medium
    latest_deep_stats_first_cafe_medium = TextComp(latest_deep_first_cafe_medium, original_first_deep)
    print("[cafe_medium_deepspeech_first_2019] Word Error Rate:"+ str(latest_deep_stats_first_cafe_medium.WER(debug)))
    print("[cafe_medium_deepspeech_first_2019] Accuracy:"+str(latest_deep_stats_first_cafe_medium.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(latest_deep_stats_first_cafe_medium.Accuracy())
    #cafe-high
    latest_deep_stats_first_cafe_high = TextComp(latest_deep_first_cafe_high, original_first_deep)
    print("[cafe_high_deepspeech_first_2019] Word Error Rate:"+ str(latest_deep_stats_first_cafe_high.WER(debug)))
    print("[cafe_high_deepspeech_first_2019] Accuracy:"+str(latest_deep_stats_first_cafe_high.Accuracy()))
    cafe_high_deep_accuracy_arr.append(latest_deep_stats_first_cafe_high.Accuracy())
    #people-low
    latest_deep_stats_first_people_low = TextComp(latest_deep_first_people_low, original_first_deep)
    print("[people_low_deepspeech_first_2019] Word Error Rate:"+ str(latest_deep_stats_first_people_low.WER(debug)))
    print("[people_low_deepspeech_first_2019] Accuracy:"+str(latest_deep_stats_first_people_low.Accuracy()))
    people_low_deep_accuracy_arr.append(latest_deep_stats_first_people_low.Accuracy())
    #people-medium
    latest_deep_stats_first_people_medium = TextComp(latest_deep_first_people_medium, original_first_deep)
    print("_people_medium_deepspeech_first_2019] Word Error Rate:"+ str(latest_deep_stats_first_people_medium.WER(debug)))
    print("[people_medium_deepspeech_first_2019] Accuracy:"+str(latest_deep_stats_first_people_medium.Accuracy()))
    people_medium_deep_accuracy_arr.append(latest_deep_stats_first_people_medium.Accuracy())
	#people-high
    latest_deep_stats_first_people_high = TextComp(latest_deep_first_people_high, original_first_deep)
    print("[people_high_deepspeech_first_2019] Word Error Rate:"+ str(latest_deep_stats_first_people_high.WER(debug)))
    print("[people_high_deepspeech_first_2019] Accuracy:"+str(latest_deep_stats_first_people_high.Accuracy()))
    people_high_deep_accuracy_arr.append(latest_deep_stats_first_people_high.Accuracy())
    #sirens-low
    latest_deep_stats_first_sirens_low = TextComp(latest_deep_first_sirens_low, original_first_deep)
    print("[sirens_low_deepspeech_first_2019] Word Error Rate:"+ str(latest_deep_stats_first_sirens_low.WER(debug)))
    print("[sirens_low_deepspeech_first_2019] Accuracy:"+str(latest_deep_stats_first_sirens_low.Accuracy()))
    sirens_low_deep_accuracy_arr.append(latest_deep_stats_first_sirens_low.Accuracy())
    #sirens-medium
    latest_deep_stats_first_sirens_medium = TextComp(latest_deep_first_sirens_medium, original_first_deep)
    print("[sirens_medium_deepspeech_first_2019] Word Error Rate:"+ str(latest_deep_stats_first_sirens_medium.WER(debug)))
    print("[sirens_medium_deepspeech_first_2019] Accuracy:"+str(latest_deep_stats_first_sirens_medium.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(latest_deep_stats_first_sirens_medium.Accuracy())
	#sirens-high
    latest_deep_stats_first_sirens_high = TextComp(latest_deep_first_sirens_high, original_first_deep)
    print("[sirens_high_deepspeech_first_2019] Word Error Rate:"+ str(latest_deep_stats_first_sirens_high.WER(debug)))
    print("[sirens_high_deepspeech_first_2019] Accuracy:"+str(latest_deep_stats_first_sirens_high.Accuracy()))
    sirens_high_deep_accuracy_arr.append(latest_deep_stats_first_sirens_high.Accuracy())
	#fifth file
    #google
    google_stats_fifth = TextComp(google_fifth, original_fifth)
    print("[google_fifth_2019] Word Error Rate:"+ str(google_stats_fifth.WER(False)))
    print("[google_fifth_2019] Accuracy:"+str(google_stats_fifth.Accuracy()))
    google_accuracy_arr.append(google_stats_fifth.Accuracy())

    #clean
    latest_deep_stats_fifth = TextComp(latest_deep_fifth, original_fifth_deep)
    print("[clean_deepspeech_fifth_2019] Word Error Rate:"+ str(latest_deep_stats_fifth.WER(debug)))
    print("[clean_deepspeech_fifth_2019] Accuracy:"+str(latest_deep_stats_fifth.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_fifth.Accuracy())
	#cafe-low
    latest_deep_stats_fifth_cafe_low = TextComp(latest_deep_fifth_cafe_low, original_fifth_deep)
    print("[cafe_low_deepspeech_fifth_2019] Word Error Rate:"+ str(latest_deep_stats_fifth_cafe_low.WER(debug)))
    print("[cafe_low_deepspeech_fifth_2019] Accuracy:"+str(latest_deep_stats_fifth_cafe_low.Accuracy()))
    cafe_low_deep_accuracy_arr.append(latest_deep_stats_fifth_cafe_low.Accuracy())
	#cafe-medium
    latest_deep_stats_fifth_cafe_medium = TextComp(latest_deep_fifth_cafe_medium, original_fifth_deep)
    print("[cafe_medium_deepspeech_fifth_2019] Word Error Rate:"+ str(latest_deep_stats_fifth_cafe_medium.WER(debug)))
    print("[cafe_medium_deepspeech_fifth_2019] Accuracy:"+str(latest_deep_stats_fifth_cafe_medium.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(latest_deep_stats_fifth_cafe_medium.Accuracy())
    #cafe-high
    latest_deep_stats_fifth_cafe_high = TextComp(latest_deep_fifth_cafe_high, original_fifth_deep)
    print("[cafe_high_deepspeech_fifth_2019] Word Error Rate:"+ str(latest_deep_stats_fifth_cafe_high.WER(debug)))
    print("[cafe_high_deepspeech_fifth_2019] Accuracy:"+str(latest_deep_stats_fifth_cafe_high.Accuracy()))
    cafe_high_deep_accuracy_arr.append(latest_deep_stats_fifth_cafe_high.Accuracy())
    #people-low
    latest_deep_stats_fifth_people_low = TextComp(latest_deep_fifth_people_low, original_fifth_deep)
    print("[people_low_deepspeech_fifth_2019] Word Error Rate:"+ str(latest_deep_stats_fifth_people_low.WER(debug)))
    print("[people_low_deepspeech_fifth_2019] Accuracy:"+str(latest_deep_stats_fifth_people_low.Accuracy()))
    people_low_deep_accuracy_arr.append(latest_deep_stats_fifth_people_low.Accuracy())
    #people-medium
    latest_deep_stats_fifth_people_medium = TextComp(latest_deep_fifth_people_medium, original_fifth_deep)
    print("[people_medium_deepspeech_fifth_2019] Word Error Rate:"+ str(latest_deep_stats_fifth_people_medium.WER(debug)))
    print("[people_medium_deepspeech_fifth_2019] Accuracy:"+str(latest_deep_stats_fifth_people_medium.Accuracy()))
    people_medium_deep_accuracy_arr.append(latest_deep_stats_fifth_people_medium.Accuracy())
	#people-high
    latest_deep_stats_fifth_people_high = TextComp(latest_deep_fifth_people_high, original_fifth_deep)
    print("[people_high_deepspeech_fifth_2019] Word Error Rate:"+ str(latest_deep_stats_fifth_people_high.WER(debug)))
    print("[people_high_deepspeech_fifth_2019] Accuracy:"+str(latest_deep_stats_fifth_people_high.Accuracy()))
    people_high_deep_accuracy_arr.append(latest_deep_stats_fifth_people_high.Accuracy())
    #sirens-low
    latest_deep_stats_fifth_sirens_low = TextComp(latest_deep_fifth_sirens_low, original_fifth_deep)
    print("[sirens_low_deepspeech_fifth_2019] Word Error Rate:"+ str(latest_deep_stats_fifth_sirens_low.WER(debug)))
    print("[sirens_low_deepspeech_fifth_2019] Accuracy:"+str(latest_deep_stats_fifth_sirens_low.Accuracy()))
    sirens_low_deep_accuracy_arr.append(latest_deep_stats_fifth_sirens_low.Accuracy())
    #sirens-medium
    latest_deep_stats_fifth_sirens_medium = TextComp(latest_deep_fifth_sirens_medium, original_fifth_deep)
    print("[sirens_medium_deepspeech_fifth_2019] Word Error Rate:"+ str(latest_deep_stats_fifth_sirens_medium.WER(debug)))
    print("[sirens_medium_deepspeech_fifth_2019] Accuracy:"+str(latest_deep_stats_fifth_sirens_medium.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(latest_deep_stats_fifth_sirens_medium.Accuracy())
	#sirens-high
    latest_deep_stats_fifth_sirens_high = TextComp(latest_deep_fifth_sirens_high, original_fifth_deep)
    print("[sirens_high_deepspeech_fifth_2019] Word Error Rate:"+ str(latest_deep_stats_fifth_sirens_high.WER(debug)))
    print("[sirens_high_deepspeech_fifth_2019] Accuracy:"+str(latest_deep_stats_fifth_sirens_high.Accuracy()))
    sirens_high_deep_accuracy_arr.append(latest_deep_stats_fifth_sirens_high.Accuracy())

    #sixth file
    #google
    google_stats_sixth = TextComp(google_sixth, original_sixth)
    print("[google_sixth_2019] Word Error Rate:"+ str(google_stats_sixth.WER(False)))
    print("[google_sixth_2019] Accuracy:"+str(google_stats_sixth.Accuracy()))
    google_accuracy_arr.append(google_stats_sixth.Accuracy())

    #clean
    latest_deep_stats_sixth = TextComp(latest_deep_sixth, original_sixth_deep)
    print("[clean_deepspeech_sixth_2019] Word Error Rate:"+ str(latest_deep_stats_sixth.WER(debug)))
    print("[clean_deepspeech_sixth_2019] Accuracy:"+str(latest_deep_stats_sixth.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_sixth.Accuracy())
	#cafe-low
    latest_deep_stats_sixth_cafe_low = TextComp(latest_deep_sixth_cafe_low, original_sixth_deep)
    print("[cafe_low_deepspeech_sixth_2019] Word Error Rate:"+ str(latest_deep_stats_sixth_cafe_low.WER(debug)))
    print("[cafe_low_deepspeech_sixth_2019] Accuracy:"+str(latest_deep_stats_sixth_cafe_low.Accuracy()))
    cafe_low_deep_accuracy_arr.append(latest_deep_stats_sixth_cafe_low.Accuracy())
	#cafe-medium
    latest_deep_stats_sixth_cafe_medium = TextComp(latest_deep_sixth_cafe_medium, original_sixth_deep)
    print("[cafe_medium_deepspeech_sixth_2019] Word Error Rate:"+ str(latest_deep_stats_sixth_cafe_medium.WER(debug)))
    print("[cafe_medium_deepspeech_sixth_2019] Accuracy:"+str(latest_deep_stats_sixth_cafe_medium.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(latest_deep_stats_sixth_cafe_medium.Accuracy())
    #cafe-high
    latest_deep_stats_sixth_cafe_high = TextComp(latest_deep_sixth_cafe_high, original_sixth_deep)
    print("[cafe_high_deepspeech_sixth_2019] Word Error Rate:"+ str(latest_deep_stats_sixth_cafe_high.WER(debug)))
    print("[cafe_high_deepspeech_sixth_2019] Accuracy:"+str(latest_deep_stats_sixth_cafe_high.Accuracy()))
    cafe_high_deep_accuracy_arr.append(latest_deep_stats_sixth_cafe_high.Accuracy())
    #people-low
    latest_deep_stats_sixth_people_low = TextComp(latest_deep_sixth_people_low, original_sixth_deep)
    print("[people_low_deepspeech_sixth_2019] Word Error Rate:"+ str(latest_deep_stats_sixth_people_low.WER(debug)))
    print("[people_low_deepspeech_sixth_2019] Accuracy:"+str(latest_deep_stats_sixth_people_low.Accuracy()))
    people_low_deep_accuracy_arr.append(latest_deep_stats_sixth_people_low.Accuracy())
    #people-medium
    latest_deep_stats_sixth_people_medium = TextComp(latest_deep_sixth_people_medium, original_sixth_deep)
    print("[people_medium_deepspeech_sixth_2019] Word Error Rate:"+ str(latest_deep_stats_sixth_people_medium.WER(debug)))
    print("[people_medium_deepspeech_sixth_2019] Accuracy:"+str(latest_deep_stats_sixth_people_medium.Accuracy()))
    people_medium_deep_accuracy_arr.append(latest_deep_stats_sixth_people_medium.Accuracy())
	#people-high
    latest_deep_stats_sixth_people_high = TextComp(latest_deep_sixth_people_high, original_sixth_deep)
    print("[people_high_deepspeech_sixth_2019] Word Error Rate:"+ str(latest_deep_stats_sixth_people_high.WER(debug)))
    print("[people_high_deepspeech_sixth_2019] Accuracy:"+str(latest_deep_stats_sixth_people_high.Accuracy()))
    people_high_deep_accuracy_arr.append(latest_deep_stats_sixth_people_high.Accuracy())
    #sirens-low
    latest_deep_stats_sixth_sirens_low = TextComp(latest_deep_sixth_sirens_low, original_sixth_deep)
    print("[sirens_low_deepspeech_sixth_2019] Word Error Rate:"+ str(latest_deep_stats_sixth_sirens_low.WER(debug)))
    print("[sirens_low_deepspeech_sixth_2019] Accuracy:"+str(latest_deep_stats_sixth_sirens_low.Accuracy()))
    sirens_low_deep_accuracy_arr.append(latest_deep_stats_sixth_sirens_low.Accuracy())
    #sirens-medium
    latest_deep_stats_sixth_sirens_medium = TextComp(latest_deep_sixth_sirens_medium, original_sixth_deep)
    print("[sirens_medium_deepspeech_sixth_2019] Word Error Rate:"+ str(latest_deep_stats_sixth_sirens_medium.WER(debug)))
    print("[sirens_medium_deepspeech_sixth_2019] Accuracy:"+str(latest_deep_stats_sixth_sirens_medium.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(latest_deep_stats_sixth_sirens_medium.Accuracy())
	#sirens-high
    latest_deep_stats_sixth_sirens_high = TextComp(latest_deep_sixth_sirens_high, original_sixth_deep)
    print("[sirens_high_deepspeech_sixth_2019] Word Error Rate:"+ str(latest_deep_stats_sixth_sirens_high.WER(debug)))
    print("[sirens_high_deepspeech_sixth_2019] Accuracy:"+str(latest_deep_stats_sixth_sirens_high.Accuracy()))
    sirens_high_deep_accuracy_arr.append(latest_deep_stats_sixth_sirens_high.Accuracy())

    #seventh file
    #google
    google_stats_seventh = TextComp(google_seventh, original_seventh)
    print("[google_seventh_2019] Word Error Rate:"+ str(google_stats_seventh.WER(False)))
    print("[google_seventh_2019] Accuracy:"+str(google_stats_seventh.Accuracy()))
    google_accuracy_arr.append(google_stats_seventh.Accuracy())
    #clean
    latest_deep_stats_seventh = TextComp(latest_deep_seventh, original_seventh_deep)
    print("[clean_deepspeech_seventh_2019] Word Error Rate:"+ str(latest_deep_stats_seventh.WER(debug)))
    print("[clean_deepspeech_seventh_2019] Accuracy:"+str(latest_deep_stats_seventh.Accuracy()))
    clean_deep_accuracy_arr.append(latest_deep_stats_seventh.Accuracy())
	#cafe-low
    latest_deep_stats_seventh_cafe_low = TextComp(latest_deep_seventh_cafe_low, original_seventh_deep)
    print("[cafe_low_deepspeech_seventh_2019] Word Error Rate:"+ str(latest_deep_stats_seventh_cafe_low.WER(debug)))
    print("[cafe_low_deepspeech_seventh_2019] Accuracy:"+str(latest_deep_stats_seventh_cafe_low.Accuracy()))
    cafe_low_deep_accuracy_arr.append(latest_deep_stats_seventh_cafe_low.Accuracy())
	#cafe-medium
    latest_deep_stats_seventh_cafe_medium = TextComp(latest_deep_seventh_cafe_medium, original_seventh_deep)
    print("[cafe_medium_deepspeech_seventh_2019] Word Error Rate:"+ str(latest_deep_stats_seventh_cafe_medium.WER(debug)))
    print("[cafe_medium_deepspeech_seventh_2019] Accuracy:"+str(latest_deep_stats_seventh_cafe_medium.Accuracy()))
    cafe_medium_deep_accuracy_arr.append(latest_deep_stats_seventh_cafe_medium.Accuracy())
    #cafe-high
    latest_deep_stats_seventh_cafe_high = TextComp(latest_deep_seventh_cafe_high, original_seventh_deep)
    print("[cafe_high_deepspeech_seventh_2019] Word Error Rate:"+ str(latest_deep_stats_seventh_cafe_high.WER(debug)))
    print("[cafe_high_deepspeech_seventh_2019] Accuracy:"+str(latest_deep_stats_seventh_cafe_high.Accuracy()))
    cafe_high_deep_accuracy_arr.append(latest_deep_stats_seventh_cafe_high.Accuracy())
    #people-low
    latest_deep_stats_seventh_people_low = TextComp(latest_deep_seventh_people_low, original_seventh_deep)
    print("[people_low_deepspeech_seventh_2019] Word Error Rate:"+ str(latest_deep_stats_seventh_people_low.WER(debug)))
    print("[people_low_deepspeech_seventh_2019] Accuracy:"+str(latest_deep_stats_seventh_people_low.Accuracy()))
    people_low_deep_accuracy_arr.append(latest_deep_stats_seventh_people_low.Accuracy())
    #people-medium
    latest_deep_stats_seventh_people_medium = TextComp(latest_deep_seventh_people_medium, original_seventh_deep)
    print("[people_medium_deepspeech_seventh_2019] Word Error Rate:"+ str(latest_deep_stats_seventh_people_medium.WER(debug)))
    print("[people_medium_deepspeech_seventh_2019] Accuracy:"+str(latest_deep_stats_seventh_people_medium.Accuracy()))
    people_medium_deep_accuracy_arr.append(latest_deep_stats_seventh_people_medium.Accuracy())
	#people-high
    latest_deep_stats_seventh_people_high = TextComp(latest_deep_seventh_people_high, original_seventh_deep)
    print("[people_high_deepspeech_seventh_2019] Word Error Rate:"+ str(latest_deep_stats_seventh_people_high.WER(debug)))
    print("[people_high_deepspeech_seventh_2019] Accuracy:"+str(latest_deep_stats_seventh_people_high.Accuracy()))
    people_high_deep_accuracy_arr.append(latest_deep_stats_seventh_people_high.Accuracy())
    #sirens-low
    latest_deep_stats_seventh_sirens_low = TextComp(latest_deep_seventh_sirens_low, original_seventh_deep)
    print("[sirens_low_deepspeech_seventh_2019] Word Error Rate:"+ str(latest_deep_stats_seventh_sirens_low.WER(debug)))
    print("[sirens_low_deepspeech_seventh_2019] Accuracy:"+str(latest_deep_stats_seventh_sirens_low.Accuracy()))
    sirens_low_deep_accuracy_arr.append(latest_deep_stats_seventh_sirens_low.Accuracy())
    #sirens-medium
    latest_deep_stats_seventh_sirens_medium = TextComp(latest_deep_seventh_sirens_medium, original_seventh_deep)
    print("[sirens_medium_deepspeech_seventh_2019] Word Error Rate:"+ str(latest_deep_stats_seventh_sirens_medium.WER(debug)))
    print("[sirens_medium_deepspeech_seventh_2019] Accuracy:"+str(latest_deep_stats_seventh_sirens_medium.Accuracy()))
    sirens_medium_deep_accuracy_arr.append(latest_deep_stats_seventh_sirens_medium.Accuracy())
	#sirens-high
    latest_deep_stats_seventh_sirens_high = TextComp(latest_deep_seventh_sirens_high, original_seventh_deep)
    print("[sirens_high_deepspeech_seventh_2019] Word Error Rate:"+ str(latest_deep_stats_seventh_sirens_high.WER(debug)))
    print("[sirens_high_deepspeech_seventh_2019] Accuracy:"+str(latest_deep_stats_seventh_sirens_high.Accuracy()))
    sirens_high_deep_accuracy_arr.append(latest_deep_stats_seventh_sirens_high.Accuracy())

	#check if every array has the rigt number of elements
    # print(google_accuracy_arr)
    # print(clean_deep_accuracy_arr)
    # print(cafe_low_deep_accuracy_arr)
    # print(cafe_medium_deep_accuracy_arr)
    # print(cafe_high_deep_accuracy_arr)
    # print(people_low_deep_accuracy_arr)
    # print(people_medium_deep_accuracy_arr)
    # print(people_high_deep_accuracy_arr)
    # print(sirens_low_deep_accuracy_arr)
    # print(sirens_medium_deep_accuracy_arr)
    # print(sirens_high_deep_accuracy_arr)

	#cafe data comparison arrays
    cafe_arr_first = []
    cafe_arr_first.append(google_accuracy_arr[0])
    cafe_arr_first.append(clean_deep_accuracy_arr[0])
    cafe_arr_first.append(cafe_low_deep_accuracy_arr[0])
    cafe_arr_first.append(cafe_medium_deep_accuracy_arr[0])
    cafe_arr_first.append(cafe_high_deep_accuracy_arr[0])

    cafe_arr_second = []
    cafe_arr_second.append(google_accuracy_arr[1])
    cafe_arr_second.append(clean_deep_accuracy_arr[1])
    cafe_arr_second.append(cafe_low_deep_accuracy_arr[1])
    cafe_arr_second.append(cafe_medium_deep_accuracy_arr[1])
    cafe_arr_second.append(cafe_high_deep_accuracy_arr[1])

    cafe_arr_third = []
    cafe_arr_third.append(google_accuracy_arr[2])
    cafe_arr_third.append(clean_deep_accuracy_arr[2])
    cafe_arr_third.append(cafe_low_deep_accuracy_arr[2])
    cafe_arr_third.append(cafe_medium_deep_accuracy_arr[2])
    cafe_arr_third.append(cafe_high_deep_accuracy_arr[2])

    cafe_arr_fourth = []
    cafe_arr_fourth.append(google_accuracy_arr[3])
    cafe_arr_fourth.append(clean_deep_accuracy_arr[3])
    cafe_arr_fourth.append(cafe_low_deep_accuracy_arr[3])
    cafe_arr_fourth.append(cafe_medium_deep_accuracy_arr[3])
    cafe_arr_fourth.append(cafe_high_deep_accuracy_arr[3])

	#people data comparison arrays
    people_arr_first = []
    people_arr_first.append(google_accuracy_arr[0])
    people_arr_first.append(clean_deep_accuracy_arr[0])
    people_arr_first.append(people_low_deep_accuracy_arr[0])
    people_arr_first.append(people_medium_deep_accuracy_arr[0])
    people_arr_first.append(people_high_deep_accuracy_arr[0])

    people_arr_second = []
    people_arr_second.append(google_accuracy_arr[1])
    people_arr_second.append(clean_deep_accuracy_arr[1])
    people_arr_second.append(people_low_deep_accuracy_arr[1])
    people_arr_second.append(people_medium_deep_accuracy_arr[1])
    people_arr_second.append(people_high_deep_accuracy_arr[1])

    people_arr_third = []
    people_arr_third.append(google_accuracy_arr[2])
    people_arr_third.append(clean_deep_accuracy_arr[2])
    people_arr_third.append(people_low_deep_accuracy_arr[2])
    people_arr_third.append(people_medium_deep_accuracy_arr[2])
    people_arr_third.append(people_high_deep_accuracy_arr[2])

    people_arr_fourth = []
    people_arr_fourth.append(google_accuracy_arr[3])
    people_arr_fourth.append(clean_deep_accuracy_arr[3])
    people_arr_fourth.append(people_low_deep_accuracy_arr[3])
    people_arr_fourth.append(people_medium_deep_accuracy_arr[3])
    people_arr_fourth.append(people_high_deep_accuracy_arr[3])

	#sirens data comparison arrays
    sirens_arr_first = []
    sirens_arr_first.append(google_accuracy_arr[0])
    sirens_arr_first.append(clean_deep_accuracy_arr[0])
    sirens_arr_first.append(sirens_low_deep_accuracy_arr[0])
    sirens_arr_first.append(sirens_medium_deep_accuracy_arr[0])
    sirens_arr_first.append(sirens_high_deep_accuracy_arr[0])

    sirens_arr_second = []
    sirens_arr_second.append(google_accuracy_arr[1])
    sirens_arr_second.append(clean_deep_accuracy_arr[1])
    sirens_arr_second.append(sirens_low_deep_accuracy_arr[1])
    sirens_arr_second.append(sirens_medium_deep_accuracy_arr[1])
    sirens_arr_second.append(sirens_high_deep_accuracy_arr[1])

    sirens_arr_third = []
    sirens_arr_third.append(google_accuracy_arr[2])
    sirens_arr_third.append(clean_deep_accuracy_arr[2])
    sirens_arr_third.append(sirens_low_deep_accuracy_arr[2])
    sirens_arr_third.append(sirens_medium_deep_accuracy_arr[2])
    sirens_arr_third.append(sirens_high_deep_accuracy_arr[2])

    sirens_arr_fourth = []
    sirens_arr_fourth.append(google_accuracy_arr[3])
    sirens_arr_fourth.append(clean_deep_accuracy_arr[3])
    sirens_arr_fourth.append(sirens_low_deep_accuracy_arr[3])
    sirens_arr_fourth.append(sirens_medium_deep_accuracy_arr[3])
    sirens_arr_fourth.append(sirens_high_deep_accuracy_arr[3])
	#data visualization
	#labels for average graph
    labels = ('Cafeteria','People Talking','Sirens')
	#labels for three sub graphs
    # labels = ('Google Speech','Clean','Low Noise','Medium Noise','High Noise')

	#average accuracy under different noise arrays

    clean_ave = 0
    for val in clean_deep_accuracy_arr:
    	clean_ave += val
    clean_ave = clean_ave/len(clean_deep_accuracy_arr)

    cafe_low_ave = 0
    for val in cafe_low_deep_accuracy_arr:
    	cafe_low_ave += val
    cafe_low_ave = cafe_low_ave/len(cafe_low_deep_accuracy_arr)

    people_low_ave = 0
    for val in people_low_deep_accuracy_arr:
    	people_low_ave += val
    people_low_ave = people_low_ave/len(people_low_deep_accuracy_arr)

    sirens_low_ave = 0
    for val in sirens_low_deep_accuracy_arr:
    	sirens_low_ave += val
    sirens_low_ave = sirens_low_ave/len(sirens_low_deep_accuracy_arr)

    cafe_medium_ave = 0
    for val in cafe_medium_deep_accuracy_arr:
    	cafe_medium_ave += val
    cafe_medium_ave = cafe_medium_ave/len(cafe_medium_deep_accuracy_arr)

    people_medium_ave = 0
    for val in people_medium_deep_accuracy_arr:
    	people_medium_ave += val
    people_medium_ave = people_medium_ave/len(people_medium_deep_accuracy_arr)

    sirens_medium_ave = 0
    for val in sirens_medium_deep_accuracy_arr:
    	sirens_medium_ave += val
    sirens_medium_ave = sirens_medium_ave/len(sirens_medium_deep_accuracy_arr)

    cafe_high_ave = 0
    for val in cafe_high_deep_accuracy_arr:
    	cafe_high_ave += val
    cafe_high_ave = cafe_high_ave/len(cafe_high_deep_accuracy_arr)

    people_high_ave = 0
    for val in people_high_deep_accuracy_arr:
    	people_high_ave += val
    people_high_ave = people_high_ave/len(people_high_deep_accuracy_arr)

    sirens_high_ave = 0
    for val in sirens_high_deep_accuracy_arr:
    	sirens_high_ave += val
    sirens_high_ave = sirens_high_ave/len(sirens_high_deep_accuracy_arr)

    clean_arr = []
    clean_arr.append(clean_ave)
    clean_arr.append(clean_ave)
    clean_arr.append(clean_ave)

    low_arr = []
    low_arr.append(cafe_low_ave)
    low_arr.append(people_low_ave)
    low_arr.append(sirens_low_ave)

    medium_arr = []
    medium_arr.append(cafe_medium_ave)
    medium_arr.append(people_medium_ave)
    medium_arr.append(sirens_medium_ave)

    high_arr = []
    high_arr.append(cafe_high_ave)
    high_arr.append(people_high_ave)
    high_arr.append(sirens_high_ave)

	#visualization for cafeteria
    # df = pd.DataFrame(np.c_[cafe_arr_first,cafe_arr_second,cafe_arr_third,cafe_arr_fourth], index=labels, columns=['Paramedic Smith','EMT 107','EMT 117','EMT 101'])
	#
    # ax = df.plot.bar()
    # # ax.set_xlabel("audio file")
    # ax.set_ylabel("Accuracy")
    # plt.xticks(rotation=360)
    # plt.suptitle("Cafeteria Noise Figure")
    # plt.show()

	# #visualization for people noise
    # df = pd.DataFrame(np.c_[people_arr_first,people_arr_second,people_arr_third,people_arr_fourth], index=labels, columns=['Paramedic Smith','EMT 107','EMT 117','EMT 101'])
	#
    # ax = df.plot.bar()
    # # ax.set_xlabel("audio file")
    # ax.set_ylabel("Accuracy")
    # plt.xticks(rotation=360)
    # plt.suptitle("People Talking")
    # plt.show()

	#visualication for Sirens Noise
    # df = pd.DataFrame(np.c_[sirens_arr_first,sirens_arr_second,sirens_arr_third,sirens_arr_fourth], index=labels, columns=['Paramedic Smith','EMT 107','EMT 117','EMT 101'])
	#
    # ax = df.plot.bar()
    # # ax.set_xlabel("audio file")
    # ax.set_ylabel("Accuracy")
    # plt.xticks(rotation=360)
    # plt.suptitle("Sirens Noise")
    # plt.show()

	#visualiczation for Average accyracy under dfferent noise profiles
    df = pd.DataFrame(np.c_[clean_arr,low_arr,medium_arr,high_arr], index = labels, columns = ['Clean','Low Noise','Medium Noise','High Noise'])
    ax = df.plot.bar()
    # ax.set_xlabel("audio file")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=360)
    plt.suptitle("Average accuracy under different noise profiles")
    plt.show()
