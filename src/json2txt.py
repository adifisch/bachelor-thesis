from os import walk
import os
import json

dataDir = 'data/unpack-faq/'
files = []
qa = []
errorFiles = []
errcnt = 0

for (root, dirnames, filenames) in walk(dataDir):
    files.extend(filenames)
    for file in files:
        # Opening JSON file
        filepath = os.path.join(root, file)
        try:
            json_file = open(filepath,'r')
            data = json.load(json_file)
            qa.append({
                'id': data['id'],
                'question': data['faqQuestionMain'],
                'answer': data['responses'][0]['messages'][0]['speech'].replace('\r\n','')
                })
        except:
            errorFiles.append(filepath)
            errcnt += 1
    qa = list({v['id']:v for v in qa}.values())

out_q = open("data/questions.txt", "a")
out_a = open("data/answers.txt", "a")
for qaPair in qa:
    out_q.write(qaPair['question'] + '\n\n')
    out_a.write(qaPair['answer'] + '\n\n')
out_q.close()
out_a.close()

out_e = open("data/errorfiles.txt", "a")
for faultyfile in errorFiles:
    out_e.write(faultyfile + '\n\n')
out_e.close()
    
        