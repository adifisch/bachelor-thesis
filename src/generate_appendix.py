labels = ''
predictions = ''

with open('appendix/labels.txt','r') as f:
    labels = f.readlines()
 
with open('appendix/predictions.txt','r') as f:
    predictions = f.readlines()


appendix = open('appendix/appendix.txt','w')
for i in range(len(labels)):
    appendix.write(labels[i].replace('\n','').replace('\t','').replace('\r','') + ' & ' + predictions[i].replace('\n','').replace('\t','').replace('\r','') + ' \\\\ \\hline \n')
