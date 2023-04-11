import requests
# Save file data to local copy
f = open("data/dainas2.txt", "r")
for x in f:
    filename = x[:-1]
    # Define the local filename to save data
    local_file = 'data/' + filename
    # Define the remote file to retrieve
    remote_url = 'http://dainas.aot.tu-berlin.de/~andreas@DAI/20210517__FAQ-Crawled/dat/' + filename
    print(remote_url)
    # Make http request for remote file data
    #data = requests.get(remote_url)
    #with open(local_file, 'wb') as file:
        #file.write(data.content)