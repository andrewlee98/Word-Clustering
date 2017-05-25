import time, random
from xgoogle.search import GoogleSearch, SearchError

f = open('a.txt','wb')

for i in range(0,2):
    wt = random.uniform(2, 5)
    gs = GoogleSearch("about")
    gs.results_per_page = 10
    gs.page = i
    results = gs.get_results()
    #Try not to annoy Google, with a random short wait
    time.sleep(wt)
    print('This is the %dth iteration and waited %f seconds') % (i, wt)
    for res in results:
        f.write(res.url.encode("utf8"))
        f.write("\n")

print("Done")
f.close()