## import modules here

########## Question 1 ##########
# do not change the heading of the function
def count(j,hashes,offset):
    result = map(lambda x: abs(x[0] - x[1])<=offset, zip(j, hashes))
    return sum(result)


def c2lsh(data_hashes, query_hashes, alpha_m, beta_n):
    offset=1
    max_reached=False #This will ensure to search offset only in between Offset_min and Offset_Max once we got candidates > Beta_n
    offset_min = 0
    offset_max = 0
    while True:
        data_hashes_2=data_hashes.filter(lambda j : count(j[1],query_hashes,offset)>=alpha_m)
        cand= data_hashes_2.count()
        if cand<beta_n and not max_reached:
            offset_min=offset
            offset= offset *2
            offset_max = offset    
        elif cand < beta_n and offset_max-offset_min ==1 :
            offset += 1
        elif cand <beta_n and max_reached:
            offset_min=offset
            offset=int((offset_max+offset_min)/2)
        elif cand == beta_n or (cand > beta_n and offset_max-offset_min <=1):
            return data_hashes_2.keys()
        elif cand > beta_n :
                max_reached=True
                offset_max=offset
                offset=int((offset+offset_min)/2)
                
    return None