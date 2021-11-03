


def find(N):

    lst = [i**5 for i in range(1, N+1)]

    # print(lst)

    st = set(lst)

    search_list = lst[:-1]
    max_elem = lst[-1]

    for ia, a in enumerate(search_list):
        for ib, b in enumerate(search_list[ia:]):
            
            sm_ab = a+b

            if sm_ab > max_elem:
                break

            for ic, c in enumerate(search_list[ia+ib:]):
                sm_abc = sm_ab + c

                if sm_abc > max_elem:
                    break

                for d in search_list[ia+ib+ic:]:
                    sm_abcb = sm_abc + d
                    
                    if sm_abcb > max_elem:
                        break

                    if sm_abcb in st:
                        print(a**0.2,b**0.2,c**0.2,d**0.2, (a+b+c+d)**0.2)
                        #return



def find2(N):
    
    lst = [i**5 for i in range(1, N+1)]

    # print(lst)

    st = set(lst)

    search_list = lst[:-1]
    max_elem = lst[-1]

    pairs = []
    sums = []
    for i, a in enumerate(search_list):
        max_b = max_elem - a
        for b in search_list[i:]:
            if b >= max_b:
                break
            pairs.append((a, b))
            # sums.append(a+b)
            # for c,d in pairs[:-1]:


    pairs.sort(key=lambda t: t[0]+t[1])


    for i, (a, b) in enumerate(pairs):
        sum_ab = a+b
        for c, d in pairs[i:]:

            sum_all = sum_ab + c + d

            if sum_all>max_elem:
                break
            if sum_all in st:
                print(a**0.2,b**0.2,c**0.2,d**0.2, (a+b+c+d)**0.2)






def find3(N):
    
    lst = [i**5 for i in range(1, N+1)]

    # print(lst)

    st = set(lst)

    search_list = lst[:-1]
    max_elem = lst[-1]


    pairs = [[]]
    for i, a in enumerate(search_list):
        max_b = max_elem - a
        for b in search_list[i:]:
            if b >= max_b:
                break

            pairs[-1].append((a, b))
            sum_ab = a+b

            max_cd = max_b - b

            for increase_pairs in pairs:
                if increase_pairs[0][0] >= max_cd:
                    continue

                #if increase_pairs[0][1] > a:
                #    continue

                for c, d in increase_pairs:
                    if d > a :
                        break
                    
                    sum_cd = c + d

                    if sum_cd > max_cd:
                        break

                    sum_all = sum_ab + sum_cd
                    if sum_all in st:
                        print(a**0.2,b**0.2,c**0.2,d**0.2, (a+b+c+d)**0.2)

        pairs.append([])




find(150)






