def string_parse(packege:str):
    packege = packege.split('-')
    packege[1] = int(packege[1])
    return packege


def packege_sort(packeges_input:list):
    if len(packeges_input) == 0 :
        return "Invalid Configuration"
    try:
        packeges = [string_parse(i) for i in packeges_input]
    except:
        return "Invalid Configuration"
    
    packeges.sort(key = lambda x:x[1])

    packeges_weight = [packege[1] for packege in packeges ]
    
    result_weight_diff = all(abs(packeges_weight[i] - packeges_weight[i+1]) <=5 for i in range(len(packeges_weight)))
    if result_weight_diff <=5:
        return [packege[0] for packege in packeges]
    else:
        return "Invalid Configuration"
    
    


print(packege_sort(['PKG4-5', 'PKG2-15', 'PKG3-25']))
print(packege_sort(['PKG1-10', 'PKG2-8', 'PKG3-14', 'PKG4-12']))