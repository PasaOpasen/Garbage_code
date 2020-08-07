

library(tidyverse)
library(readxl)

remains = read_xlsx('remains.xlsx')
colnames(remains) = c('id', 'size', 'free_count', 'self_cost', 'sell_cost')


sells = read_xlsx('sells.xlsx')
colnames(sells) = c('id_old', 'size', 'month', 'sell_count')


# prepare ids

new.id = sells$id_old

for(index in seq_along(new.id)){
  
  id = new.id[index]
  
  if(grepl('/', id)){
    pair = strsplit(id, '/')[[1]]
    
    if(nchar(pair[1])==12){
      id = paste0(pair[1],pair[2])
    }else if(nchar(pair[1])==14){
      
      left12 = substr(pair[1],1,12)
      last2 = substr(pair[1],13,14)
      right = substr(pair[2],1,2)
      
      id = paste0(left12, right, last2)
      
    }else{
      id = pair[1]
    }
    
  }
  
  new.id[index] = id
}


sells$id_old = new.id
colnames(sells)[1] = 'id'

remains$id = str_remove(remains$id, ' ')



# error objects


sell_unique = unique(sells$id)
remains_unique = unique(remains$id)

diffs = setdiff(sell_unique, remains_unique)

sizes = sapply(diffs, function(d) sells[sells$id == d,2][[1]][[1]] )

remains = rbind(remains, 
                tibble(id = diffs, 
                       size = sizes,
                       free_count = 0, self_cost = 200, sell_cost = 1000 ))















