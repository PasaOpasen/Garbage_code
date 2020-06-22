
source('functions.r')


#' Title
#'
#' @param id.group     id columns in data frame with groups labels
#' @param group.labels vector of group labels
#' @param id.columns   vector containing id's of comparing columns 
#' @param type.columns vector containing types: "num","bin" or "cat" of comparing columns 
#' @param data         data frame 
#' @param PATH         path to write result
#' @param FILENAME     filename for result
#' @param dbg          0 - off, 1- onn debug mode 
#'
#' @return
#' @export
#'
#' @examples
cmp.2group.all = function(id.group, 
                          group.labels = NULL,
                          id.columns,
                          type.columns,
                          data, 
                          PATH = NULL, 
                          FILENAME = NULL, 
                          result = NULL,
                          dbg    = 1
)
{
    require(dplyr)
    
    #dt1 = data[data[[id.group]] == group.labels[1], ]
    #dt2 = data[data[[id.group]] == group.labels[2], ]
    
    result = data.frame( n.1         = character(),
                         group.1     = character(),
                         n.2         = character(),
                         group.2     = character(),
                         diff        = character(),
                         p.value     = character(),
                         method      = character()
    )
    colnames(result)[1] = group.labels[1]
    colnames(result)[2] = group.labels[2]
    
    for( i in 1:length(id.columns))
    {
        
        i.col  = id.columns[i]
        if (dbg == 1){
            cat("\nCurrent column | ", colnames(data)[i.col],"| id = ", i.col)
        }
        
        dt     = parse.data.frame.with.na( dt = data,
                                           groups.column = id.group,
                                           group.labels  = group.labels,
                                           values.column = i.col
        )
        
        # 1 - id.group
        # 2 - i.col
        x = dt[dt[[1]] == group.labels[1], ][,2]
        y = dt[dt[[1]] == group.labels[2], ][,2]
        n.x = na.count.string(x)
        n.y = na.count.string(y)        
        
        dt     = parse.data.frame(  dt            = data,
                                    groups.column = id.group,
                                    group.labels  = group.labels,
                                    values.column = i.col
        )
        # 1 - id.group
        # 2 - i.col
        x = dt[dt[[1]] == group.labels[1], ][,2]
        y = dt[dt[[1]] == group.labels[2], ][,2]
        
        if(dbg == 1){
            
            if(length(x) < 3 ) {
                cat ("WARNING in group: ", group.labels[1], "column:  ", colnames(data)[i.col], "\n" )
                cat ("first group too small")
                #stop("first group too small")
                x = NA
            }
            if(length(y) < 3 ) {
                cat ("WARNING in group: ", group.labels[2], "column:  ", colnames(data)[i.col],  "\n" )
                cat ("second group too small" )
                y = NA
                #stop("first group too small")
            }
        }
        
        if(type.columns[i] == "num")
        {
            if(
                (all(is.na(x)) == FALSE) &
                (all(is.na(y)) == FALSE)
            )
            {    
                #apply U-test Mana-Uitni
                res         = wilcox.test(x, y, paired=FALSE, conf.int = TRUE)
                res.stat    = round(res$statistic, 4)
                res.p.value = p_value_formatted(res$p.value)
                
                res.mu      = -round(res$estimate,  2)
                res.ci      = paste(c( '[', 
                                       as.character(-round(res$conf.int[2],2)),
                                       '; ',
                                       as.character(-round(res$conf.int[1],2)), 
                                       ']'), collapse=''
                )
                res.df = data.frame( n.1      = n.x,
                                     group.1  = paste0(as.character(median.my(x)),
                                                       " ",
                                                       quantile.interval.my(x),
                                                       " ",
                                                       as.character(round(mean(x, na.rm = TRUE), 2)),
                                                       "?",
                                                       as.character(round(sd(x, na.rm = TRUE), 2))
                                     ),
                                     n.2      = n.y,
                                     group.2  = paste0(as.character(median.my(y)),
                                                       " ",
                                                       quantile.interval.my(y),
                                                       " ",
                                                       as.character(round(mean(y, na.rm = TRUE), 2)),
                                                       "?",
                                                       as.character(round(sd(y, na.rm = TRUE), 2))
                                     ),
                                     diff     = paste0( as.character(res.mu)," ", res.ci),
                                     p.value     = as.character(res.p.value),
                                     method      = "U-test Mann-Whitney"
                )
            }
            if(
                (all(is.na(x)) == TRUE) & 
                (all(is.na(y)) == FALSE) 
            )
            {
                res.df = data.frame( n.1      = n.x,
                                     group.1  = "NA",
                                     n.2      = n.y,
                                     group.2  = paste0( as.character(median.my(y)),
                                                        " ",
                                                        quantile.interval.my(y),
                                                        " ",
                                                        as.character(round(mean(y, na.rm = TRUE), 2)),
                                                        "?",
                                                        as.character(round(sd(y, na.rm = TRUE), 2))
                                     ),
                                     diff     = "-",
                                     p.value  = "-",
                                     method      = "WARNING: ALL Na's values in 1st group"
                )
                
            }
            if(
                (all(is.na(x)) == FALSE) & 
                (all(is.na(y)) == TRUE) 
            )
            {
                res.df = data.frame(  n.1      = n.x,
                                      group.1  = paste0(as.character(median.my(x)),
                                                        " ",
                                                        quantile.interval.my(x),
                                                        " ",
                                                        as.character(round(mean(x, na.rm = TRUE), 2)),
                                                        "?",
                                                        as.character(round(sd(x, na.rm = TRUE), 2))
                                      ),
                                      n.2      = n.y,
                                      group.2  = "NA",
                                      diff     = "-",
                                      p.value  = "-",
                                      method      = "WARNING: ALL Na's values in 2nd group"
                )
                
            }
            
            if(
                (all(is.na(x)) == TRUE) & 
                (all(is.na(y)) == TRUE) 
            )
            {
                res.df = data.frame( n.1      = n.x,
                                     group.1  = "NA",
                                     n.2      = n.y,
                                     group.2  = "NA",
                                     diff     = "-",
                                     p.value  = "-",
                                     method   = "WARNING: ALL Na's values in both group"
                )
                
            }
            
            rownames(res.df)[1] = colnames(data)[i.col]
            colnames(res.df)[1] = group.labels[1]
            colnames(res.df)[2] = group.labels[2]
            result = rbind(result, res.df)
        }
        if(type.columns[i] == "bin")
        {
            # freq.matrix:
            # -----------------
            #        |  0  1
            # --------=--------
            # group1 |  7 33
            # group2 | 33 78
            #
            
            freq.matrix =  table(dt)
            nrow.f.m = nrow(freq.matrix)
            ncol.f.m = ncol(freq.matrix)
            
            if (dbg == 1) cat("\nfreq matrix         : ", freq.matrix)
            if (dbg == 1) cat("\ncolnames freq matrix: ", colnames(freq.matrix))
            if (dbg == 1) cat("\nrownames freq matrix: ", length(rownames(freq.matrix)),"\n")
            
            prcnt.1     = "NULL"
            prcnt.2     = "NULL"
            res.p.value = "NULL"
            m.name      = "NULL"
            res.diff    = "NULL"
            
            
            
            if(
                (ncol.f.m == 2) & 
                (nrow.f.m == 2) &
                (all(is.na(x)) == FALSE) &
                (all(is.na(y)) == FALSE)
            ) 
            {    
                res = fisher.test(freq.matrix)
                
                prcnt.1 = bin.my(x, digits=0)
                prcnt.2 = bin.my(y, digits=0)
                
                res.p.value = p_value_formatted(res$p.value)
                
                res.diff = ifelse(freq.matrix[1,2] == 0,
                                  "-",
                                  paste0( round(res$estimate,1)," [",
                                          round(res$conf.int[1],1),
                                          "; ",
                                          round(res$conf.int[2],1),
                                          "]"
                                  )
                )
                m.name = "Fisher test"
            } 
            if (
                (
                    (length(colnames(freq.matrix)) == 1) & 
                    (length(rownames(freq.matrix)) == 2) &
                    (colnames(freq.matrix)[1] == 0)&
                    (all(is.na(x)) == FALSE) &
                    (all(is.na(y)) == FALSE)
                )
                |
                (
                    (length(colnames(freq.matrix)) == 1) & 
                    (length(rownames(freq.matrix)) == 2) &
                    (colnames(freq.matrix)[1] == 1) &
                    (all(is.na(x)) == FALSE) &
                    (all(is.na(y)) == FALSE)
                )
                
            )
            {
                prcnt.1 = bin.my(x, digits=0)
                prcnt.2 = bin.my(y, digits=0)
                res.p.value = ">0.999"
                res.diff    = "-"
                m.name      = "WARNING: equal cases in both groups"
            }
            
            if(
                (all(is.na(x)) == TRUE) 
                |
                (all(is.na(y)) == TRUE)
            )
            {
                prcnt.1 = bin.my(x, digits=0)
                prcnt.2 = bin.my(y, digits=0)
                res.p.value = " - "
                res.diff    = " - "
                m.name      = "WARNING: All NA's !"
            }
            
            
            res.df = data.frame( n.1      = n.x,
                                 group.1  = prcnt.1,
                                 n.2      = n.y,
                                 group.2  = prcnt.2,
                                 diff     = res.diff,
                                 p.value  = as.character(res.p.value),
                                 method   = m.name
            )
            rownames(res.df)[1] = colnames(data)[i.col]
            colnames(res.df)[1] = group.labels[1]
            colnames(res.df)[2] = group.labels[2]
            result = rbind(result, res.df)
        }
        if(type.columns[i] == "cat")
        {
            
            if(
                (all(is.na(x)) == FALSE) &
                (all(is.na(y)) == FALSE)
            )
            {
                n.matrix    = table(dt)
                prop.matrix = round((prop.table(n.matrix, 1) * 100), 1)
                
                # the code below concatenates rows of n.matrix and prop.matrix in the following format:
                # (this is example of group.1 cell content in data.frame)
                #  ----------------
                # | 0 - 35 (31.2%) |
                # | 1 - 76 (67.9%) |
                # | 2 - 0 (0%)     |
                # | 3 - 1 (0.9%)   |
                #  ----------------
                # where 0, 1, 2, 3 - category names/IDs
                # 35, 76, 0, 1 number of peoples in each category
                # % - how much peoples (in percentages) in each category (100% - total number of peoples in group = 35+76+0+1 = 112)
                res1 = paste(paste0(colnames(n.matrix), " - ", as.numeric(n.matrix[1,]), " (", as.numeric(prop.matrix[1,]), "%)"), collapse = "\n")
                res2 = paste(paste0(colnames(n.matrix), " - ", as.numeric(n.matrix[2,]), " (", as.numeric(prop.matrix[2,]), "%)"), collapse = "\n")
                
                
                if (
                    dim(n.matrix)[2] > 1
                )
                {
                    res = fisher.test(n.matrix)
                    res.p.value = as.character(p_value_formatted(res$p.value))
                    res.method  = "Fisher test"
                }
                if (
                    dim(n.matrix)[2] == 1
                )
                {
                    res.p.value = " - "
                    res.method  = "WARNING: only one category!"
                }
                
            }
            
            if(
                (all(is.na(x)) == TRUE) &
                (all(is.na(y)) == FALSE)
            )
            {
                n.matrix    = table(y)
                prop.matrix = round((prop.table(n.matrix, 1) * 100), 1)
                
                res1 = " - "
                res2 = paste(paste0(names(n.matrix), " - ", as.numeric(n.matrix[1]), " (", as.numeric(prop.matrix[1]), "%)"), collapse = "\n")
                res.p.value = " - "
                res.method  = "WARNING: NA's values only in 1st group!"
                
                rownames(n.matrix)[1] = group.labels[1] # for right joining to result
            }
            if(
                (all(is.na(x)) == FALSE) &
                (all(is.na(y)) == TRUE)
            )
            {
                n.matrix    = table(x)
                prop.matrix = round((prop.table(n.matrix, 1) * 100), 1)
                
                res1 = paste(paste0(names(n.matrix), " - ", as.numeric(n.matrix[1]), " (", as.numeric(prop.matrix[1]), "%)"), collapse = "\n")
                res2 = " - "
                res.p.value = " - "
                res.method  = "WARNING: NA's values only in 2nd group!"
                
                rownames(n.matrix)[1] = group.labels[1] # for right joining to result
            }
            if(
                (all(is.na(x)) == TRUE) &
                (all(is.na(y)) == TRUE)
            )
            {
                res1 = " - "
                res2 = " - "
                res.p.value = " - "
                res.method  = "WARNING: NA's values only in both groups"
                
                n.matrix = NULL
                rownames(n.matrix)[1] = group.labels[1] # for right joining to result
                
            }
            
            res.df = data.frame( n.1      = n.x,
                                 group.1  = res1,
                                 n.2      = n.y,
                                 group.2  = res2,
                                 diff     = " ",
                                 p.value  = res.p.value,
                                 method   = res.method
            )
            rownames(res.df)[1] = colnames(data)[i.col]
            
            # n.matrix rownames are ordered in the same way as initial data, not order we set for group.labels !!!
            if (group.labels[1] == rownames(n.matrix)[1]) {
                colnames(res.df)[1] = group.labels[1]
                colnames(res.df)[2] = group.labels[2]
            } else {
                colnames(res.df)[1] = group.labels[2]
                colnames(res.df)[2] = group.labels[1]
            }
            
            result = rbind(result, res.df)
        }
        
    }
    
    path.data = paste0(PATH, FILENAME,".csv")
    result %>% write.csv2(file=path.data)
    return(result)
}

#' Title
#'
#' @param id.group     id columns in data frame with groups labels
#' @param group.labels vector of group labels
#' @param id.columns   vector containing id's of comparing columns 
#' @param type.columns vector containing types: "num","bin" or "cat" of comparing columns 
#' @param data         data frame 
#' @param PATH         path to write result
#' @param FILENAME     filename for result
#' @param dbg          0 - off, 1- onn debug mode 
#'
#' @return
#' @export
#'
#' @examples
cmp.any.group.all = function(id.group, 
                          group.labels = NULL,
                          id.columns,
                          type.columns,
                          data, 
                          PATH = NULL, 
                          FILENAME = NULL, 
                          result = NULL,
                          dbg    = 1
)
{
    require(dplyr)
    
    columns.count = length(type.columns)
    
    #counts = sapply(group.labels, function(x) length(data[group.labels==x,]))
    #ft = matrix(rep(counts, columns.count), nrow = columns.count,byrow = T)
    #colnames(ft) = paste('size of group', group.labels)
    
    #print(ft)
    
    result.real = data.frame()
    combs = combn(group.labels,2,simplify=F)
    pair=character(length = length(combs))
    for(k in seq(combs)){
        pair[k]=paste0(combs[[k]][1],'+',combs[[k]][2])
    }
    
  
    for(i in seq(columns.count))
    {
        
        i.col  = id.columns[i]
        if (dbg == 1){
            cat("\nCurrent column |", colnames(data)[i.col],"| id =", i.col)
        }
        
        dt  = parse.data.frame.with.na( dt = data,
                                           groups.column = id.group,
                                           group.labels  = group.labels,
                                           values.column = i.col)
        
        n.g = character(length(group.labels))
        for(g in seq(group.labels)){
               # 1 - id.group
        # 2 - i.col     
            n.g[g] =  na.count.string(dt[ dt[[1]] == group.labels[g], ][,2])
        }
        
        dt  = parse.data.frame(  dt   = data,
                                    groups.column = id.group,
                                    group.labels  = group.labels,
                                    values.column = i.col )
        
        n.pr = list()
        for(g in seq(group.labels)){
            # 1 - id.group
            # 2 - i.col     
            n.pr[[g]] = dt[dt[[1]] == group.labels[g], ][,2]
        }
        
        if(dbg == 1){
            
            for(g in seq_along(group.labels)){
                if(length(n.pr[[g]]) < 3 ) {
                    cat ("WARNING in group: ", group.labels[g], "column:  ", colnames(data)[i.col], "\n" )
                    cat ("this group is too small")
                    #stop("first group too small")
                    n.pr[[g]] = NA
                }
            }
            
        }
        
        result = data.frame(tmp = 0)        
        for(v in combs){
            
            n1 = v[1]
            n2 = v[2]
            
            x = n.pr[[which(group.labels==n1)]]
            y = n.pr[[which(group.labels==n2)]]
            
            n.x = n.g[which(group.labels == n1)]
            n.y = n.g[which(group.labels == n2)]
            
            if(F){
             print(n1)
            print(n2)
            
            print(x)
            print(y)
            
            print(n.x)
            print(n.y)              
            }
           
            if(type.columns[i] == "num")
            {
                if(
                    (all(is.na(x)) == FALSE) &
                    (all(is.na(y)) == FALSE)
                )
                {    
                    #apply U-test Mana-Uitni
                    res         = wilcox.test(x, y, paired=FALSE, conf.int = TRUE)
                    res.stat    = round(res$statistic, 4)
                    res.p.value = p_value_formatted(res$p.value)
                    
                    res.mu      = -round(res$estimate,  2)
                    res.ci      = paste(c( '[', 
                                           as.character(-round(res$conf.int[2],2)),
                                           '; ',
                                           as.character(-round(res$conf.int[1],2)), 
                                           ']'), collapse=''
                    )
                    res.df = data.frame( n.1      = n.x,
                                         group.1  = paste0(as.character(median.my(x)),
                                                           " ",
                                                           quantile.interval.my(x),
                                                           " ",
                                                           as.character(round(mean(x, na.rm = TRUE), 2)),
                                                           "?",
                                                           as.character(round(sd(x, na.rm = TRUE), 2))
                                         ),
                                         n.2      = n.y,
                                         group.2  = paste0(as.character(median.my(y)),
                                                           " ",
                                                           quantile.interval.my(y),
                                                           " ",
                                                           as.character(round(mean(y, na.rm = TRUE), 2)),
                                                           "?",
                                                           as.character(round(sd(y, na.rm = TRUE), 2))
                                         ),
                                         diff     = paste0( as.character(res.mu)," ", res.ci),
                                         p.value     = as.character(res.p.value),
                                         method      = "U-test Mann-Whitney"
                    )
                }
                if(
                    (all(is.na(x)) == TRUE) & 
                    (all(is.na(y)) == FALSE) 
                )
                {
                    res.df = data.frame( n.1      = n.x,
                                         group.1  = "NA",
                                         n.2      = n.y,
                                         group.2  = paste0( as.character(median.my(y)),
                                                            " ",
                                                            quantile.interval.my(y),
                                                            " ",
                                                            as.character(round(mean(y, na.rm = TRUE), 2)),
                                                            "?",
                                                            as.character(round(sd(y, na.rm = TRUE), 2))
                                         ),
                                         diff     = "-",
                                         p.value  = "-",
                                         method      = "WARNING: ALL Na's values in 1st group"
                    )
                    
                }
                if(
                    (all(is.na(x)) == FALSE) & 
                    (all(is.na(y)) == TRUE) 
                )
                {
                    res.df = data.frame(  n.1      = n.x,
                                          group.1  = paste0(as.character(median.my(x)),
                                                            " ",
                                                            quantile.interval.my(x),
                                                            " ",
                                                            as.character(round(mean(x, na.rm = TRUE), 2)),
                                                            "?",
                                                            as.character(round(sd(x, na.rm = TRUE), 2))
                                          ),
                                          n.2      = n.y,
                                          group.2  = "NA",
                                          diff     = "-",
                                          p.value  = "-",
                                          method      = "WARNING: ALL Na's values in 2nd group"
                    )
                    
                }
                
                if(
                    (all(is.na(x)) == TRUE) & 
                    (all(is.na(y)) == TRUE) 
                )
                {
                    res.df = data.frame( n.1      = n.x,
                                         group.1  = "NA",
                                         n.2      = n.y,
                                         group.2  = "NA",
                                         diff     = "-",
                                         p.value  = "-",
                                         method   = "WARNING: ALL Na's values in both group"
                    )
                    
                }
                
                rownames(res.df)[1] = colnames(data)[i.col]
                #colnames(res.df)[c(2,4)] = c(n1,n2)
                result = cbind(result, res.df)
            }
            if(type.columns[i] == "bin")
            {
                # freq.matrix:
                # -----------------
                #        |  0  1
                # --------=--------
                # group1 |  7 33
                # group2 | 33 78
                #
                
                freq.matrix =  table(dt)
                nrow.f.m = nrow(freq.matrix)
                ncol.f.m = ncol(freq.matrix)
                
                if (dbg == 1){
                    cat("\nfreq matrix         : ", freq.matrix)
                cat("\ncolnames freq matrix: ", colnames(freq.matrix))
                cat("\nrownames freq matrix: ", length(rownames(freq.matrix)),"\n")
                }
                
                prcnt.1     = "NULL"
                prcnt.2     = "NULL"
                res.p.value = "NULL"
                m.name      = "NULL"
                res.diff    = "NULL"
                
                
                
                if(
                    (ncol.f.m == 2) & 
                    (nrow.f.m == 2) &
                    (all(is.na(x)) == FALSE) &
                    (all(is.na(y)) == FALSE)
                ) 
                {    
                    res = fisher.test(freq.matrix)
                    
                    prcnt.1 = bin.my(x, digits=0)
                    prcnt.2 = bin.my(y, digits=0)
                    
                    res.p.value = p_value_formatted(res$p.value)
                    
                    res.diff = ifelse(freq.matrix[1,2] == 0,
                                      "-",
                                      paste0( round(res$estimate,1)," [",
                                              round(res$conf.int[1],1),
                                              "; ",
                                              round(res$conf.int[2],1),
                                              "]"
                                      )
                    )
                    m.name = "Fisher test"
                } 
                if (
                    (
                        (length(colnames(freq.matrix)) != length(rownames(freq.matrix))) & 
                        #(length(rownames(freq.matrix)) == 2) &
                        #(colnames(freq.matrix)[1] == 0)&
                        (all(is.na(x)) == FALSE) &
                        (all(is.na(y)) == FALSE)
                    )
                    #|
                    #(
                    #    (length(colnames(freq.matrix)) == 1) & 
                    #    (length(rownames(freq.matrix)) == 2) &
                        #(colnames(freq.matrix)[1] == 1) &
                    #    (all(is.na(x)) == FALSE) &
                    #    (all(is.na(y)) == FALSE)
                    #)
                    
                )
                {
                    prcnt.1 = bin.my(x, digits=0)
                    prcnt.2 = bin.my(y, digits=0)
                    res.p.value = ">0.999"
                    res.diff    = "-"
                    m.name      = "WARNING: equal cases in both groups"
                }
                
                if(
                    (all(is.na(x)) == TRUE) 
                    |
                    (all(is.na(y)) == TRUE)
                )
                {
                    prcnt.1 = bin.my(x, digits=0)
                    prcnt.2 = bin.my(y, digits=0)
                    res.p.value = " - "
                    res.diff    = " - "
                    m.name      = "WARNING: All NA's !"
                }
                
                
                res.df = data.frame( n.1      = n.x,
                                     group.1  = prcnt.1,
                                     n.2      = n.y,
                                     group.2  = prcnt.2,
                                     diff     = res.diff,
                                     p.value  = as.character(res.p.value),
                                     method   = m.name
                )
                rownames(res.df)[1] = colnames(data)[i.col]
                #colnames(res.df) = c(n1,n2)
                result = cbind(result, res.df)
            }
            if(type.columns[i] == "cat")
            {
                
                if(
                    (all(is.na(x)) == FALSE) &
                    (all(is.na(y)) == FALSE)
                )
                {
                    n.matrix    = table(dt)
                    prop.matrix = round((prop.table(n.matrix, 1) * 100), 1)
                    
                    # the code below concatenates rows of n.matrix and prop.matrix in the following format:
                    # (this is example of group.1 cell content in data.frame)
                    #  ----------------
                    # | 0 - 35 (31.2%) |
                    # | 1 - 76 (67.9%) |
                    # | 2 - 0 (0%)     |
                    # | 3 - 1 (0.9%)   |
                    #  ----------------
                    # where 0, 1, 2, 3 - category names/IDs
                    # 35, 76, 0, 1 number of peoples in each category
                    # % - how much peoples (in percentages) in each category (100% - total number of peoples in group = 35+76+0+1 = 112)
                    res1 = paste(paste0(colnames(n.matrix), " - ", as.numeric(n.matrix[1,]), " (", as.numeric(prop.matrix[1,]), "%)"), collapse = "\n")
                    res2 = paste(paste0(colnames(n.matrix), " - ", as.numeric(n.matrix[2,]), " (", as.numeric(prop.matrix[2,]), "%)"), collapse = "\n")
                    
                    
                    if (
                        dim(n.matrix)[2] > 1
                    )
                    {
                        res = fisher.test(n.matrix)
                        res.p.value = as.character(p_value_formatted(res$p.value))
                        res.method  = "Fisher test"
                    }
                    if (
                        dim(n.matrix)[2] == 1
                    )
                    {
                        res.p.value = " - "
                        res.method  = "WARNING: only one category!"
                    }
                    
                }
                
                if(
                    (all(is.na(x)) == TRUE) &
                    (all(is.na(y)) == FALSE)
                )
                {
                    n.matrix    = table(y)
                    prop.matrix = round((prop.table(n.matrix, 1) * 100), 1)
                    
                    res1 = " - "
                    res2 = paste(paste0(names(n.matrix), " - ", as.numeric(n.matrix[1]), " (", as.numeric(prop.matrix[1]), "%)"), collapse = "\n")
                    res.p.value = " - "
                    res.method  = "WARNING: NA's values only in 1st group!"
                    
                    rownames(n.matrix)[1] = group.labels[1] # for right joining to result
                }
                if(
                    (all(is.na(x)) == FALSE) &
                    (all(is.na(y)) == TRUE)
                )
                {
                    n.matrix    = table(x)
                    prop.matrix = round((prop.table(n.matrix, 1) * 100), 1)
                    
                    res1 = paste(paste0(names(n.matrix), " - ", as.numeric(n.matrix[1]), " (", as.numeric(prop.matrix[1]), "%)"), collapse = "\n")
                    res2 = " - "
                    res.p.value = " - "
                    res.method  = "WARNING: NA's values only in 2nd group!"
                    
                    rownames(n.matrix)[1] = group.labels[1] # for right joining to result
                }
                if(
                    (all(is.na(x)) == TRUE) &
                    (all(is.na(y)) == TRUE)
                )
                {
                    res1 = " - "
                    res2 = " - "
                    res.p.value = " - "
                    res.method  = "WARNING: NA's values only in both groups"
                    
                    n.matrix = NULL
                    rownames(n.matrix)[1] = group.labels[1] # for right joining to result
                    
                }
                
                res.df = data.frame( n.1      = n.x,
                                     group.1  = res1,
                                     n.2      = n.y,
                                     group.2  = res2,
                                     diff     = " ",
                                     p.value  = res.p.value,
                                     method   = res.method
                )
                rownames(res.df)[1] = colnames(data)[i.col]
                
                # n.matrix rownames are ordered in the same way as initial data, not order we set for group.labels !!!
                if (group.labels[1] == rownames(n.matrix)[1]) {
                    #colnames(res.df)[1] = n1
                    #colnames(res.df)[2] = n2
                } else {
                    #colnames(res.df)[1] = n2
                    #colnames(res.df)[2] = n1
                }
                
                result = cbind(result, res.df)
            } 
        }
        
        #print(result.real)
        result.real = rbind(result.real,result)
        
    }
    
    path.data = paste0(PATH,'/', FILENAME,".csv")
    
    
    result.real = result.real[,-c(1)]
    colnames(result.real)[1:ncol(result.real) %% 7 == 5]= paste('diff of',pair)
    colnames(result.real)[1:ncol(result.real) %% 7 == 6]= paste('p.value of',pair)
    for(i in seq(combs)){
        colnames(result.real)[(i-1)*7+c(1,3)]=paste('count of',combs[[i]])
        colnames(result.real)[(i-1)*7+c(2,4)]=paste('stat of',combs[[i]])
    }
    method=result.real$method
    result.real = cbind(result.real[unique(colnames(result.real))] %>% select(!starts_with('method')),method)
    
    result.real %>% write.csv2(file=path.data)
    return(result.real)
}


#' Title
#'
#' @param id.group     id columns in data frame with groups labels
#' @param group.labels vector of group labels
#' @param columns.list list containing id's and types: "num","bin" or "cat" of comparing columns 
#' @param data         data frame 
#' @param PATH         path to write result
#' @param FILENAME     filename for result
#'
#' @return
#' @export
#'
#' @examples
cmp.any.group.all.vars = function(id.group,
                               group.labels = NULL,
                               columns.list = NULL,
                               data, 
                               PATH = NULL, 
                               FILENAME = NULL, 
                               result = NULL
)
{
    x = unlist(columns.list, recursive = TRUE)
    x.length = length(x)
    id.columns   = x[seq(1,x.length,2)]
    type.columns = x[seq(2,x.length,2)]
    id.columns   = as.integer(id.columns)
    type.columns = as.character(type.columns)
    
    res = cmp.any.group.all(id.group     = id.group,
                                   group.labels = group.labels,
                                   id.columns   = id.columns,
                                   type.columns = type.columns,
                                   data         = data, 
                                   PATH         = PATH, 
                                   FILENAME     = FILENAME, 
                                   result       = NULL
    )
    return (res)
}


###################
